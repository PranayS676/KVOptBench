"""Analyze prefill vs decode pressure from request-level result JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

TTFT_BOUNDARY_MS = 300.0
DECODE_BOUNDARY_MS = 30.0

PREFILL_DECODE_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "strategy",
    "input_token_bucket",
    "output_token_bucket",
    "expected_bottleneck",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "tpot_ms_mean",
    "itl_ms_mean",
    "output_tokens_per_second_mean",
    "bottleneck_classification",
    "missing_metrics",
    "requests",
    "success_rate",
]


def compare_prefill_decode_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write prefill/decode comparison CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_prefill_decode_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_prefill_decode_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prefill/decode timing metrics by input and output token bucket."""
    if frame.empty:
        return pd.DataFrame(columns=PREFILL_DECODE_COLUMNS)

    enriched = frame.copy()
    for column in [
        "provider",
        "engine",
        "model_id",
        "strategy",
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "output_tokens_per_second",
        "success",
    ]:
        if column not in enriched:
            enriched[column] = None
    if "metadata" not in enriched:
        enriched["metadata"] = None
    if "missing_metrics" not in enriched:
        enriched["missing_metrics"] = None

    enriched["input_token_bucket"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "input_token_bucket"), axis=1
    )
    enriched["output_token_bucket"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "output_token_bucket"), axis=1
    )
    enriched["expected_bottleneck"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "expected_bottleneck"), axis=1
    )
    enriched = enriched[
        enriched["input_token_bucket"].notna() & enriched["output_token_bucket"].notna()
    ].copy()
    if enriched.empty:
        return pd.DataFrame(columns=PREFILL_DECODE_COLUMNS)

    for column in [
        "input_token_bucket",
        "output_token_bucket",
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "output_tokens_per_second",
    ]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    enriched["success"] = enriched["success"].fillna(False).astype(bool)

    rows: list[dict[str, Any]] = []
    group_cols = [
        "provider",
        "engine",
        "model_id",
        "strategy",
        "input_token_bucket",
        "output_token_bucket",
        "expected_bottleneck",
    ]
    for keys, group in enriched.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        ttft = group["ttft_ms"].dropna()
        tpot = group["tpot_ms"].dropna()
        itl = group["itl_ms"].dropna()
        output_tps = group["output_tokens_per_second"].dropna()
        requests = int(len(group))
        successes = int(group["success"].sum())
        ttft_p50 = _quantile(ttft, 0.50)
        ttft_p95 = _quantile(ttft, 0.95)
        tpot_mean = _mean(tpot)
        itl_mean = _mean(itl)
        output_tps_mean = _mean(output_tps)

        row.update(
            {
                "input_token_bucket": int(row["input_token_bucket"]),
                "output_token_bucket": int(row["output_token_bucket"]),
                "ttft_ms_p50": ttft_p50,
                "ttft_ms_p95": ttft_p95,
                "tpot_ms_mean": tpot_mean,
                "itl_ms_mean": itl_mean,
                "output_tokens_per_second_mean": output_tps_mean,
                "bottleneck_classification": classify_bottleneck(
                    ttft_ms=ttft_p50,
                    tpot_ms=tpot_mean,
                    itl_ms=itl_mean,
                ),
                "missing_metrics": _missing_metrics(group),
                "requests": requests,
                "success_rate": round(successes / requests, 4) if requests else None,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows, columns=PREFILL_DECODE_COLUMNS)
    return result.sort_values(
        ["provider", "engine", "model_id", "strategy", "input_token_bucket", "output_token_bucket"]
    ).reset_index(drop=True)


def classify_bottleneck(
    *,
    ttft_ms: float | None,
    tpot_ms: float | None,
    itl_ms: float | None,
) -> str:
    """Classify bottleneck from observed timing metrics."""
    decode_ms = tpot_ms if tpot_ms is not None else itl_ms
    if ttft_ms is None or decode_ms is None:
        return "insufficient_prefill_decode_signal"
    high_prefill = ttft_ms >= TTFT_BOUNDARY_MS
    high_decode = decode_ms >= DECODE_BOUNDARY_MS
    if high_prefill and high_decode:
        return "mixed"
    if high_prefill:
        return "prefill_bound"
    if high_decode:
        return "decode_bound"
    return "balanced"


def _jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.jsonl"))


def _read_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in _jsonl_files(input_path):
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSONL result rows found under {input_path}")
    return rows


def _workload_metadata_value(row, key: str):
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        workload_metadata = metadata.get("workload_metadata")
        if isinstance(workload_metadata, dict) and workload_metadata.get(key) is not None:
            return workload_metadata[key]
    if key == "input_token_bucket":
        return row.get("target_input_tokens")
    if key == "output_token_bucket":
        return row.get("target_output_tokens")
    return None


def _quantile(values: pd.Series, quantile: float) -> float | None:
    if values.empty:
        return None
    return round(float(values.quantile(quantile)), 3)


def _mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.mean()), 3)


def _missing_metrics(group: pd.DataFrame) -> str:
    missing: set[str] = set()
    for value in group.get("missing_metrics", []):
        if isinstance(value, list):
            missing.update(str(item) for item in value)
        elif isinstance(value, str) and value:
            missing.update(value.split(";"))
    return ";".join(sorted(item for item in missing if item))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prefill/decode JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_prefill_decode_results(input_path=args.input, output_path=args.output)
    print(f"Wrote prefill/decode comparison to {output}")


if __name__ == "__main__":
    main()
