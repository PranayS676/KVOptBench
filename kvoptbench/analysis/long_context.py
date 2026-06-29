"""Analyze long-context pressure from request-level result JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

TTFT_GROWTH_MULTIPLE = 2.0
TTFT_GROWTH_MIN_DELTA_MS = 500.0
THROUGHPUT_DROP_RATIO = 0.5
FAILURE_PRESSURE_SUCCESS_RATE = 0.95

LONG_CONTEXT_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "strategy",
    "context_token_bucket",
    "pressure_level",
    "expected_pressure",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "e2e_latency_ms_p50",
    "e2e_latency_ms_p95",
    "input_tokens_per_second_mean",
    "output_tokens_per_second_mean",
    "pressure_classification",
    "missing_metrics",
    "requests",
    "success_rate",
    "error_rate",
]


def compare_long_context_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write long-context comparison CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_long_context_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_long_context_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-context timing and throughput metrics by context bucket."""
    if frame.empty:
        return pd.DataFrame(columns=LONG_CONTEXT_COLUMNS)

    enriched = frame.copy()
    for column in [
        "provider",
        "engine",
        "model_id",
        "strategy",
        "ttft_ms",
        "e2e_latency_ms",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "success",
    ]:
        if column not in enriched:
            enriched[column] = None
    if "metadata" not in enriched:
        enriched["metadata"] = None
    if "missing_metrics" not in enriched:
        enriched["missing_metrics"] = None

    enriched["context_token_bucket"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "context_token_bucket"), axis=1
    )
    enriched["pressure_level"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "pressure_level"), axis=1
    )
    enriched["expected_pressure"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "expected_pressure"), axis=1
    )
    enriched = enriched[enriched["context_token_bucket"].notna()].copy()
    if enriched.empty:
        return pd.DataFrame(columns=LONG_CONTEXT_COLUMNS)

    for column in [
        "context_token_bucket",
        "ttft_ms",
        "e2e_latency_ms",
        "input_tokens_per_second",
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
        "context_token_bucket",
        "pressure_level",
        "expected_pressure",
    ]
    for keys, group in enriched.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        ttft = group["ttft_ms"].dropna()
        e2e = group["e2e_latency_ms"].dropna()
        input_tps = group["input_tokens_per_second"].dropna()
        output_tps = group["output_tokens_per_second"].dropna()
        requests = int(len(group))
        successes = int(group["success"].sum())
        success_rate = round(successes / requests, 4) if requests else None
        error_rate = round(1.0 - success_rate, 4) if success_rate is not None else None

        row.update(
            {
                "context_token_bucket": int(row["context_token_bucket"]),
                "ttft_ms_p50": _quantile(ttft, 0.50),
                "ttft_ms_p95": _quantile(ttft, 0.95),
                "e2e_latency_ms_p50": _quantile(e2e, 0.50),
                "e2e_latency_ms_p95": _quantile(e2e, 0.95),
                "input_tokens_per_second_mean": _mean(input_tps),
                "output_tokens_per_second_mean": _mean(output_tps),
                "pressure_classification": None,
                "missing_metrics": _missing_metrics(group),
                "requests": requests,
                "success_rate": success_rate,
                "error_rate": error_rate,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows, columns=LONG_CONTEXT_COLUMNS)
    result = result.sort_values(
        ["provider", "engine", "model_id", "strategy", "context_token_bucket"]
    ).reset_index(drop=True)
    return _classify_result_frame(result)


def classify_pressure(
    *,
    success_rate: float | None,
    ttft_ms: float | None,
    baseline_ttft_ms: float | None,
    output_tokens_per_second: float | None,
    baseline_output_tokens_per_second: float | None,
) -> str:
    """Classify long-context pressure using only observed request metrics."""
    if success_rate is not None and success_rate < FAILURE_PRESSURE_SUCCESS_RATE:
        return "failure_pressure"
    if ttft_ms is None and output_tokens_per_second is None:
        return "insufficient_long_context_signal"
    if baseline_ttft_ms is not None and ttft_ms is not None:
        grows_by_multiple = ttft_ms >= baseline_ttft_ms * TTFT_GROWTH_MULTIPLE
        grows_by_delta = (ttft_ms - baseline_ttft_ms) >= TTFT_GROWTH_MIN_DELTA_MS
        if grows_by_multiple and grows_by_delta:
            return "prefill_latency_growth"
    if (
        baseline_output_tokens_per_second is not None
        and output_tokens_per_second is not None
        and output_tokens_per_second <= baseline_output_tokens_per_second * THROUGHPUT_DROP_RATIO
    ):
        return "throughput_degradation"
    return "stable_long_context"


def _classify_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    classified = frame.copy()
    group_cols = ["provider", "engine", "model_id", "strategy"]
    for _, index in classified.groupby(group_cols, dropna=False).groups.items():
        group = classified.loc[index].sort_values("context_token_bucket")
        baseline_ttft = _first_numeric(group["ttft_ms_p50"])
        baseline_output_tps = _first_numeric(group["output_tokens_per_second_mean"])
        for row_index, row in group.iterrows():
            classified.loc[row_index, "pressure_classification"] = classify_pressure(
                success_rate=_to_float(row.get("success_rate")),
                ttft_ms=_to_float(row.get("ttft_ms_p50")),
                baseline_ttft_ms=baseline_ttft,
                output_tokens_per_second=_to_float(row.get("output_tokens_per_second_mean")),
                baseline_output_tokens_per_second=baseline_output_tps,
            )
    return classified[LONG_CONTEXT_COLUMNS]


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
    if key == "context_token_bucket":
        return row.get("target_input_tokens")
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


def _first_numeric(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])


def _to_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare long-context JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_long_context_results(input_path=args.input, output_path=args.output)
    print(f"Wrote long-context comparison to {output}")


if __name__ == "__main__":
    main()
