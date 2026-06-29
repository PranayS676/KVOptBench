"""Compare baseline and speculative decoding experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

QUALITY_REGRESSION_THRESHOLD = -0.05
LATENCY_REGRESSION_PCT = 20.0
LATENCY_IMPROVEMENT_PCT = -5.0
THROUGHPUT_IMPROVEMENT_PCT = 5.0
SUCCESS_RATE_FLOOR = 0.95

SPECULATIVE_DECODING_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "workload",
    "output_token_bucket",
    "baseline_strategy",
    "speculative_strategy",
    "baseline_ttft_ms_p50",
    "speculative_ttft_ms_p50",
    "ttft_delta_pct",
    "baseline_e2e_latency_ms_p50",
    "speculative_e2e_latency_ms_p50",
    "e2e_delta_pct",
    "baseline_output_tokens_per_second_mean",
    "speculative_output_tokens_per_second_mean",
    "throughput_delta_pct",
    "baseline_quality_score_mean",
    "speculative_quality_score_mean",
    "quality_delta",
    "missing_metrics",
    "requests",
    "baseline_success_rate",
    "speculative_success_rate",
    "success_rate_delta",
    "speculative_decoding_interpretation",
]


def compare_speculative_decoding_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write baseline-vs-speculative comparison CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_speculative_decoding_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_speculative_decoding_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline and speculative decoding results by output token bucket."""
    if frame.empty:
        return pd.DataFrame(columns=SPECULATIVE_DECODING_COLUMNS)

    enriched = frame.copy()
    for column in [
        "provider",
        "engine",
        "model_id",
        "workload",
        "strategy",
        "ttft_ms",
        "e2e_latency_ms",
        "output_tokens_per_second",
        "quality_score",
        "success",
    ]:
        if column not in enriched:
            enriched[column] = None
    if "metadata" not in enriched:
        enriched["metadata"] = None
    if "missing_metrics" not in enriched:
        enriched["missing_metrics"] = None

    enriched["output_token_bucket"] = enriched.apply(
        lambda row: _workload_metadata_value(row, "output_token_bucket"), axis=1
    )
    for column in [
        "output_token_bucket",
        "ttft_ms",
        "e2e_latency_ms",
        "output_tokens_per_second",
        "quality_score",
    ]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    enriched["success"] = enriched["success"].fillna(False).astype(bool)

    aggregated = _aggregate_by_strategy(enriched)
    if aggregated.empty:
        return pd.DataFrame(columns=SPECULATIVE_DECODING_COLUMNS)

    rows: list[dict[str, Any]] = []
    compare_cols = ["provider", "engine", "model_id", "workload", "output_token_bucket"]
    for keys, group in aggregated.groupby(compare_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_group = dict(zip(compare_cols, keys, strict=True))
        baseline = _strategy_row(group, "baseline")
        speculative = _strategy_row(group, "speculative_decoding")
        if baseline is None or speculative is None:
            continue

        rows.append(_build_comparison_row(by_group, baseline, speculative))

    result = pd.DataFrame(rows, columns=SPECULATIVE_DECODING_COLUMNS)
    if result.empty:
        return pd.DataFrame(columns=SPECULATIVE_DECODING_COLUMNS)
    return result.sort_values(
        ["provider", "engine", "model_id", "workload", "output_token_bucket"]
    ).reset_index(drop=True)


def interpret_speculative_decoding_result(
    *,
    quality_delta: float | None,
    e2e_delta_pct: float | None,
    ttft_delta_pct: float | None,
    throughput_delta_pct: float | None,
    speculative_success_rate: float | None,
) -> str:
    """Interpret speculative decoding tradeoffs without inventing unavailable telemetry."""
    if speculative_success_rate is None or speculative_success_rate < SUCCESS_RATE_FLOOR:
        return "insufficient_speculative_decoding_signal"
    if quality_delta is not None and quality_delta < QUALITY_REGRESSION_THRESHOLD:
        return "quality_regression"
    if (e2e_delta_pct is not None and e2e_delta_pct > LATENCY_REGRESSION_PCT) or (
        ttft_delta_pct is not None and ttft_delta_pct > LATENCY_REGRESSION_PCT
    ):
        return "latency_regression"
    latency_improved = e2e_delta_pct is not None and e2e_delta_pct <= LATENCY_IMPROVEMENT_PCT
    throughput_improved = (
        throughput_delta_pct is not None and throughput_delta_pct >= THROUGHPUT_IMPROVEMENT_PCT
    )
    if latency_improved or throughput_improved:
        return "speculative_decoding_promising"
    if (
        quality_delta is None
        and e2e_delta_pct is None
        and ttft_delta_pct is None
        and throughput_delta_pct is None
    ):
        return "insufficient_speculative_decoding_signal"
    return "no_observed_benefit"


def _aggregate_by_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["provider", "engine", "model_id", "workload", "output_token_bucket", "strategy"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        requests = int(len(group))
        successes = int(group["success"].sum())
        row.update(
            {
                "ttft_ms_p50": _quantile(group["ttft_ms"].dropna(), 0.50),
                "e2e_latency_ms_p50": _quantile(group["e2e_latency_ms"].dropna(), 0.50),
                "output_tokens_per_second_mean": _mean(
                    group["output_tokens_per_second"].dropna()
                ),
                "quality_score_mean": _mean(group["quality_score"].dropna()),
                "requests": requests,
                "success_rate": round(successes / requests, 4) if requests else None,
                "missing_metrics": _missing_metrics(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_comparison_row(
    by_group: dict[str, Any], baseline: pd.Series, speculative: pd.Series
) -> dict[str, Any]:
    baseline_ttft = _to_float(baseline.get("ttft_ms_p50"))
    speculative_ttft = _to_float(speculative.get("ttft_ms_p50"))
    baseline_e2e = _to_float(baseline.get("e2e_latency_ms_p50"))
    speculative_e2e = _to_float(speculative.get("e2e_latency_ms_p50"))
    baseline_tps = _to_float(baseline.get("output_tokens_per_second_mean"))
    speculative_tps = _to_float(speculative.get("output_tokens_per_second_mean"))
    baseline_quality = _to_float(baseline.get("quality_score_mean"))
    speculative_quality = _to_float(speculative.get("quality_score_mean"))
    baseline_success_rate = _to_float(baseline.get("success_rate"))
    speculative_success_rate = _to_float(speculative.get("success_rate"))
    quality_delta = _subtract(speculative_quality, baseline_quality)
    ttft_delta_pct = _delta_pct(speculative_ttft, baseline_ttft)
    e2e_delta_pct = _delta_pct(speculative_e2e, baseline_e2e)
    throughput_delta_pct = _delta_pct(speculative_tps, baseline_tps)
    success_rate_delta = _subtract(speculative_success_rate, baseline_success_rate)

    return {
        **by_group,
        "output_token_bucket": _nullable_int(by_group.get("output_token_bucket")),
        "baseline_strategy": str(baseline.get("strategy")),
        "speculative_strategy": str(speculative.get("strategy")),
        "baseline_ttft_ms_p50": baseline_ttft,
        "speculative_ttft_ms_p50": speculative_ttft,
        "ttft_delta_pct": ttft_delta_pct,
        "baseline_e2e_latency_ms_p50": baseline_e2e,
        "speculative_e2e_latency_ms_p50": speculative_e2e,
        "e2e_delta_pct": e2e_delta_pct,
        "baseline_output_tokens_per_second_mean": baseline_tps,
        "speculative_output_tokens_per_second_mean": speculative_tps,
        "throughput_delta_pct": throughput_delta_pct,
        "baseline_quality_score_mean": baseline_quality,
        "speculative_quality_score_mean": speculative_quality,
        "quality_delta": quality_delta,
        "missing_metrics": _join_missing(
            baseline.get("missing_metrics"),
            speculative.get("missing_metrics"),
        ),
        "requests": int(baseline.get("requests", 0)) + int(speculative.get("requests", 0)),
        "baseline_success_rate": baseline_success_rate,
        "speculative_success_rate": speculative_success_rate,
        "success_rate_delta": success_rate_delta,
        "speculative_decoding_interpretation": interpret_speculative_decoding_result(
            quality_delta=quality_delta,
            e2e_delta_pct=e2e_delta_pct,
            ttft_delta_pct=ttft_delta_pct,
            throughput_delta_pct=throughput_delta_pct,
            speculative_success_rate=speculative_success_rate,
        ),
    }


def _strategy_row(group: pd.DataFrame, strategy: str) -> pd.Series | None:
    matches = group[group["strategy"] == strategy]
    if matches.empty:
        return None
    return matches.iloc[0]


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


def _join_missing(*values) -> str:
    missing: set[str] = set()
    for value in values:
        if isinstance(value, str) and value:
            missing.update(value.split(";"))
    return ";".join(sorted(item for item in missing if item))


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 4)


def _delta_pct(new_value: float | None, baseline_value: float | None) -> float | None:
    if new_value is None or baseline_value is None or baseline_value == 0:
        return None
    return round(((new_value - baseline_value) / baseline_value) * 100.0, 3)


def _to_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _nullable_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare speculative decoding JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_speculative_decoding_results(input_path=args.input, output_path=args.output)
    print(f"Wrote speculative decoding comparison to {output}")


if __name__ == "__main__":
    main()
