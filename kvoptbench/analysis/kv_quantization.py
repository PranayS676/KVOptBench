"""Compare baseline and quantized KV cache experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kvoptbench.analysis.comparison_stats import (
    add_group_metric_stats,
    repeated_comparison_columns,
    repeated_comparison_fields,
)

QUALITY_REGRESSION_THRESHOLD = -0.05
LATENCY_REGRESSION_PCT = 20.0
MEMORY_IMPROVEMENT_PCT = -5.0
THROUGHPUT_IMPROVEMENT_PCT = 5.0
SUCCESS_RATE_FLOOR = 0.95
REPEATED_METRIC_SPECS = [
    ("baseline_ttft_ms", "quantized_ttft_ms", "ttft_ms"),
    ("baseline_e2e_latency_ms", "quantized_e2e_latency_ms", "e2e_latency_ms"),
    (
        "baseline_output_tokens_per_second",
        "quantized_output_tokens_per_second",
        "output_tokens_per_second",
    ),
    ("baseline_quality_score", "quantized_quality_score", "quality_score"),
    ("baseline_gpu_memory_peak_gb", "quantized_gpu_memory_peak_gb", "gpu_memory_peak_gb"),
]

KV_QUANTIZATION_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "workload",
    "context_token_bucket",
    "baseline_strategy",
    "quantized_strategy",
    "baseline_ttft_ms_p50",
    "quantized_ttft_ms_p50",
    "ttft_delta_pct",
    "baseline_e2e_latency_ms_p50",
    "quantized_e2e_latency_ms_p50",
    "e2e_delta_pct",
    "baseline_output_tokens_per_second_mean",
    "quantized_output_tokens_per_second_mean",
    "throughput_delta_pct",
    "baseline_quality_score_mean",
    "quantized_quality_score_mean",
    "quality_delta",
    "baseline_gpu_memory_peak_gb",
    "quantized_gpu_memory_peak_gb",
    "memory_delta_pct",
    "missing_metrics",
    "requests",
    "baseline_success_rate",
    "quantized_success_rate",
    "quantization_interpretation",
    *repeated_comparison_columns(REPEATED_METRIC_SPECS),
]


def compare_kv_quantization_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write baseline-vs-quantized comparison CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_kv_quantization_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_kv_quantization_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline and kv_fp8 strategy results by workload and context bucket."""
    if frame.empty:
        return pd.DataFrame(columns=KV_QUANTIZATION_COLUMNS)

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
        "gpu_memory_peak_gb",
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
    for column in [
        "context_token_bucket",
        "ttft_ms",
        "e2e_latency_ms",
        "output_tokens_per_second",
        "quality_score",
        "gpu_memory_peak_gb",
    ]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    enriched["success"] = enriched["success"].fillna(False).astype(bool)

    aggregated = _aggregate_by_strategy(enriched)
    if aggregated.empty:
        return pd.DataFrame(columns=KV_QUANTIZATION_COLUMNS)

    rows: list[dict[str, Any]] = []
    compare_cols = ["provider", "engine", "model_id", "workload", "context_token_bucket"]
    for keys, group in aggregated.groupby(compare_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_group = dict(zip(compare_cols, keys, strict=True))
        baseline = _strategy_row(group, "baseline")
        quantized = _strategy_row(group, "kv_fp8")
        if baseline is None or quantized is None:
            continue

        row = _build_comparison_row(by_group, baseline, quantized)
        rows.append(row)

    result = pd.DataFrame(rows, columns=KV_QUANTIZATION_COLUMNS)
    if result.empty:
        return pd.DataFrame(columns=KV_QUANTIZATION_COLUMNS)
    return result.sort_values(
        ["provider", "engine", "model_id", "workload", "context_token_bucket"]
    ).reset_index(drop=True)


def interpret_quantization_result(
    *,
    quality_delta: float | None,
    e2e_delta_pct: float | None,
    ttft_delta_pct: float | None,
    throughput_delta_pct: float | None,
    memory_delta_pct: float | None,
    quantized_success_rate: float | None,
) -> str:
    """Interpret quantization tradeoffs without inventing unavailable telemetry."""
    if quantized_success_rate is None or quantized_success_rate < SUCCESS_RATE_FLOOR:
        return "insufficient_quantization_signal"
    if quality_delta is not None and quality_delta < QUALITY_REGRESSION_THRESHOLD:
        return "quality_regression"
    if (e2e_delta_pct is not None and e2e_delta_pct > LATENCY_REGRESSION_PCT) or (
        ttft_delta_pct is not None and ttft_delta_pct > LATENCY_REGRESSION_PCT
    ):
        return "latency_regression"
    memory_improved = memory_delta_pct is not None and memory_delta_pct <= MEMORY_IMPROVEMENT_PCT
    throughput_improved = (
        throughput_delta_pct is not None and throughput_delta_pct >= THROUGHPUT_IMPROVEMENT_PCT
    )
    if memory_improved or throughput_improved:
        return "quantization_promising"
    if (
        quality_delta is None
        and e2e_delta_pct is None
        and throughput_delta_pct is None
        and memory_delta_pct is None
    ):
        return "insufficient_quantization_signal"
    return "no_observed_benefit"


def _aggregate_by_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["provider", "engine", "model_id", "workload", "context_token_bucket", "strategy"]
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
                "gpu_memory_peak_gb": _mean(group["gpu_memory_peak_gb"].dropna()),
                "requests": requests,
                "success_rate": round(successes / requests, 4) if requests else None,
                "missing_metrics": _missing_metrics(group),
            }
        )
        add_group_metric_stats(
            row,
            group,
            [
                "ttft_ms",
                "e2e_latency_ms",
                "output_tokens_per_second",
                "quality_score",
                "gpu_memory_peak_gb",
            ],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_comparison_row(
    by_group: dict[str, Any], baseline: pd.Series, quantized: pd.Series
) -> dict[str, Any]:
    baseline_ttft = _to_float(baseline.get("ttft_ms_p50"))
    quantized_ttft = _to_float(quantized.get("ttft_ms_p50"))
    baseline_e2e = _to_float(baseline.get("e2e_latency_ms_p50"))
    quantized_e2e = _to_float(quantized.get("e2e_latency_ms_p50"))
    baseline_tps = _to_float(baseline.get("output_tokens_per_second_mean"))
    quantized_tps = _to_float(quantized.get("output_tokens_per_second_mean"))
    baseline_quality = _to_float(baseline.get("quality_score_mean"))
    quantized_quality = _to_float(quantized.get("quality_score_mean"))
    baseline_memory = _to_float(baseline.get("gpu_memory_peak_gb"))
    quantized_memory = _to_float(quantized.get("gpu_memory_peak_gb"))
    quality_delta = _subtract(quantized_quality, baseline_quality)
    ttft_delta_pct = _delta_pct(quantized_ttft, baseline_ttft)
    e2e_delta_pct = _delta_pct(quantized_e2e, baseline_e2e)
    throughput_delta_pct = _delta_pct(quantized_tps, baseline_tps)
    memory_delta_pct = _delta_pct(quantized_memory, baseline_memory)
    missing_metrics = _join_missing(
        baseline.get("missing_metrics"),
        quantized.get("missing_metrics"),
    )
    quantized_success_rate = _to_float(quantized.get("success_rate"))

    row = {
        **by_group,
        "context_token_bucket": _nullable_int(by_group.get("context_token_bucket")),
        "baseline_strategy": str(baseline.get("strategy")),
        "quantized_strategy": str(quantized.get("strategy")),
        "baseline_ttft_ms_p50": baseline_ttft,
        "quantized_ttft_ms_p50": quantized_ttft,
        "ttft_delta_pct": ttft_delta_pct,
        "baseline_e2e_latency_ms_p50": baseline_e2e,
        "quantized_e2e_latency_ms_p50": quantized_e2e,
        "e2e_delta_pct": e2e_delta_pct,
        "baseline_output_tokens_per_second_mean": baseline_tps,
        "quantized_output_tokens_per_second_mean": quantized_tps,
        "throughput_delta_pct": throughput_delta_pct,
        "baseline_quality_score_mean": baseline_quality,
        "quantized_quality_score_mean": quantized_quality,
        "quality_delta": quality_delta,
        "baseline_gpu_memory_peak_gb": baseline_memory,
        "quantized_gpu_memory_peak_gb": quantized_memory,
        "memory_delta_pct": memory_delta_pct,
        "missing_metrics": missing_metrics,
        "requests": int(baseline.get("requests", 0)) + int(quantized.get("requests", 0)),
        "baseline_success_rate": _to_float(baseline.get("success_rate")),
        "quantized_success_rate": quantized_success_rate,
        "quantization_interpretation": interpret_quantization_result(
            quality_delta=quality_delta,
            e2e_delta_pct=e2e_delta_pct,
            ttft_delta_pct=ttft_delta_pct,
            throughput_delta_pct=throughput_delta_pct,
            memory_delta_pct=memory_delta_pct,
            quantized_success_rate=quantized_success_rate,
        ),
    }
    for source_metric, baseline_prefix, candidate_prefix, effect_prefix in [
        ("ttft_ms", "baseline_ttft_ms", "quantized_ttft_ms", "ttft_ms"),
        ("e2e_latency_ms", "baseline_e2e_latency_ms", "quantized_e2e_latency_ms", "e2e_latency_ms"),
        (
            "output_tokens_per_second",
            "baseline_output_tokens_per_second",
            "quantized_output_tokens_per_second",
            "output_tokens_per_second",
        ),
        ("quality_score", "baseline_quality_score", "quantized_quality_score", "quality_score"),
        (
            "gpu_memory_peak_gb",
            "baseline_gpu_memory_peak_gb",
            "quantized_gpu_memory_peak_gb",
            "gpu_memory_peak_gb",
        ),
    ]:
        row.update(
            repeated_comparison_fields(
                baseline,
                quantized,
                source_metric=source_metric,
                baseline_prefix=baseline_prefix,
                candidate_prefix=candidate_prefix,
                effect_prefix=effect_prefix,
            )
        )
    return row


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


def _join_missing(*values) -> str:
    missing: set[str] = set()
    for value in values:
        if isinstance(value, str) and value:
            missing.update(value.split(";"))
    return ";".join(sorted(item for item in missing if item))


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 3)


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
    parser = argparse.ArgumentParser(description="Compare KV quantization JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_kv_quantization_results(input_path=args.input, output_path=args.output)
    print(f"Wrote KV quantization comparison to {output}")


if __name__ == "__main__":
    main()
