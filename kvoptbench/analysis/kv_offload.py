"""Compare baseline and KV offload experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

QUALITY_REGRESSION_THRESHOLD = -0.05
LATENCY_REGRESSION_PCT = 20.0
MEMORY_IMPROVEMENT_PCT = -5.0
THROUGHPUT_IMPROVEMENT_PCT = 5.0
SUCCESS_RATE_FLOOR = 0.95

KV_OFFLOAD_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "workload",
    "context_token_bucket",
    "baseline_strategy",
    "offload_strategy",
    "baseline_ttft_ms_p50",
    "offload_ttft_ms_p50",
    "ttft_delta_pct",
    "baseline_e2e_latency_ms_p50",
    "offload_e2e_latency_ms_p50",
    "e2e_delta_pct",
    "baseline_output_tokens_per_second_mean",
    "offload_output_tokens_per_second_mean",
    "throughput_delta_pct",
    "baseline_quality_score_mean",
    "offload_quality_score_mean",
    "quality_delta",
    "baseline_gpu_memory_peak_gb",
    "offload_gpu_memory_peak_gb",
    "memory_delta_pct",
    "missing_metrics",
    "requests",
    "baseline_success_rate",
    "offload_success_rate",
    "offload_interpretation",
]


def compare_kv_offload_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write baseline-vs-offload comparison CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_kv_offload_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_kv_offload_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline and kv_offload strategy results by workload and context bucket."""
    if frame.empty:
        return pd.DataFrame(columns=KV_OFFLOAD_COLUMNS)

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
        return pd.DataFrame(columns=KV_OFFLOAD_COLUMNS)

    rows: list[dict[str, Any]] = []
    compare_cols = ["provider", "engine", "model_id", "workload", "context_token_bucket"]
    for keys, group in aggregated.groupby(compare_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_group = dict(zip(compare_cols, keys, strict=True))
        baseline = _strategy_row(group, "baseline")
        offload = _strategy_row(group, "kv_offload")
        if baseline is None or offload is None:
            continue

        row = _build_comparison_row(by_group, baseline, offload)
        rows.append(row)

    result = pd.DataFrame(rows, columns=KV_OFFLOAD_COLUMNS)
    if result.empty:
        return pd.DataFrame(columns=KV_OFFLOAD_COLUMNS)
    return result.sort_values(
        ["provider", "engine", "model_id", "workload", "context_token_bucket"]
    ).reset_index(drop=True)


def interpret_offload_result(
    *,
    quality_delta: float | None,
    e2e_delta_pct: float | None,
    ttft_delta_pct: float | None,
    throughput_delta_pct: float | None,
    memory_delta_pct: float | None,
    offload_success_rate: float | None,
    missing_metrics: str,
) -> str:
    """Interpret offload tradeoffs without inventing unavailable telemetry."""
    if offload_success_rate is None or offload_success_rate < SUCCESS_RATE_FLOOR:
        return "insufficient_offload_signal"
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
        return "offload_promising"
    if memory_delta_pct is None and "gpu_memory_peak_gb" in missing_metrics.split(";"):
        return "memory_telemetry_missing"
    if (
        quality_delta is None
        and e2e_delta_pct is None
        and throughput_delta_pct is None
        and memory_delta_pct is None
    ):
        return "insufficient_offload_signal"
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
        rows.append(row)
    return pd.DataFrame(rows)


def _build_comparison_row(by_group: dict[str, Any], baseline: pd.Series, offload: pd.Series) -> dict[str, Any]:
    baseline_ttft = _to_float(baseline.get("ttft_ms_p50"))
    offload_ttft = _to_float(offload.get("ttft_ms_p50"))
    baseline_e2e = _to_float(baseline.get("e2e_latency_ms_p50"))
    offload_e2e = _to_float(offload.get("e2e_latency_ms_p50"))
    baseline_tps = _to_float(baseline.get("output_tokens_per_second_mean"))
    offload_tps = _to_float(offload.get("output_tokens_per_second_mean"))
    baseline_quality = _to_float(baseline.get("quality_score_mean"))
    offload_quality = _to_float(offload.get("quality_score_mean"))
    baseline_memory = _to_float(baseline.get("gpu_memory_peak_gb"))
    offload_memory = _to_float(offload.get("gpu_memory_peak_gb"))
    quality_delta = _subtract(offload_quality, baseline_quality)
    ttft_delta_pct = _delta_pct(offload_ttft, baseline_ttft)
    e2e_delta_pct = _delta_pct(offload_e2e, baseline_e2e)
    throughput_delta_pct = _delta_pct(offload_tps, baseline_tps)
    memory_delta_pct = _delta_pct(offload_memory, baseline_memory)
    missing_metrics = _join_missing(
        baseline.get("missing_metrics"),
        offload.get("missing_metrics"),
    )
    offload_success_rate = _to_float(offload.get("success_rate"))

    return {
        **by_group,
        "context_token_bucket": _nullable_int(by_group.get("context_token_bucket")),
        "baseline_strategy": str(baseline.get("strategy")),
        "offload_strategy": str(offload.get("strategy")),
        "baseline_ttft_ms_p50": baseline_ttft,
        "offload_ttft_ms_p50": offload_ttft,
        "ttft_delta_pct": ttft_delta_pct,
        "baseline_e2e_latency_ms_p50": baseline_e2e,
        "offload_e2e_latency_ms_p50": offload_e2e,
        "e2e_delta_pct": e2e_delta_pct,
        "baseline_output_tokens_per_second_mean": baseline_tps,
        "offload_output_tokens_per_second_mean": offload_tps,
        "throughput_delta_pct": throughput_delta_pct,
        "baseline_quality_score_mean": baseline_quality,
        "offload_quality_score_mean": offload_quality,
        "quality_delta": quality_delta,
        "baseline_gpu_memory_peak_gb": baseline_memory,
        "offload_gpu_memory_peak_gb": offload_memory,
        "memory_delta_pct": memory_delta_pct,
        "missing_metrics": missing_metrics,
        "requests": int(baseline.get("requests", 0)) + int(offload.get("requests", 0)),
        "baseline_success_rate": _to_float(baseline.get("success_rate")),
        "offload_success_rate": offload_success_rate,
        "offload_interpretation": interpret_offload_result(
            quality_delta=quality_delta,
            e2e_delta_pct=e2e_delta_pct,
            ttft_delta_pct=ttft_delta_pct,
            throughput_delta_pct=throughput_delta_pct,
            memory_delta_pct=memory_delta_pct,
            offload_success_rate=offload_success_rate,
            missing_metrics=missing_metrics,
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
    parser = argparse.ArgumentParser(description="Compare KV offload JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_kv_offload_results(input_path=args.input, output_path=args.output)
    print(f"Wrote KV offload comparison to {output}")


if __name__ == "__main__":
    main()
