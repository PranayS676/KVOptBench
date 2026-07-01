"""Analyze cache behavior across shared-prefix overlap ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kvoptbench.analysis.cache import cache_miss_penalty_ms, miss_penalty_per_1k_tokens
from kvoptbench.analysis.statistics import flatten_metric_stats, mean_effect_size

MEANINGFUL_CACHE_GAIN_MS = 25.0

PREFIX_SWEEP_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "strategy",
    "shared_prefix_ratio",
    "shared_prefix_tokens",
    "cold_ttft_ms",
    "warm_ttft_ms",
    "cache_gain_ms",
    "miss_penalty_per_1k_tokens",
    "requests",
    "success_rate",
    "interpretation",
    "cold_ttft_ms_count",
    "cold_ttft_ms_std",
    "cold_ttft_ms_ci95_low",
    "cold_ttft_ms_ci95_high",
    "cold_ttft_ms_stats_status",
    "warm_ttft_ms_count",
    "warm_ttft_ms_std",
    "warm_ttft_ms_ci95_low",
    "warm_ttft_ms_ci95_high",
    "warm_ttft_ms_stats_status",
    "ttft_ms_effect_size",
]


def compare_prefix_sweep_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write prefix-overlap sweep CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_prefix_sweep_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_prefix_sweep_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare cold/warm TTFT by shared-prefix ratio."""
    if frame.empty:
        return pd.DataFrame(columns=PREFIX_SWEEP_COLUMNS)

    enriched = frame.copy()
    for column in [
        "provider",
        "engine",
        "model_id",
        "strategy",
        "cache_state",
        "ttft_ms",
        "success",
    ]:
        if column not in enriched:
            enriched[column] = None
    if "metadata" not in enriched:
        enriched["metadata"] = None
    if "shared_prefix_tokens" not in enriched:
        enriched["shared_prefix_tokens"] = 0
    if "target_input_tokens" not in enriched:
        enriched["target_input_tokens"] = 0

    enriched["shared_prefix_ratio"] = enriched.apply(_shared_prefix_ratio, axis=1)
    enriched = enriched[enriched["shared_prefix_ratio"].notna()].copy()
    if enriched.empty:
        return pd.DataFrame(columns=PREFIX_SWEEP_COLUMNS)

    enriched["ttft_ms"] = pd.to_numeric(enriched["ttft_ms"], errors="coerce")
    enriched["shared_prefix_tokens"] = pd.to_numeric(
        enriched["shared_prefix_tokens"], errors="coerce"
    ).fillna(0)
    enriched["success"] = enriched["success"].fillna(False).astype(bool)

    rows: list[dict[str, Any]] = []
    group_cols = ["provider", "engine", "model_id", "strategy", "shared_prefix_ratio"]
    for keys, group in enriched.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_group = dict(zip(group_cols, keys, strict=True))
        cold_values = group[group["cache_state"] == "cold"]["ttft_ms"].dropna()
        warm_values = group[group["cache_state"] == "warm"]["ttft_ms"].dropna()
        cold_ttft = _mean_ttft_values(cold_values)
        warm_ttft = _mean_ttft_values(warm_values)
        cache_gain = cache_miss_penalty_ms(cold_ttft, warm_ttft)
        shared_prefix_tokens = int(group["shared_prefix_tokens"].max())
        requests = int(len(group))
        successes = int(group["success"].sum())
        cold_stats = _renamed_stats("cold_ttft_ms", cold_values)
        warm_stats = _renamed_stats("warm_ttft_ms", warm_values)

        rows.append(
            {
                **by_group,
                "shared_prefix_tokens": shared_prefix_tokens,
                "cold_ttft_ms": cold_ttft,
                "warm_ttft_ms": warm_ttft,
                "cache_gain_ms": cache_gain,
                "miss_penalty_per_1k_tokens": miss_penalty_per_1k_tokens(
                    cache_gain, shared_prefix_tokens
                ),
                "requests": requests,
                "success_rate": round(successes / requests, 4) if requests else None,
                "interpretation": interpret_prefix_sweep(
                    by_group["shared_prefix_ratio"], cache_gain
                ),
                **cold_stats,
                **warm_stats,
                "ttft_ms_effect_size": mean_effect_size(cold_values, warm_values),
            }
        )

    result = pd.DataFrame(rows, columns=PREFIX_SWEEP_COLUMNS)
    return result.sort_values(
        ["provider", "engine", "model_id", "strategy", "shared_prefix_ratio"]
    ).reset_index(drop=True)


def interpret_prefix_sweep(shared_prefix_ratio: float | None, cache_gain_ms: float | None) -> str:
    """Label a prefix-overlap row without fabricating missing metrics."""
    if cache_gain_ms is None:
        return "insufficient_prefix_sweep_signal"
    if shared_prefix_ratio is None or shared_prefix_ratio <= 0:
        return "no_prefix_overlap"
    if cache_gain_ms <= 0:
        return "no_prefix_cache_gain"
    if cache_gain_ms < MEANINGFUL_CACHE_GAIN_MS:
        return "weak_prefix_cache_gain"
    return "meaningful_prefix_cache_gain"


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


def _shared_prefix_ratio(row) -> float | None:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        config_metadata = metadata.get("config_metadata")
        if (
            isinstance(config_metadata, dict)
            and config_metadata.get("workload_profile") == "random_prefix"
        ):
            return None
        workload_metadata = metadata.get("workload_metadata")
        if isinstance(workload_metadata, dict) and workload_metadata.get(
            "shared_prefix_ratio"
        ) is not None:
            return _round_ratio(workload_metadata["shared_prefix_ratio"])
    if "partial_prefix" not in str(row.get("workload", "")):
        return None
    shared_prefix_tokens = _to_float(row.get("shared_prefix_tokens"))
    target_input_tokens = _to_float(row.get("target_input_tokens"))
    if shared_prefix_tokens is None or target_input_tokens is None or target_input_tokens <= 0:
        return None
    return _round_ratio(shared_prefix_tokens / target_input_tokens)


def _mean_ttft_values(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return round(float(values.mean()), 3)


def _renamed_stats(prefix: str, values: pd.Series) -> dict[str, object]:
    stats = flatten_metric_stats(prefix, values)
    return {
        key: value
        for key, value in stats.items()
        if key.endswith(("_count", "_std", "_ci95_low", "_ci95_high", "_stats_status"))
    }


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _round_ratio(value) -> float:
    return round(float(value), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prefix-overlap sweep JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_prefix_sweep_results(input_path=args.input, output_path=args.output)
    print(f"Wrote prefix sweep comparison to {output}")


if __name__ == "__main__":
    main()
