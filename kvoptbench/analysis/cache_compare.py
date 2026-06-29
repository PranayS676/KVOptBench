"""Build cache-specific comparison CSVs from request-level result JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from kvoptbench.analysis.cache import (
    cache_miss_penalty_ms,
    infer_workload_profile,
    interpret_cache_signal,
    miss_penalty_per_1k_tokens,
)

CACHE_COMPARISON_COLUMNS = [
    "provider",
    "engine",
    "model_id",
    "strategy",
    "shared_cold_ttft_ms",
    "shared_warm_ttft_ms",
    "random_cold_ttft_ms",
    "random_warm_ttft_ms",
    "shared_cache_miss_penalty_ms",
    "random_cache_miss_penalty_ms",
    "control_adjusted_cache_gain_ms",
    "shared_prefix_tokens",
    "miss_penalty_per_1k_tokens",
    "interpretation",
]


def compare_cache_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Read request-level JSONL rows and write cache comparison CSV."""
    output_path = Path(output_path)
    rows = _read_rows(Path(input_path))
    comparison = build_cache_comparison(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def build_cache_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare shared-prefix cold/warm TTFT against random-prefix controls."""
    if frame.empty:
        return pd.DataFrame(columns=CACHE_COMPARISON_COLUMNS)

    enriched = frame.copy()
    for column in ["provider", "engine", "model_id", "strategy", "cache_state", "ttft_ms"]:
        if column not in enriched:
            enriched[column] = None
    if "shared_prefix_tokens" not in enriched:
        enriched["shared_prefix_tokens"] = 0
    if "metadata" not in enriched:
        enriched["metadata"] = None

    enriched["_workload_profile"] = enriched.apply(infer_workload_profile, axis=1)
    enriched["ttft_ms"] = pd.to_numeric(enriched["ttft_ms"], errors="coerce")
    enriched["shared_prefix_tokens"] = pd.to_numeric(
        enriched["shared_prefix_tokens"], errors="coerce"
    ).fillna(0)

    rows: list[dict[str, Any]] = []
    group_cols = ["provider", "engine", "model_id", "strategy"]
    for keys, group in enriched.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_group = dict(zip(group_cols, keys, strict=True))

        shared_cold = _mean_ttft(group, "shared_prefix", "cold")
        shared_warm = _mean_ttft(group, "shared_prefix", "warm")
        random_cold = _mean_ttft(group, "random_prefix", "cold")
        random_warm = _mean_ttft(group, "random_prefix", "warm")
        if shared_cold is None and shared_warm is None:
            continue

        shared_penalty = cache_miss_penalty_ms(shared_cold, shared_warm)
        random_penalty = cache_miss_penalty_ms(random_cold, random_warm)
        control_adjusted = _subtract(shared_penalty, random_penalty)
        shared_prefix_tokens = _shared_prefix_tokens(group)

        rows.append(
            {
                **by_group,
                "shared_cold_ttft_ms": shared_cold,
                "shared_warm_ttft_ms": shared_warm,
                "random_cold_ttft_ms": random_cold,
                "random_warm_ttft_ms": random_warm,
                "shared_cache_miss_penalty_ms": shared_penalty,
                "random_cache_miss_penalty_ms": random_penalty,
                "control_adjusted_cache_gain_ms": control_adjusted,
                "shared_prefix_tokens": shared_prefix_tokens,
                "miss_penalty_per_1k_tokens": miss_penalty_per_1k_tokens(
                    shared_penalty, shared_prefix_tokens
                ),
                "interpretation": interpret_cache_signal(shared_penalty, random_penalty),
            }
        )
    return pd.DataFrame(rows, columns=CACHE_COMPARISON_COLUMNS)


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


def _mean_ttft(group: pd.DataFrame, workload_profile: str, cache_state: str) -> float | None:
    values = group[
        (group["_workload_profile"] == workload_profile) & (group["cache_state"] == cache_state)
    ]["ttft_ms"].dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 3)


def _shared_prefix_tokens(group: pd.DataFrame) -> int:
    shared = group[group["_workload_profile"] == "shared_prefix"]["shared_prefix_tokens"]
    if shared.empty:
        return 0
    return int(shared.max())


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cache experiment JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = compare_cache_results(input_path=args.input, output_path=args.output)
    print(f"Wrote cache comparison to {output}")


if __name__ == "__main__":
    main()
