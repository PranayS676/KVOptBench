"""Cache analysis helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def cache_miss_penalty_ms(ttft_cold_ms: float | None, ttft_warm_ms: float | None) -> float | None:
    if ttft_cold_ms is None or ttft_warm_ms is None:
        return None
    return round(ttft_cold_ms - ttft_warm_ms, 3)


def prefix_cache_speedup(cache_off_ttft_ms: float | None, cache_on_ttft_ms: float | None) -> float | None:
    """Return cache-off/cache-on TTFT speedup when both values are usable."""
    if cache_off_ttft_ms is None or cache_on_ttft_ms is None or cache_on_ttft_ms <= 0:
        return None
    return round(cache_off_ttft_ms / cache_on_ttft_ms, 3)


def miss_penalty_per_1k_tokens(
    cache_miss_penalty: float | None, missed_prefix_tokens: int | float | None
) -> float | None:
    """Normalize a cache miss penalty by missed prefix tokens."""
    if cache_miss_penalty is None or not missed_prefix_tokens or missed_prefix_tokens <= 0:
        return None
    return round(float(cache_miss_penalty) / (float(missed_prefix_tokens) / 1000.0), 3)


def summarize_cold_warm_ttft(
    frame: pd.DataFrame,
    group_cols: Sequence[str] = ("engine", "strategy", "workload"),
) -> pd.DataFrame:
    """Summarize cold/warm TTFT deltas from request-level rows."""
    required = {"cache_state", "ttft_ms", "shared_prefix_tokens", *group_cols}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    filtered = frame[frame["cache_state"].isin(["cold", "warm"])].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["ttft_ms"] = pd.to_numeric(filtered["ttft_ms"], errors="coerce")
    filtered["shared_prefix_tokens"] = pd.to_numeric(
        filtered["shared_prefix_tokens"], errors="coerce"
    ).fillna(0)

    summaries: list[dict] = []
    for keys, group in filtered.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cold = group[group["cache_state"] == "cold"]["ttft_ms"].dropna()
        warm = group[group["cache_state"] == "warm"]["ttft_ms"].dropna()
        if cold.empty or warm.empty:
            continue
        cold_mean = round(float(cold.mean()), 3)
        warm_mean = round(float(warm.mean()), 3)
        shared_prefix_tokens = int(group["shared_prefix_tokens"].max())
        penalty = cache_miss_penalty_ms(cold_mean, warm_mean)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "cold_ttft_ms_mean": cold_mean,
                "warm_ttft_ms_mean": warm_mean,
                "cache_miss_penalty_ms": penalty,
                "shared_prefix_tokens": shared_prefix_tokens,
                "miss_penalty_per_1k_tokens": miss_penalty_per_1k_tokens(
                    penalty, shared_prefix_tokens
                ),
            }
        )
        summaries.append(row)
    return pd.DataFrame(summaries)

