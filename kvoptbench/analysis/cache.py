"""Cache analysis helpers."""

from __future__ import annotations


def cache_miss_penalty_ms(ttft_cold_ms: float | None, ttft_warm_ms: float | None) -> float | None:
    if ttft_cold_ms is None or ttft_warm_ms is None:
        return None
    return round(ttft_cold_ms - ttft_warm_ms, 3)

