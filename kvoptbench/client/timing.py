"""Timing and token-estimation helpers."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def estimate_tokens(text: str) -> int:
    """Estimate tokens without requiring a model tokenizer."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def milliseconds(seconds: float) -> float:
    return round(seconds * 1000, 3)


def average_inter_token_latency_ms(token_timestamps: Sequence[float]) -> float | None:
    if len(token_timestamps) < 2:
        return None
    gaps = [
        milliseconds(token_timestamps[index] - token_timestamps[index - 1])
        for index in range(1, len(token_timestamps))
    ]
    return round(statistics.mean(gaps), 3) if gaps else None

