"""Simple placeholder strategy selector."""

from __future__ import annotations

from pathlib import Path


def select_strategy_from_summary(input_path: str | Path) -> str:
    return (
        f"Strategy selection placeholder for {input_path}. "
        "KVOptBench records benchmark data before strategy optimization is enabled."
    )

