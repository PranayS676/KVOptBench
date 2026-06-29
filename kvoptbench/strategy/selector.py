"""Simple placeholder strategy selector for Milestone 1."""

from __future__ import annotations

from pathlib import Path


def select_strategy_from_summary(input_path: str | Path) -> str:
    return (
        f"Strategy selection placeholder for {input_path}. "
        "Milestone 1 records data; strategy optimization starts after local harness validation."
    )

