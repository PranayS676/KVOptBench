"""Compatibility wrapper for the strategy advisor."""

from __future__ import annotations

from pathlib import Path

from kvoptbench.strategy.advisor import build_strategy_advisor_report


def select_strategy_from_summary(input_path: str | Path) -> str:
    report = build_strategy_advisor_report(summary_path=input_path)
    return f"Strategy Advisor overall recommendation: {report.overall_recommendation}"
