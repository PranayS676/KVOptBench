"""Needle retrieval evaluator."""

from __future__ import annotations

from kvoptbench.evals.exact_match import contains_expected_answer
from kvoptbench.schemas import QualityResult, WorkloadItem


def evaluate_needle(output: str, item: WorkloadItem) -> QualityResult:
    result = contains_expected_answer(output, item.expected_answer)
    result.quality_method = "needle"
    return result

