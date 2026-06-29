"""Tool-call evaluator placeholder."""

from __future__ import annotations

from kvoptbench.schemas import QualityResult, WorkloadItem


def evaluate_tool_calling(output: str, item: WorkloadItem) -> QualityResult:
    passed = bool(item.expected_answer and item.expected_answer.lower() in output.lower())
    return QualityResult(
        quality_score=1.0 if passed else 0.0,
        quality_method="tool_calling_placeholder",
        passed=passed,
        details={"placeholder": True, "expected": item.expected_answer},
    )

