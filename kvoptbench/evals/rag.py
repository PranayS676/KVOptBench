"""RAG faithfulness evaluator placeholder."""

from __future__ import annotations

from kvoptbench.schemas import QualityResult, WorkloadItem


def evaluate_rag(output: str, item: WorkloadItem) -> QualityResult:
    passed = bool(item.expected_answer and item.expected_answer.lower() in output.lower())
    return QualityResult(
        quality_score=1.0 if passed else 0.0,
        quality_method="rag_placeholder",
        passed=passed,
        details={"placeholder": True, "source_id": item.metadata.get("source_id")},
    )

