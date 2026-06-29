"""JSON validity evaluator."""

from __future__ import annotations

import json

from kvoptbench.schemas import QualityResult


def evaluate_json_validity(output: str) -> QualityResult:
    try:
        json.loads(output)
    except json.JSONDecodeError as exc:
        return QualityResult(
            quality_score=0.0,
            quality_method="json_validity",
            passed=False,
            details={"error": str(exc)},
        )
    return QualityResult(
        quality_score=1.0,
        quality_method="json_validity",
        passed=True,
        details={},
    )

