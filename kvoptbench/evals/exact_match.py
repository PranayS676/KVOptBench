"""String matching evaluators."""

from __future__ import annotations

from kvoptbench.schemas import QualityResult


def exact_match(output: str, expected: str | None) -> QualityResult:
    passed = bool(expected is not None and output.strip().lower() == expected.strip().lower())
    return QualityResult(
        quality_score=1.0 if passed else 0.0,
        quality_method="exact_match",
        passed=passed,
        details={"expected": expected},
    )


def contains_expected_answer(output: str, expected: str | None) -> QualityResult:
    passed = bool(expected and expected.lower() in output.lower())
    return QualityResult(
        quality_score=1.0 if passed else 0.0,
        quality_method="contains_expected",
        passed=passed,
        details={"expected": expected},
    )

