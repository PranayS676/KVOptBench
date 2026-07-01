"""Answer scoring helpers for QA-style public benchmark workloads."""

from __future__ import annotations

import re
import string
from collections import Counter

from kvoptbench.schemas import QualityResult, WorkloadItem

_ARTICLES = {"a", "an", "the"}


def evaluate_answer(output: str, item: WorkloadItem, *, method: str) -> QualityResult:
    """Score an answer with exact, contains, and token-F1 signals."""
    expected_answers = _expected_answers(item)
    if not expected_answers:
        return QualityResult(
            quality_score=None,
            quality_method=method,
            passed=None,
            details={"reason": "missing_expected_answer"},
        )

    normalized_output = _normalize(output)
    scored = [_score_one(normalized_output, expected) for expected in expected_answers]
    best = max(scored, key=lambda value: value["score"])
    passed = bool(best["contains_match"] or best["exact_match"] or best["token_f1"] >= 0.8)
    return QualityResult(
        quality_score=round(float(best["score"]), 4),
        quality_method=method,
        passed=passed,
        details={
            "expected_answers": expected_answers,
            "best_expected_answer": best["expected_answer"],
            "exact_match": best["exact_match"],
            "contains_match": best["contains_match"],
            "best_token_f1": round(float(best["token_f1"]), 4),
        },
    )


def _expected_answers(item: WorkloadItem) -> list[str]:
    values: list[str] = []
    for key in ["expected_answers", "answers", "gold_answers"]:
        raw = item.metadata.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if str(value).strip())
        elif isinstance(raw, str) and raw.strip():
            values.append(raw)
    if item.expected_answer:
        values.append(item.expected_answer)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize(value)
        if normalized and normalized not in seen:
            deduped.append(value)
            seen.add(normalized)
    return deduped


def _score_one(normalized_output: str, expected_answer: str) -> dict[str, float | str | bool]:
    normalized_expected = _normalize(expected_answer)
    exact = normalized_output == normalized_expected
    contains = bool(normalized_expected and normalized_expected in normalized_output)
    token_f1 = _token_f1(normalized_output, normalized_expected)
    score = 1.0 if exact or contains else token_f1
    return {
        "expected_answer": expected_answer,
        "exact_match": exact,
        "contains_match": contains,
        "token_f1": token_f1,
        "score": score,
    }


def _token_f1(output: str, expected: str) -> float:
    output_tokens = output.split()
    expected_tokens = expected.split()
    if not output_tokens or not expected_tokens:
        return 0.0
    overlap = Counter(output_tokens) & Counter(expected_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(output_tokens)
    recall = common / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    lowered = text.lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in re.split(r"\s+", without_punctuation) if token]
    tokens = [token for token in tokens if token not in _ARTICLES]
    return " ".join(tokens)
