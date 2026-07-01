"""RAG answer and source matching evaluator."""

from __future__ import annotations

from kvoptbench.schemas import QualityResult, WorkloadItem


def evaluate_rag(output: str, item: WorkloadItem) -> QualityResult:
    answer_passed = bool(item.expected_answer and item.expected_answer.lower() in output.lower())
    expected_sources = _expected_source_ids(item)
    found_sources = [source for source in expected_sources if source.lower() in output.lower()]
    missing_sources = [source for source in expected_sources if source not in found_sources]
    if expected_sources:
        source_recall = len(found_sources) / len(expected_sources)
        score = ((1.0 if answer_passed else 0.0) + source_recall) / 2
        passed = bool(answer_passed and not missing_sources)
        return QualityResult(
            quality_score=round(score, 4),
            quality_method="rag_source_match",
            passed=passed,
            details={
                "placeholder": False,
                "expected_source_ids": expected_sources,
                "found_source_ids": found_sources,
                "missing_source_ids": missing_sources,
                "source_recall": round(source_recall, 4),
                "answer_passed": answer_passed,
            },
        )
    return QualityResult(
        quality_score=1.0 if answer_passed else 0.0,
        quality_method="rag_answer_match",
        passed=answer_passed,
        details={
            "placeholder": False,
            "expected_source_ids": [],
            "answer_passed": answer_passed,
        },
    )


def _expected_source_ids(item: WorkloadItem) -> list[str]:
    values: list[str] = []
    for key in [
        "expected_source_ids",
        "expected_source_id",
        "expected_doc_ids",
        "expected_doc_id",
        "source_ids",
        "source_id",
    ]:
        raw = item.metadata.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if str(value).strip())
        elif isinstance(raw, str) and raw.strip():
            values.append(raw)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        lowered = value.lower()
        if lowered not in seen:
            deduped.append(value)
            seen.add(lowered)
    return deduped

