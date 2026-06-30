"""QASPER dataset adapter for cache-aware workload preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kvoptbench.datasets.hashing import sha256_text
from kvoptbench.datasets.manifest import (
    DatasetAdapterInfo,
    DatasetPrepareOptions,
    DatasetPrepareResult,
    build_manifest,
    source_hash,
    write_manifest,
    write_workload_jsonl,
)
from kvoptbench.datasets.token_counting import count_tokens
from kvoptbench.schemas import WorkloadItem, utc_now_iso

QASPER_SOURCE_URL = "https://huggingface.co/datasets/allenai/qasper"
QASPER_TEMPLATE = (
    "You are given a paper excerpt. Answer the question using only the paper context.\n\n"
    "Paper:\n{prefix}\n\nQuestion: {question}\nAnswer:"
)
PARTIAL_PREFIX_RATIOS = (0.0, 0.25, 0.5, 0.75, 0.9)


class QasperAdapter:
    """Prepare QASPER-style records for shared-prefix cache experiments."""

    info = DatasetAdapterInfo(
        name="qasper",
        dataset_name="QASPER",
        source_url=QASPER_SOURCE_URL,
        supported_modes=("shared_prefix", "random_prefix", "partial_prefix_sweep"),
        license="cc-by-4.0",
        rights_note="Use upstream QASPER attribution and license terms for real data.",
        download_supported=False,
        fixture_supported=True,
    )

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        if options.mode not in self.info.supported_modes:
            raise ValueError(f"QASPER adapter does not support mode '{options.mode}'")
        if options.download:
            raise ValueError("QASPER downloads are not implemented; pass --source-path")
        if options.source_path is None:
            raise ValueError("QASPER adapter requires --source-path")

        started = time.perf_counter()
        started_at = utc_now_iso()
        records = _load_records(options.source_path)
        items, exclusions = _generate_items(records, options)
        if options.max_items is not None:
            items = items[: options.max_items]
        if not items:
            raise ValueError("QASPER adapter produced no workload rows")

        workload_sha256 = write_workload_jsonl(items, options.out)
        finished_at = utc_now_iso()
        manifest = build_manifest(
            info=self.info,
            options=options,
            items=items,
            workload_sha256=workload_sha256,
            prompt_template=f"qasper_{options.mode}_v1",
            prompt_template_text=QASPER_TEMPLATE,
            license_review_status="fixture_only",
            redistribution_policy="tiny_fixture_allowed",
            excluded_reasons=exclusions,
            rights_note=self.info.rights_note,
            source_sha256=source_hash(options.source_path),
            generation_started_at=started_at,
            generation_finished_at=finished_at,
            generation_duration_sec=round(time.perf_counter() - started, 3),
            known_limitations=[
                "Default implementation reads local fixture/source files only; no network download.",
                "Token counts use char_approx_4 unless a future tokenizer adapter is provided.",
            ],
        )
        write_manifest(manifest, options.manifest)
        return DatasetPrepareResult(
            output_path=options.out,
            manifest_path=options.manifest,
            row_count=len(items),
            workload_sha256=workload_sha256,
            excluded_count=len(exclusions),
        )


def _generate_items(
    records: list[dict[str, Any]], options: DatasetPrepareOptions
) -> tuple[list[WorkloadItem], list[str]]:
    if options.mode == "shared_prefix":
        return _shared_prefix_items(records, options)
    if options.mode == "random_prefix":
        return _random_prefix_items(records, options)
    if options.mode == "partial_prefix_sweep":
        return _partial_prefix_items(records, options)
    raise ValueError(f"Unsupported QASPER mode: {options.mode}")


def _shared_prefix_items(
    records: list[dict[str, Any]], options: DatasetPrepareOptions
) -> tuple[list[WorkloadItem], list[str]]:
    items: list[WorkloadItem] = []
    exclusions: list[str] = []
    for record in records:
        paper_id = _record_id(record)
        prefix = _document_prefix(record)
        for question in _questions(record):
            answer = _answer_text(question)
            if answer is None:
                exclusions.append("missing_answer")
                continue
            question_id = _question_id(question, len(items))
            item = _make_item(
                options=options,
                mode="shared_prefix",
                task_id=f"qasper_shared_{_slug(paper_id)}_{_slug(question_id)}",
                workload="qasper_shared_prefix",
                category="prefix_cache",
                source_document_id=paper_id,
                source_question_id=question_id,
                question_document_id=paper_id,
                prefix=prefix,
                question_text=_question_text(question),
                expected_answer=answer,
                answer_type=_answer_type(question),
                prefix_group_id=f"qasper_{paper_id}",
                shared_prefix_text=prefix,
                control_type=None,
                prefix_overlap_ratio=1.0,
            )
            items.append(item)
    return items, exclusions


def _random_prefix_items(
    records: list[dict[str, Any]], options: DatasetPrepareOptions
) -> tuple[list[WorkloadItem], list[str]]:
    if len(records) < 2:
        raise ValueError("QASPER random_prefix mode requires at least two source documents")

    items: list[WorkloadItem] = []
    exclusions: list[str] = []
    prefixes = [_document_prefix(record) for record in records]
    paper_ids = [_record_id(record) for record in records]
    for record_index, record in enumerate(records):
        question_document_id = _record_id(record)
        control_index = (record_index + 1) % len(records)
        control_document_id = paper_ids[control_index]
        control_base = prefixes[control_index]
        for question in _questions(record):
            answer = _answer_text(question)
            if answer is None:
                exclusions.append("missing_answer")
                continue
            question_id = _question_id(question, len(items))
            control_prefix = (
                f"{control_base}\n\nRandom-prefix control nonce: "
                f"{question_document_id}:{question_id}"
            )
            item = _make_item(
                options=options,
                mode="random_prefix",
                task_id=(
                    f"qasper_random_{_slug(control_document_id)}_"
                    f"{_slug(question_document_id)}_{_slug(question_id)}"
                ),
                workload="qasper_random_prefix",
                category="prefix_cache_control",
                source_document_id=control_document_id,
                source_question_id=question_id,
                question_document_id=question_document_id,
                prefix=control_prefix,
                question_text=_question_text(question),
                expected_answer=answer,
                answer_type=_answer_type(question),
                prefix_group_id=None,
                shared_prefix_text="",
                control_type="random_prefix",
                prefix_overlap_ratio=0.0,
            )
            items.append(item)
    return items, exclusions


def _partial_prefix_items(
    records: list[dict[str, Any]], options: DatasetPrepareOptions
) -> tuple[list[WorkloadItem], list[str]]:
    if not records:
        return [], ["missing_source_record"]
    record = records[0]
    paper_id = _record_id(record)
    questions = _questions(record)
    if not questions:
        return [], ["missing_question"]
    question = questions[0]
    answer = _answer_text(question)
    if answer is None:
        return [], ["missing_answer"]

    base_prefix = _document_prefix(record)
    alternate_prefix = _document_prefix(records[1]) if len(records) > 1 else base_prefix
    question_id = _question_id(question, 0)
    items: list[WorkloadItem] = []
    for ratio in PARTIAL_PREFIX_RATIOS:
        shared_chars = int(len(base_prefix) * ratio)
        shared_text = base_prefix[:shared_chars].rstrip()
        unique_text = (
            f"{alternate_prefix}\n\nPartial-prefix unique suffix for ratio {ratio:.2f}."
        )
        prefix = f"{shared_text}\n\n{unique_text}".strip() if shared_text else unique_text
        item = _make_item(
            options=options,
            mode="partial_prefix_sweep",
            task_id=f"qasper_partial_{_slug(paper_id)}_{int(ratio * 100):03d}",
            workload="qasper_partial_prefix_sweep",
            category="prefix_cache_sweep",
            source_document_id=paper_id,
            source_question_id=question_id,
            question_document_id=paper_id,
            prefix=prefix,
            question_text=_question_text(question),
            expected_answer=answer,
            answer_type=_answer_type(question),
            prefix_group_id=f"qasper_partial_{paper_id}" if ratio > 0 else None,
            shared_prefix_text=shared_text,
            control_type="partial_prefix_sweep",
            prefix_overlap_ratio=ratio,
        )
        items.append(item)
    return items, []


def _make_item(
    *,
    options: DatasetPrepareOptions,
    mode: str,
    task_id: str,
    workload: str,
    category: str,
    source_document_id: str,
    source_question_id: str,
    question_document_id: str,
    prefix: str,
    question_text: str,
    expected_answer: str,
    answer_type: str,
    prefix_group_id: str | None,
    shared_prefix_text: str,
    control_type: str | None,
    prefix_overlap_ratio: float,
) -> WorkloadItem:
    prompt, measured_prefix, truncated = _render_prompt(prefix, question_text, options)
    measured_input_tokens = count_tokens(prompt, options.token_count_method)
    measured_prefix_tokens = count_tokens(measured_prefix, options.token_count_method)
    shared_prefix_tokens = (
        count_tokens(shared_prefix_text, options.token_count_method) if shared_prefix_text else 0
    )
    metadata = {
        "dataset": "qasper",
        "source": "qasper",
        "dataset_source_url": QASPER_SOURCE_URL,
        "dataset_revision": None,
        "source_license": "cc-by-4.0",
        "mode": mode,
        "control_type": control_type,
        "split": options.split,
        "source_record_id": source_document_id,
        "source_document_id": source_document_id,
        "source_question_id": source_question_id,
        "question_document_id": question_document_id,
        "prefix_hash": sha256_text(shared_prefix_text or measured_prefix),
        "prompt_hash": sha256_text(prompt),
        "expected_answer_hash": sha256_text(expected_answer),
        "tokenizer_id": options.tokenizer_id,
        "tokenizer_revision": options.tokenizer_revision,
        "token_count_method": options.token_count_method,
        "target_input_tokens": options.target_input_tokens,
        "target_output_tokens": options.target_output_tokens,
        "measured_input_tokens": measured_input_tokens,
        "measured_shared_prefix_tokens": shared_prefix_tokens,
        "measured_suffix_tokens": max(0, measured_input_tokens - measured_prefix_tokens),
        "context_bucket": options.target_input_tokens,
        "prompt_template": f"qasper_{mode}_v1",
        "prompt_template_hash": sha256_text(QASPER_TEMPLATE),
        "adapter": "qasper",
        "adapter_version": "0.1.0",
        "truncated": truncated,
        "truncation_policy": "tail" if truncated else "none",
        "excluded_reason": None,
        "answer_type": answer_type,
        "evaluator": "contains_expected",
        "evaluator_version": "0.1.0",
        "difficulty": None,
        "prefix_overlap_ratio": prefix_overlap_ratio,
        "redistributable_prompt": False,
        "redistributable_output": False,
        "rights_note": "Use upstream QASPER attribution and license terms for real data.",
        "license_review_status": "fixture_only",
        "redistribution_policy": "tiny_fixture_allowed",
    }
    return WorkloadItem(
        task_id=task_id,
        workload=workload,
        category=category,
        prompt=prompt,
        expected_answer=expected_answer,
        target_input_tokens=options.target_input_tokens,
        target_output_tokens=options.target_output_tokens,
        prefix_group_id=prefix_group_id,
        shared_prefix_tokens=shared_prefix_tokens,
        eval_type="contains_expected",
        metadata=metadata,
    )


def _render_prompt(
    prefix: str, question_text: str, options: DatasetPrepareOptions
) -> tuple[str, str, bool]:
    suffix = f"\n\nQuestion: {question_text}\nAnswer:"
    budget_chars = options.target_input_tokens * 4
    if len(prefix) + len(suffix) <= budget_chars:
        return QASPER_TEMPLATE.format(prefix=prefix, question=question_text), prefix, False
    available_prefix_chars = max(0, budget_chars - len(suffix) - 128)
    measured_prefix = prefix[:available_prefix_chars].rstrip()
    return QASPER_TEMPLATE.format(prefix=measured_prefix, question=question_text), measured_prefix, True


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"QASPER source path does not exist: {path}")
    if path.is_dir():
        raise ValueError("QASPER source path must be a JSON or JSONL file")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("records") or [payload]
        if not isinstance(payload, list):
            raise ValueError("QASPER JSON source must contain a list of records")
        return [dict(record) for record in payload]
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid QASPER JSONL line {line_no}: {exc}") from exc
    return records


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("paper_id") or record.get("id") or record.get("_id") or "unknown")


def _document_prefix(record: dict[str, Any]) -> str:
    parts: list[str] = []
    title = record.get("title")
    if title:
        parts.append(f"Title: {title}")
    abstract = record.get("abstract")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    sections = record.get("sections") or record.get("full_text") or []
    if isinstance(sections, str):
        parts.append(sections)
    elif isinstance(sections, list):
        for section in sections:
            if isinstance(section, str):
                parts.append(section)
            elif isinstance(section, dict):
                heading = section.get("heading") or section.get("section_name")
                text = section.get("text") or section.get("paragraphs") or ""
                if isinstance(text, list):
                    text = " ".join(str(value) for value in text)
                parts.append(f"{heading}: {text}" if heading else str(text))
    return "\n\n".join(part for part in parts if part).strip()


def _questions(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("questions") or record.get("qas") or []
    if not isinstance(raw, list):
        return []
    return [question for question in raw if isinstance(question, dict)]


def _question_id(question: dict[str, Any], fallback_index: int) -> str:
    return str(question.get("question_id") or question.get("id") or f"q{fallback_index}")


def _question_text(question: dict[str, Any]) -> str:
    return str(question.get("question") or question.get("question_text") or "").strip()


def _answer_text(question: dict[str, Any]) -> str | None:
    if question.get("expected_answer"):
        return str(question["expected_answer"])
    answers = question.get("answers") or question.get("answer") or []
    if isinstance(answers, str):
        return answers
    if isinstance(answers, dict):
        return _answer_from_dict(answers)
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            if isinstance(answer, dict):
                extracted = _answer_from_dict(answer)
                if extracted:
                    return extracted
    return None


def _answer_type(question: dict[str, Any]) -> str:
    answers = question.get("answers") or []
    if isinstance(answers, list) and answers and isinstance(answers[0], dict):
        return str(answers[0].get("answer_type") or "free_form")
    return str(question.get("answer_type") or "free_form")


def _answer_from_dict(answer: dict[str, Any]) -> str | None:
    for key in ["answer", "text", "free_form_answer", "yes_no", "unanswerable"]:
        value = answer.get(key)
        if value not in (None, ""):
            return str(value)
    spans = answer.get("extractive_spans")
    if isinstance(spans, list) and spans:
        return str(spans[0])
    return None


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
