"""LongBench adapter for long-context frontier-model workloads."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kvoptbench.datasets.cache import read_json_records, resolve_dataset_cache_dir, write_json_records
from kvoptbench.datasets.hashing import sha256_text
from kvoptbench.datasets.huggingface import load_hf_dataset_records
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

LONGBENCH_SOURCE_URL = "https://huggingface.co/datasets/zai-org/LongBench"
LONGBENCH_HF_DATASET = "THUDM/LongBench"
LONGBENCH_DOWNLOAD_DEFAULT_MAX_ITEMS = 100
MODE_DEFAULT_SUBSETS = {
    "long_context_qa": ("qasper", "multifieldqa_en", "hotpotqa"),
    "long_context_retrieval": ("passage_retrieval_en",),
    "code_context": ("repobench-p",),
}
LONGBENCH_TEMPLATE = (
    "Answer the task using only the long context below.\n\n"
    "Context:\n{context}\n\nTask:\n{question}\n\nAnswer:"
)


class LongBenchAdapter:
    """Prepare LongBench rows for long-context quality/latency experiments."""

    info = DatasetAdapterInfo(
        name="longbench",
        dataset_name="LongBench",
        source_url=LONGBENCH_SOURCE_URL,
        supported_modes=("long_context_qa", "long_context_retrieval", "code_context"),
        license=None,
        rights_note="Use upstream LongBench dataset terms and per-subset attribution.",
        download_supported=True,
        fixture_supported=True,
    )

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        if options.mode not in self.info.supported_modes:
            raise ValueError(f"LongBench adapter does not support mode '{options.mode}'")
        run_options = _options_with_download_defaults(options)
        source_path, download_meta = _resolve_source_path(run_options)

        started = time.perf_counter()
        started_at = utc_now_iso()
        records = _load_records(source_path)
        items, exclusions = _generate_items(records, run_options)
        if run_options.max_items is not None:
            items = items[: run_options.max_items]
        if not items:
            raise ValueError("LongBench adapter produced no workload rows")

        workload_sha256 = write_workload_jsonl(items, run_options.out)
        finished_at = utc_now_iso()
        manifest = build_manifest(
            info=self.info,
            options=run_options,
            items=items,
            workload_sha256=workload_sha256,
            prompt_template=f"longbench_{run_options.mode}_v1",
            prompt_template_text=LONGBENCH_TEMPLATE,
            license_review_status=_license_review_status(run_options),
            redistribution_policy=_redistribution_policy(run_options),
            excluded_reasons=exclusions,
            rights_note=self.info.rights_note,
            source_sha256=source_hash(source_path),
            generation_started_at=started_at,
            generation_finished_at=finished_at,
            generation_duration_sec=round(time.perf_counter() - started, 3),
            cache_path=download_meta.get("cache_path"),
            download_method=download_meta.get("download_method"),
            downloaded_at=download_meta.get("downloaded_at"),
            source_url=download_meta.get("source_url"),
            source_revision=run_options.dataset_revision,
            known_limitations=[
                "LongBench adapter uses contains-expected placeholder quality checks.",
                "Downloaded rows are cached locally; raw public dataset files should not be committed.",
            ],
        )
        write_manifest(manifest, run_options.manifest)
        return DatasetPrepareResult(
            output_path=run_options.out,
            manifest_path=run_options.manifest,
            row_count=len(items),
            workload_sha256=workload_sha256,
            excluded_count=len(exclusions),
        )


def _generate_items(
    records: list[dict[str, Any]], options: DatasetPrepareOptions
) -> tuple[list[WorkloadItem], list[str]]:
    items: list[WorkloadItem] = []
    exclusions: list[str] = []
    wanted_subsets = set(options.subset or MODE_DEFAULT_SUBSETS[options.mode])
    for index, record in enumerate(records):
        subset = _subset_name(record)
        if wanted_subsets and subset not in wanted_subsets:
            continue
        context = _context(record)
        question = _question(record)
        expected_answer = _expected_answer(record)
        if not context:
            exclusions.append("missing_context")
            continue
        if not question:
            exclusions.append("missing_question")
            continue
        if not expected_answer:
            exclusions.append("missing_answer")
            continue
        prompt, measured_context, truncated = _render_prompt(context, question, options)
        measured_input_tokens = count_tokens(prompt, options.token_count_method)
        record_id = _record_id(record, index)
        item = WorkloadItem(
            task_id=f"longbench_{_slug(options.mode)}_{_slug(subset)}_{_slug(record_id)}",
            workload=f"longbench_{options.mode}",
            category="long_context",
            prompt=prompt,
            expected_answer=expected_answer,
            target_input_tokens=options.target_input_tokens,
            target_output_tokens=options.target_output_tokens,
            prefix_group_id=f"longbench_{subset}_{record_id}",
            shared_prefix_tokens=count_tokens(measured_context, options.token_count_method),
            eval_type="contains_expected",
            metadata={
                "dataset": "longbench",
                "source": "longbench",
                "dataset_source_url": LONGBENCH_SOURCE_URL,
                "dataset_revision": options.dataset_revision,
                "mode": options.mode,
                "subset": subset,
                "split": options.split,
                "source_record_id": record_id,
                "source_document_id": record_id,
                "source_question_id": record_id,
                "prefix_hash": sha256_text(measured_context),
                "prompt_hash": sha256_text(prompt),
                "expected_answer_hash": sha256_text(expected_answer),
                "tokenizer_id": options.tokenizer_id,
                "tokenizer_revision": options.tokenizer_revision,
                "token_count_method": options.token_count_method,
                "target_input_tokens": options.target_input_tokens,
                "target_output_tokens": options.target_output_tokens,
                "measured_input_tokens": measured_input_tokens,
                "measured_shared_prefix_tokens": count_tokens(
                    measured_context, options.token_count_method
                ),
                "measured_suffix_tokens": max(
                    0,
                    measured_input_tokens - count_tokens(measured_context, options.token_count_method),
                ),
                "context_bucket": options.target_input_tokens,
                "prompt_template": f"longbench_{options.mode}_v1",
                "prompt_template_hash": sha256_text(LONGBENCH_TEMPLATE),
                "adapter": "longbench",
                "adapter_version": "0.1.0",
                "truncated": truncated,
                "truncation_policy": "tail" if truncated else "none",
                "excluded_reason": None,
                "answer_type": "longbench_answer",
                "evaluator": "contains_expected",
                "evaluator_version": "0.1.0",
                "redistributable_prompt": False,
                "redistributable_output": False,
                "rights_note": "Use upstream LongBench dataset terms and per-subset attribution.",
                "license_review_status": _license_review_status(options),
                "redistribution_policy": _redistribution_policy(options),
            },
        )
        items.append(item)
    return items, exclusions


def _render_prompt(
    context: str, question: str, options: DatasetPrepareOptions
) -> tuple[str, str, bool]:
    suffix = f"\n\nTask:\n{question}\n\nAnswer:"
    budget_chars = options.target_input_tokens * 4
    if len(context) + len(suffix) <= budget_chars:
        return LONGBENCH_TEMPLATE.format(context=context, question=question), context, False
    available_context_chars = max(0, budget_chars - len(suffix) - 128)
    measured_context = context[:available_context_chars].rstrip()
    return LONGBENCH_TEMPLATE.format(context=measured_context, question=question), measured_context, True


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"LongBench source path does not exist: {path}")
    if path.is_dir():
        raise ValueError("LongBench source path must be a JSON or JSONL file")
    if path.suffix.lower() == ".json":
        return read_json_records(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid LongBench JSONL line {line_no}: {exc}") from exc
    return records


def _options_with_download_defaults(options: DatasetPrepareOptions) -> DatasetPrepareOptions:
    updates: dict[str, Any] = {}
    if options.download and options.max_items is None:
        updates["max_items"] = LONGBENCH_DOWNLOAD_DEFAULT_MAX_ITEMS
    if options.download and not options.subset:
        updates["subset"] = MODE_DEFAULT_SUBSETS[options.mode]
    return options.model_copy(update=updates) if updates else options


def _resolve_source_path(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    if options.download:
        return _download_longbench_source(options)
    if options.source_path is None:
        raise ValueError("LongBench adapter requires --source-path or --download")
    return options.source_path, {}


def _download_longbench_source(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    split = options.split or "test"
    subsets = options.subset or MODE_DEFAULT_SUBSETS[options.mode]
    cache_dir = resolve_dataset_cache_dir("longbench", options.cache_dir)
    cache_path = cache_dir / f"{options.mode}_{split}_{'_'.join(_slug(value) for value in subsets)}.json"
    if cache_path.exists() and not options.force:
        return cache_path, {
            "cache_path": str(cache_path),
            "download_method": "huggingface.datasets",
            "downloaded_at": None,
            "source_url": LONGBENCH_SOURCE_URL,
        }

    records: list[dict[str, Any]] = []
    for subset in subsets:
        subset_records = load_hf_dataset_records(
            LONGBENCH_HF_DATASET,
            subset=subset,
            split=split,
            revision=options.dataset_revision,
            max_items=options.max_items,
            trust_remote_code=True,
        )
        for record in subset_records:
            record.setdefault("subset", subset)
            records.append(record)
    if not records:
        raise ValueError(f"LongBench download returned no rows for split '{split}'")
    write_json_records(cache_path, records, force=True)
    return cache_path, {
        "cache_path": str(cache_path),
        "download_method": "huggingface.datasets",
        "downloaded_at": utc_now_iso(),
        "source_url": LONGBENCH_SOURCE_URL,
    }


def _record_id(record: dict[str, Any], fallback_index: int) -> str:
    return str(record.get("_id") or record.get("id") or record.get("question_id") or fallback_index)


def _subset_name(record: dict[str, Any]) -> str:
    return str(record.get("subset") or record.get("dataset") or record.get("category") or "unknown")


def _context(record: dict[str, Any]) -> str:
    value = record.get("context") or record.get("document") or record.get("passage") or ""
    return _stringify(value)


def _question(record: dict[str, Any]) -> str:
    value = record.get("input") or record.get("question") or record.get("query") or ""
    return _stringify(value).strip()


def _expected_answer(record: dict[str, Any]) -> str | None:
    value = record.get("expected_answer") or record.get("answer") or record.get("answers")
    if isinstance(value, list):
        for item in value:
            if str(item).strip():
                return str(item).strip()
        return None
    if value in (None, ""):
        return None
    return str(value).strip()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify(item) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_stringify(raw)}" for key, raw in value.items())
    return str(value)


def _license_review_status(options: DatasetPrepareOptions) -> str:
    return "upstream_terms_need_user_review" if options.download else "fixture_only"


def _redistribution_policy(options: DatasetPrepareOptions) -> str:
    return "generated_workloads_not_redistributable" if options.download else "tiny_fixture_allowed"


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
