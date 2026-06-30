"""Berkeley Function Calling Leaderboard adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kvoptbench.datasets.cache import read_json_records, resolve_dataset_cache_dir, write_json_records
from kvoptbench.datasets.download import download_file
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

BFCL_SOURCE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_DEFAULT_FILES = ("BFCL_v3_simple.json",)
BFCL_DOWNLOAD_DEFAULT_MAX_ITEMS = 100
BFCL_TOOL_SCHEMA = {
    "type": "object",
    "required": ["tool", "arguments"],
    "properties": {
        "tool": {"type": "string"},
        "arguments": {"type": "object"},
    },
}
BFCL_TEMPLATE = (
    "Return a JSON tool call with tool and arguments fields.\n\n"
    "Available tools:\n{tools}\n\nUser request:\n{request}\n\nAnswer:"
)


class BfclAdapter:
    """Prepare BFCL rows for tool-calling quality/latency experiments."""

    info = DatasetAdapterInfo(
        name="bfcl",
        dataset_name="Berkeley Function Calling Leaderboard",
        source_url=BFCL_SOURCE_URL,
        supported_modes=("tool_calling",),
        license=None,
        rights_note="Use upstream BFCL dataset terms and attribution.",
        download_supported=True,
        fixture_supported=True,
    )

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        if options.mode not in self.info.supported_modes:
            raise ValueError(f"BFCL adapter does not support mode '{options.mode}'")
        run_options = _options_with_download_defaults(options)
        source_path, download_meta = _resolve_source_path(run_options)

        started = time.perf_counter()
        started_at = utc_now_iso()
        records = _load_records(source_path)
        items, exclusions = _generate_items(records, run_options)
        if run_options.max_items is not None:
            items = items[: run_options.max_items]
        if not items:
            raise ValueError("BFCL adapter produced no workload rows")

        workload_sha256 = write_workload_jsonl(items, run_options.out)
        finished_at = utc_now_iso()
        manifest = build_manifest(
            info=self.info,
            options=run_options,
            items=items,
            workload_sha256=workload_sha256,
            prompt_template="bfcl_tool_calling_v1",
            prompt_template_text=BFCL_TEMPLATE,
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
                "Tool-call quality uses a placeholder expected-tool-name check.",
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
    for index, record in enumerate(records):
        request = _request(record)
        tools = _tools(record)
        expected_tool = _expected_tool(record)
        if not request:
            exclusions.append("missing_request")
            continue
        if not tools:
            exclusions.append("missing_tools")
            continue
        if not expected_tool:
            exclusions.append("missing_expected_tool")
            continue
        openai_tools = _openai_tools(tools)
        tools_text = json.dumps(tools, ensure_ascii=False, sort_keys=True)
        prompt = BFCL_TEMPLATE.format(tools=tools_text, request=request)
        measured_input_tokens = count_tokens(prompt, options.token_count_method)
        record_id = _record_id(record, index)
        items.append(
            WorkloadItem(
                task_id=f"bfcl_tool_calling_{_slug(record_id)}",
                workload="bfcl_tool_calling",
                category="structured_output",
                prompt=prompt,
                expected_answer=expected_tool,
                expected_schema=BFCL_TOOL_SCHEMA,
                target_input_tokens=options.target_input_tokens,
                target_output_tokens=options.target_output_tokens,
                prefix_group_id=None,
                shared_prefix_tokens=0,
                eval_type="tool_calling",
                metadata={
                    "dataset": "bfcl",
                    "source": "bfcl",
                    "dataset_source_url": BFCL_SOURCE_URL,
                    "dataset_revision": options.dataset_revision,
                    "mode": options.mode,
                    "split": options.split,
                    "subset": ",".join(options.subset) if options.subset else None,
                    "source_record_id": record_id,
                    "source_document_id": record_id,
                    "source_question_id": record_id,
                    "expected_tool": expected_tool,
                    "openai_tools": openai_tools,
                    "tool_choice": "auto",
                    "tool_count": len(tools),
                    "prefix_hash": sha256_text(tools_text),
                    "prompt_hash": sha256_text(prompt),
                    "expected_answer_hash": sha256_text(expected_tool),
                    "tokenizer_id": options.tokenizer_id,
                    "tokenizer_revision": options.tokenizer_revision,
                    "token_count_method": options.token_count_method,
                    "target_input_tokens": options.target_input_tokens,
                    "target_output_tokens": options.target_output_tokens,
                    "measured_input_tokens": measured_input_tokens,
                    "measured_shared_prefix_tokens": 0,
                    "measured_suffix_tokens": measured_input_tokens,
                    "context_bucket": options.target_input_tokens,
                    "prompt_template": "bfcl_tool_calling_v1",
                    "prompt_template_hash": sha256_text(BFCL_TEMPLATE),
                    "adapter": "bfcl",
                    "adapter_version": "0.1.0",
                    "truncated": False,
                    "truncation_policy": "none",
                    "excluded_reason": None,
                    "answer_type": "tool_name",
                    "evaluator": "tool_calling",
                    "evaluator_version": "0.2.0",
                    "redistributable_prompt": False,
                    "redistributable_output": False,
                    "rights_note": "Use upstream BFCL dataset terms and attribution.",
                    "license_review_status": _license_review_status(options),
                    "redistribution_policy": _redistribution_policy(options),
                },
            )
        )
    return items, exclusions


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"BFCL source path does not exist: {path}")
    if path.is_dir():
        raise ValueError("BFCL source path must be a JSON or JSONL file")
    if path.suffix.lower() == ".json":
        return read_json_records(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BFCL JSONL line {line_no}: {exc}") from exc
    return records


def _options_with_download_defaults(options: DatasetPrepareOptions) -> DatasetPrepareOptions:
    if options.download and options.max_items is None:
        return options.model_copy(update={"max_items": BFCL_DOWNLOAD_DEFAULT_MAX_ITEMS})
    return options


def _resolve_source_path(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    if options.download:
        return _download_bfcl_source(options)
    if options.source_path is None:
        raise ValueError("BFCL adapter requires --source-path or --download")
    return options.source_path, {}


def _download_bfcl_source(options: DatasetPrepareOptions) -> tuple[Path, dict[str, str | None]]:
    selected_files = _selected_bfcl_files(options)
    cache_dir = resolve_dataset_cache_dir("bfcl", options.cache_dir)
    cache_path = cache_dir / f"{'_'.join(_slug(file_name) for file_name in selected_files)}.json"
    if cache_path.exists() and not options.force:
        return cache_path, {
            "cache_path": str(cache_path),
            "download_method": "huggingface_file",
            "downloaded_at": None,
            "source_url": BFCL_SOURCE_URL,
        }

    records: list[dict[str, Any]] = []
    downloaded_at: str | None = None
    for file_name in selected_files:
        raw_path = cache_dir / "raw" / file_name
        result = download_file(_bfcl_file_url(file_name, options.dataset_revision), raw_path, force=options.force)
        downloaded_at = downloaded_at or result.downloaded_at
        for record in _load_records(result.path):
            record.setdefault("source_file", file_name)
            records.append(record)
            if options.max_items is not None and len(records) >= options.max_items:
                break
        if options.max_items is not None and len(records) >= options.max_items:
            break
    if not records:
        raise ValueError("BFCL download returned no records")
    write_json_records(cache_path, records, force=True)
    return cache_path, {
        "cache_path": str(cache_path),
        "download_method": "huggingface_file",
        "downloaded_at": downloaded_at or utc_now_iso(),
        "source_url": BFCL_SOURCE_URL,
    }


def _selected_bfcl_files(options: DatasetPrepareOptions) -> tuple[str, ...]:
    if not options.subset:
        return BFCL_DEFAULT_FILES
    return tuple(_normalize_bfcl_file_name(value) for value in options.subset)


def _normalize_bfcl_file_name(value: str) -> str:
    if value.endswith(".json"):
        return value
    if value.startswith("BFCL_"):
        return f"{value}.json"
    return f"BFCL_v3_{value}.json"


def _bfcl_file_url(file_name: str, revision: str | None) -> str:
    ref = revision or "main"
    return f"https://huggingface.co/datasets/{BFCL_HF_DATASET}/resolve/{ref}/{file_name}"


def _record_id(record: dict[str, Any], fallback_index: int) -> str:
    return str(record.get("id") or record.get("_id") or record.get("question_id") or fallback_index)


def _request(record: dict[str, Any]) -> str:
    value = record.get("prompt") or record.get("question") or record.get("instruction") or record.get("input")
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "").strip()


def _tools(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("tools") or record.get("functions") or record.get("function")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"name": value}]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else {"name": str(item)} for item in value]
    return []


def _openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool)
            continue
        name = tool.get("name") or tool.get("function_name") or tool.get("tool")
        if not name:
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        function: dict[str, Any] = {
            "name": str(name),
            "parameters": parameters,
        }
        if tool.get("description"):
            function["description"] = str(tool["description"])
        normalized.append({"type": "function", "function": function})
    return normalized


def _expected_tool(record: dict[str, Any]) -> str | None:
    for key in ("expected_tool", "tool", "function_name", "target_tool"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    answer = record.get("answer") or record.get("expected_answer") or record.get("ground_truth")
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            return answer.strip() or None
    if isinstance(answer, dict):
        for key in ("tool", "name", "function", "function_name"):
            value = answer.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _license_review_status(options: DatasetPrepareOptions) -> str:
    return "upstream_terms_need_user_review" if options.download else "fixture_only"


def _redistribution_policy(options: DatasetPrepareOptions) -> str:
    return "generated_workloads_not_redistributable" if options.download else "tiny_fixture_allowed"


def _slug(value: str | None) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")
