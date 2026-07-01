"""Import vLLM bench artifacts into KVOptBench-like rows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kvoptbench.importers.metrics import (
    MAPPING_REGISTRY_VERSION,
    expected_metric_fields,
    first_bool,
    first_string,
    map_imported_metrics,
    raw_field_names,
)
from kvoptbench.importers.reader import SourceRecordSet, read_source_records


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VllmBenchImportRow(BaseModel):
    """A KVOptBench-like imported row that can feed summary/report tooling later."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    experiment_id: str
    provider: str = "local"
    engine: str = "vllm"
    model_id: str
    strategy: str = "imported"
    workload: str
    task_id: str
    concurrency: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_latency_ms: float | None = None
    requests_per_second: float | None = None
    input_tokens_per_second: float | None = None
    output_tokens_per_second: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_peak_gb: float | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    created_at: str = Field(default_factory=_utc_now_iso)
    missing_metrics: list[str] = Field(default_factory=list)
    metric_provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def import_vllm_bench(
    source: str | Path,
    *,
    experiment_id: str,
    workload: str,
    provider: str = "local",
    engine: str = "vllm",
    strategy: str = "imported",
    model_id: str | None = None,
    run_id: str | None = None,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Read a local vLLM bench JSON/JSONL/CSV artifact into KVOptBench-like rows."""
    source_path = Path(source)
    source_records = read_source_records(source_path)
    imported: list[dict[str, Any]] = []

    for index, raw_row in enumerate(source_records.records):
        row = _normalize_row(
            raw_row,
            row_index=index,
            source_records=source_records,
            experiment_id=experiment_id,
            workload=workload,
            provider=provider,
            engine=engine,
            strategy=strategy,
            model_id=model_id,
            run_id=run_id,
            concurrency=concurrency,
        )
        imported.append(row.model_dump(mode="json"))

    return imported


def _normalize_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    source_records: SourceRecordSet,
    experiment_id: str,
    workload: str,
    provider: str,
    engine: str,
    strategy: str,
    model_id: str | None,
    run_id: str | None,
    concurrency: int,
) -> VllmBenchImportRow:
    mapped = map_imported_metrics(
        raw_row,
        external_tool="vllm_bench",
        expected_metrics=expected_metric_fields("vllm_bench", "request"),
        granularity="request",
    )
    values = mapped.values
    resolved_model = (
        model_id or first_string(raw_row, ["model_id", "model", "model_name"]) or "unknown"
    )
    task_id = first_string(raw_row, ["task_id", "request_id", "id"]) or f"vllm-bench-{row_index + 1}"

    error_message = first_string(raw_row, ["error_message", "error", "exception"])
    success = first_bool(raw_row, ["success", "succeeded", "completed"])
    if success is None:
        success = error_message is None

    return VllmBenchImportRow(
        run_id=run_id or "vllm-bench-import",
        experiment_id=experiment_id,
        provider=provider,
        engine=engine,
        model_id=resolved_model,
        strategy=strategy,
        workload=workload,
        task_id=task_id,
        concurrency=concurrency,
        input_tokens=values.get("input_tokens"),
        output_tokens=values.get("output_tokens"),
        ttft_ms=values["ttft_ms"],
        tpot_ms=values["tpot_ms"],
        e2e_latency_ms=values["e2e_latency_ms"],
        requests_per_second=values.get("requests_per_second"),
        input_tokens_per_second=values.get("input_tokens_per_second"),
        output_tokens_per_second=values.get("output_tokens_per_second"),
        gpu_memory_used_gb=values.get("gpu_memory_used_gb"),
        gpu_memory_peak_gb=values.get("gpu_memory_peak_gb"),
        success=success,
        error_type=first_string(raw_row, ["error_type"]),
        error_message=error_message,
        missing_metrics=mapped.missing_metrics,
        metric_provenance=mapped.metric_provenance,
        metadata={
            "source": "vllm_bench",
            "source_path": source_records.source_file_name,
            "source_file_name": source_records.source_file_name,
            "source_sha256": source_records.source_sha256,
            "source_format": source_records.source_format,
            "source_row_index": row_index,
            "import_granularity": "request",
            "adapter_version": "1",
            "mapping_registry_version": MAPPING_REGISTRY_VERSION,
            "missing_metric_reasons": mapped.missing_metric_reasons,
            "raw_fields": raw_field_names(raw_row),
        },
    )
