"""Shared adapter pipeline for external benchmark artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from kvoptbench.importers.metrics import (
    MAPPING_REGISTRY_VERSION,
    available_metrics,
    expected_metric_fields,
    first_bool,
    first_int,
    first_string,
    map_aggregate_metrics,
    map_imported_metrics,
    normalize_key,
    raw_field_names,
)
from kvoptbench.importers.reader import SourceRecordSet, read_source_records


ADAPTER_VERSION = "1"
ImportGranularity = Literal["auto", "request", "aggregate"]


@dataclass(frozen=True)
class ImportAdapterResult:
    """Normalized output from one import adapter invocation."""

    tool: str
    adapter_version: str
    mapping_registry_version: str
    granularity: Literal["request", "aggregate"]
    request_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    source_manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def import_external_benchmark(
    source: str | Path,
    *,
    external_tool: str,
    experiment_id: str,
    workload: str,
    provider: str = "local",
    engine: str = "unknown",
    strategy: str = "imported",
    model_id: str | None = None,
    run_id: str | None = None,
    concurrency: int = 1,
    granularity: ImportGranularity = "auto",
) -> ImportAdapterResult:
    """Import a GenAI-Perf or AIPerf artifact without running any benchmark."""
    source_records = read_source_records(source)
    detected_granularity = detect_granularity(
        source_records.records,
        external_tool=external_tool,
        requested_granularity=granularity,
    )
    manifest = source_manifest(source_records, external_tool)

    if detected_granularity == "request":
        rows = [
            _request_row(
                raw_row,
                row_index=index,
                source_records=source_records,
                external_tool=external_tool,
                experiment_id=experiment_id,
                workload=workload,
                provider=provider,
                engine=engine,
                strategy=strategy,
                model_id=model_id,
                run_id=run_id,
                concurrency=concurrency,
            )
            for index, raw_row in enumerate(source_records.records)
        ]
        return ImportAdapterResult(
            tool=external_tool,
            adapter_version=ADAPTER_VERSION,
            mapping_registry_version=MAPPING_REGISTRY_VERSION,
            granularity="request",
            request_rows=rows,
            missing_metrics=_collect_missing(rows),
            source_manifest=manifest,
        )

    aggregate_records = coerce_aggregate_records(source_records.records)
    rows = [
        _summary_row(
            raw_row,
            row_index=index,
            source_records=source_records,
            external_tool=external_tool,
            experiment_id=experiment_id,
            workload=workload,
            provider=provider,
            engine=engine,
            strategy=strategy,
            model_id=model_id,
            concurrency=concurrency,
        )
        for index, raw_row in enumerate(aggregate_records)
    ]
    return ImportAdapterResult(
        tool=external_tool,
        adapter_version=ADAPTER_VERSION,
        mapping_registry_version=MAPPING_REGISTRY_VERSION,
        granularity="aggregate",
        summary_rows=rows,
        missing_metrics=_collect_missing(rows),
        source_manifest=manifest,
    )


def detect_granularity(
    records: list[dict[str, Any]],
    *,
    external_tool: str,
    requested_granularity: ImportGranularity = "auto",
) -> Literal["request", "aggregate"]:
    """Detect source granularity without promoting aggregate data to requests."""
    if requested_granularity in {"request", "aggregate"}:
        return requested_granularity
    if _looks_metric_table(records):
        return "aggregate"

    request_like = [
        _looks_request_record(row, external_tool=external_tool)
        and not _looks_aggregate_record(row, external_tool=external_tool)
        for row in records
    ]
    aggregate_like = [
        _looks_aggregate_record(row, external_tool=external_tool) for row in records
    ]

    if records and all(request_like):
        return "request"
    if records and any(request_like) and len(records) > 1 and not any(aggregate_like):
        return "request"
    if records and all(aggregate_like):
        return "aggregate"
    if len(records) == 1 and request_like[0]:
        return "request"
    raise ValueError(
        "Ambiguous import granularity; pass granularity='request' or "
        "granularity='aggregate'."
    )


def coerce_aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse common metric/value tables into a single aggregate row."""
    if not _looks_metric_table(records):
        return records

    metric_col = _first_present_key(records[0], ["metric", "metric_name", "name", "field"])
    value_col = _first_present_key(records[0], ["value", "mean", "avg"])
    statistic_col = _first_present_key(records[0], ["statistic", "stat", "percentile"])
    if metric_col is None or value_col is None:
        return records

    aggregate: dict[str, Any] = {}
    for row in records:
        metric_name = str(row.get(metric_col, "")).strip()
        if not metric_name:
            continue
        statistic = str(row.get(statistic_col, "")).strip() if statistic_col else ""
        key = f"{metric_name}_{statistic}" if statistic else metric_name
        aggregate[key] = row.get(value_col)
    return [aggregate]


def source_manifest(source_records: SourceRecordSet, external_tool: str) -> dict[str, Any]:
    """Build public-safe source metadata for manifests or CLI wiring."""
    return {
        "schema_version": "1",
        "tool": external_tool,
        "adapter_version": ADAPTER_VERSION,
        "mapping_registry_version": MAPPING_REGISTRY_VERSION,
        "source": {
            "file_name": source_records.source_file_name,
            "format": source_records.source_format,
            "sha256": source_records.source_sha256,
            "sanitized_label": sanitize_source_label(source_records.source_file_name),
        },
        "warnings": [],
    }


def sanitize_source_label(source_file_name: str) -> str:
    """Create a deterministic public-safe label from a file name."""
    stem = Path(source_file_name).stem
    label = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return label or "import-source"


def _request_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    source_records: SourceRecordSet,
    external_tool: str,
    experiment_id: str,
    workload: str,
    provider: str,
    engine: str,
    strategy: str,
    model_id: str | None,
    run_id: str | None,
    concurrency: int,
) -> dict[str, Any]:
    mapped = map_imported_metrics(
        raw_row,
        external_tool=external_tool,
        expected_metrics=expected_metric_fields(external_tool, "request"),
        granularity="request",
    )
    values = mapped.values
    error_message = first_string(raw_row, ["error_message", "error.message", "exception"])
    error_type = values.get("error_type") or first_string(raw_row, ["error_type", "error.type"])
    success = first_bool(raw_row, ["success", "succeeded", "completed", "ok"])
    if success is None:
        success = error_message is None and error_type is None

    return {
        "run_id": run_id or f"{external_tool}-import",
        "experiment_id": experiment_id,
        "official_run": False,
        "provider": provider,
        "engine": engine,
        "model_id": model_id or first_string(raw_row, ["model_id", "model", "model_name"]) or "unknown",
        "strategy": strategy,
        "workload": workload,
        "task_id": _task_id(raw_row, external_tool, row_index),
        "concurrency": first_int(raw_row, ["concurrency"]) or concurrency,
        "input_tokens": values.get("input_tokens"),
        "output_tokens": values.get("output_tokens"),
        "ttft_ms": values.get("ttft_ms"),
        "tpot_ms": values.get("tpot_ms"),
        "itl_ms": values.get("itl_ms"),
        "e2e_latency_ms": values.get("e2e_latency_ms"),
        "requests_per_second": values.get("requests_per_second"),
        "input_tokens_per_second": values.get("input_tokens_per_second"),
        "output_tokens_per_second": values.get("output_tokens_per_second"),
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
        "missing_metrics": mapped.missing_metrics,
        "metric_provenance": mapped.metric_provenance,
        "metadata": _metadata(
            raw_row,
            row_index=row_index,
            source_records=source_records,
            external_tool=external_tool,
            granularity="request",
            missing_metric_reasons=mapped.missing_metric_reasons,
        ),
    }


def _summary_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    source_records: SourceRecordSet,
    external_tool: str,
    experiment_id: str,
    workload: str,
    provider: str,
    engine: str,
    strategy: str,
    model_id: str | None,
    concurrency: int,
) -> dict[str, Any]:
    mapped = map_aggregate_metrics(
        raw_row,
        external_tool=external_tool,
        expected_metrics=expected_metric_fields(external_tool, "aggregate"),
    )
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "provider": provider,
        "engine": engine,
        "model_id": model_id or first_string(raw_row, ["model_id", "model", "model_name"]) or "unknown",
        "strategy": strategy,
        "workload": workload,
        "concurrency": first_int(raw_row, ["concurrency"]) or concurrency,
    }
    row.update(mapped.values)
    row.update(
        {
            "missing_metrics": mapped.missing_metrics,
            "metric_provenance": mapped.metric_provenance,
            "metadata": _metadata(
                raw_row,
                row_index=row_index,
                source_records=source_records,
                external_tool=external_tool,
                granularity="aggregate",
                missing_metric_reasons=mapped.missing_metric_reasons,
            ),
        }
    )
    return row


def _metadata(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    source_records: SourceRecordSet,
    external_tool: str,
    granularity: Literal["request", "aggregate"],
    missing_metric_reasons: dict[str, str],
) -> dict[str, Any]:
    return {
        "source": external_tool,
        "source_file_name": source_records.source_file_name,
        "source_sha256": source_records.source_sha256,
        "source_format": source_records.source_format,
        "source_row_index": row_index,
        "import_granularity": granularity,
        "adapter_version": ADAPTER_VERSION,
        "mapping_registry_version": MAPPING_REGISTRY_VERSION,
        "missing_metric_reasons": missing_metric_reasons,
        "raw_fields": raw_field_names(raw_row),
    }


def _looks_metric_table(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    first = {normalize_key(key) for key in records[0]}
    has_metric = bool(first.intersection({"metric", "metric_name", "name", "field"}))
    has_value = bool(first.intersection({"value", "mean", "avg"}))
    return has_metric and has_value


def _looks_request_record(raw_row: dict[str, Any], *, external_tool: str) -> bool:
    if first_string(raw_row, ["request_id", "task_id", "id"]):
        return True
    resolved = available_metrics(raw_row, external_tool=external_tool, granularity="request")
    fields = {metric.mapping.normalized_field for metric in resolved}
    return bool({"input_tokens", "output_tokens"}.intersection(fields)) and bool(
        {"ttft_ms", "itl_ms", "e2e_latency_ms"}.intersection(fields)
    )


def _looks_aggregate_record(raw_row: dict[str, Any], *, external_tool: str) -> bool:
    normalized_keys = {normalize_key(key) for key in raw_field_names(raw_row)}
    aggregate_markers = {
        "request_count",
        "requests",
        "num_requests",
        "error_count",
        "errors",
        "success_rate",
        "request_throughput",
        "requests_per_second",
        "output_token_throughput",
    }
    has_statistic_field = any(
        key.startswith(("mean_", "p50_", "p95_"))
        or key.endswith(("_mean_ms", "_p50_ms", "_p95_ms", "_mean"))
        for key in normalized_keys
    )
    if normalized_keys.intersection(aggregate_markers) or has_statistic_field:
        return True
    resolved = available_metrics(raw_row, external_tool=external_tool, granularity="aggregate")
    return bool(resolved)


def _first_present_key(row: dict[str, Any], aliases: list[str]) -> str | None:
    normalized_aliases = {normalize_key(alias) for alias in aliases}
    for key in row:
        if normalize_key(key) in normalized_aliases:
            return key
    return None


def _task_id(raw_row: dict[str, Any], external_tool: str, row_index: int) -> str:
    return (
        first_string(raw_row, ["task_id", "request_id", "id"])
        or f"{external_tool}-request-{row_index + 1}"
    )


def _collect_missing(rows: list[dict[str, Any]]) -> list[str]:
    missing: set[str] = set()
    for row in rows:
        for metric in row.get("missing_metrics", []):
            missing.add(str(metric))
    return sorted(missing)
