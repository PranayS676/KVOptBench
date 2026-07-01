"""Metric mapping registry shared by offline benchmark importers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal


MAPPING_REGISTRY_VERSION = "1"
Granularity = Literal["request", "aggregate", "any"]


@dataclass(frozen=True)
class MetricMapping:
    """One auditable mapping from external benchmark output to KVOptBench fields."""

    external_tool: str
    aliases: tuple[str, ...]
    normalized_field: str
    unit: str | None
    measurement_method: str
    loss_or_caveat: str | None = None
    required: bool = False
    granularity: Granularity = "any"
    statistic: str = "value"
    converter: str = "float"


@dataclass(frozen=True)
class SourceValue:
    """A source value plus the public-safe source field used to find it."""

    source_field: str
    value: Any


@dataclass(frozen=True)
class ResolvedMetric:
    """A parsed metric value with mapping metadata."""

    mapping: MetricMapping
    source_field: str
    value: Any


@dataclass(frozen=True)
class MetricMappingResult:
    """Mapped values, provenance, and missing metric explanations for one row."""

    values: dict[str, Any]
    metric_provenance: dict[str, dict[str, Any]]
    missing_metrics: list[str]
    missing_metric_reasons: dict[str, str]


def normalize_key(key: Any) -> str:
    """Normalize external field names while retaining enough shape for aliases."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def flatten_record(record: dict[str, Any]) -> dict[str, SourceValue]:
    """Flatten nested source objects into normalized lookup keys."""
    flat: dict[str, SourceValue] = {}

    def visit(prefix: str, value: Any, source_field: str) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_source = f"{source_field}.{child_key}" if source_field else str(child_key)
                child_prefix = f"{prefix}.{child_key}" if prefix else str(child_key)
                visit(child_prefix, child_value, child_source)
            return

        normalized = normalize_key(prefix)
        if normalized and normalized not in flat:
            flat[normalized] = SourceValue(source_field=source_field or prefix, value=value)

    for key, value in record.items():
        visit(str(key), value, str(key))
    return flat


def map_imported_metrics(
    raw_row: dict[str, Any],
    *,
    external_tool: str,
    expected_metrics: list[str],
    granularity: Granularity = "request",
) -> MetricMappingResult:
    """Map one request-like row and emit provenance for available and missing metrics."""
    resolved = available_metrics(raw_row, external_tool=external_tool, granularity=granularity)
    values: dict[str, Any] = {}
    metric_provenance: dict[str, dict[str, Any]] = {}

    for metric in resolved:
        field = metric.mapping.normalized_field
        if field in values:
            continue
        values[field] = metric.value
        metric_provenance[field] = _available_provenance(metric)

    missing_metrics: list[str] = []
    missing_metric_reasons: dict[str, str] = {}
    for field in expected_metrics:
        if values.get(field) is not None:
            continue
        mapping = first_mapping(external_tool, field, granularity)
        values[field] = None
        missing_metrics.append(field)
        reason = missing_reason(external_tool, field, granularity)
        missing_metric_reasons[field] = reason
        metric_provenance[field] = _missing_provenance(mapping, field, reason)

    return MetricMappingResult(
        values=values,
        metric_provenance=metric_provenance,
        missing_metrics=missing_metrics,
        missing_metric_reasons=missing_metric_reasons,
    )


def map_aggregate_metrics(
    raw_row: dict[str, Any],
    *,
    external_tool: str,
    expected_metrics: list[str],
) -> MetricMappingResult:
    """Map one aggregate row to summary-compatible metric columns."""
    resolved = available_metrics(raw_row, external_tool=external_tool, granularity="aggregate")
    values: dict[str, Any] = {}
    metric_provenance: dict[str, dict[str, Any]] = {}
    available_fields: set[str] = set()

    for metric in resolved:
        column = aggregate_column_name(metric.mapping)
        if column in values:
            continue
        values[column] = metric.value
        available_fields.add(metric.mapping.normalized_field)
        metric_provenance[column] = _available_provenance(metric)

    missing_metrics: list[str] = []
    missing_metric_reasons: dict[str, str] = {}
    for field in expected_metrics:
        if field in available_fields:
            continue
        mapping = first_mapping(external_tool, field, "aggregate")
        missing_metrics.append(field)
        reason = missing_reason(external_tool, field, "aggregate")
        missing_metric_reasons[field] = reason
        metric_provenance[field] = _missing_provenance(mapping, field, reason)

    return MetricMappingResult(
        values=values,
        metric_provenance=metric_provenance,
        missing_metrics=missing_metrics,
        missing_metric_reasons=missing_metric_reasons,
    )


def available_metrics(
    raw_row: dict[str, Any],
    *,
    external_tool: str,
    granularity: Granularity,
) -> list[ResolvedMetric]:
    """Return all parseable mapped metrics in registry order."""
    flat = flatten_record(raw_row)
    metrics: list[ResolvedMetric] = []
    for mapping in mappings_for(external_tool, granularity):
        for alias in mapping.aliases:
            source = flat.get(normalize_key(alias))
            if source is None:
                continue
            parsed = convert_value(source.value, mapping.converter)
            if parsed is None:
                continue
            metrics.append(
                ResolvedMetric(mapping=mapping, source_field=source.source_field, value=parsed)
            )
            break
    return metrics


def mappings_for(external_tool: str, granularity: Granularity = "any") -> list[MetricMapping]:
    """Return mappings for a tool and granularity, with generic mappings included."""
    return [
        mapping
        for mapping in METRIC_MAPPINGS
        if mapping.external_tool == external_tool
        and (mapping.granularity == "any" or granularity == "any" or mapping.granularity == granularity)
    ]


def first_mapping(
    external_tool: str,
    normalized_field: str,
    granularity: Granularity,
) -> MetricMapping | None:
    """Find the primary registry row for a normalized field."""
    for mapping in mappings_for(external_tool, granularity):
        if mapping.normalized_field == normalized_field:
            return mapping
    return None


def expected_metric_fields(external_tool: str, granularity: Granularity) -> list[str]:
    """Metric fields that should be explicit when missing for a tool/mode."""
    return list(EXPECTED_METRICS.get((external_tool, granularity), ()))


def required_metric_fields(external_tool: str, granularity: Granularity) -> list[str]:
    """Metric fields that make an import unusable when missing."""
    required: list[str] = []
    for mapping in mappings_for(external_tool, granularity):
        if not mapping.required or mapping.normalized_field in required:
            continue
        required.append(mapping.normalized_field)
    return required


def mapping_registry_payload(
    *,
    external_tool: str | None = None,
    granularity: Granularity = "any",
) -> dict[str, Any]:
    """Return the public mapping registry metadata used by import adapters."""
    mappings = [
        asdict(mapping)
        for mapping in METRIC_MAPPINGS
        if (external_tool is None or mapping.external_tool == external_tool)
        and (
            granularity == "any"
            or mapping.granularity == "any"
            or mapping.granularity == granularity
        )
    ]
    return {
        "mapping_registry_version": MAPPING_REGISTRY_VERSION,
        "tool": external_tool,
        "granularity": granularity,
        "expected_metrics": (
            expected_metric_fields(external_tool, granularity)
            if external_tool is not None and granularity != "any"
            else []
        ),
        "required_metrics": (
            required_metric_fields(external_tool, granularity)
            if external_tool is not None and granularity != "any"
            else []
        ),
        "mappings": mappings,
    }


def metric_aliases(external_tool: str, normalized_field: str, granularity: Granularity) -> list[str]:
    """Human-readable alias list used in missing-metric reasons."""
    aliases: list[str] = []
    for mapping in mappings_for(external_tool, granularity):
        if mapping.normalized_field == normalized_field:
            aliases.extend(mapping.aliases)
    return aliases


def aggregate_column_name(mapping: MetricMapping) -> str:
    """Render an aggregate metric as a summary-compatible column name."""
    if mapping.statistic in {"value", "count"}:
        return mapping.normalized_field
    return f"{mapping.normalized_field}_{mapping.statistic}"


def missing_reason(external_tool: str, normalized_field: str, granularity: Granularity) -> str:
    """Build a deterministic missing reason for one expected metric."""
    aliases = metric_aliases(external_tool, normalized_field, granularity)
    tool_label = TOOL_LABELS.get(external_tool, external_tool)
    if aliases:
        return f"No {tool_label} field matched aliases: {', '.join(aliases)}."
    return f"No {tool_label} mapping is registered for {normalized_field}."


def convert_value(value: Any, converter: str) -> Any:
    """Convert one source value according to the registry converter."""
    if converter == "string":
        text = str(value).strip() if value is not None else ""
        return text or None
    if converter == "bool":
        return to_bool(value)

    parsed = to_float(value)
    if parsed is None:
        return None
    if converter in {"int", "count"}:
        return int(parsed)
    if converter in {"float", "milliseconds", "ratio"}:
        return float(parsed)
    if converter == "seconds_to_ms":
        return float(parsed) * 1000
    if converter == "microseconds_to_ms":
        return float(parsed) / 1000
    if converter == "nanoseconds_to_ms":
        return float(parsed) / 1_000_000
    if converter == "percent_to_ratio":
        return float(parsed) / 100
    raise ValueError(f"Unsupported metric converter: {converter}")


def to_float(value: Any) -> float | None:
    """Parse numeric values without converting missing data to zero."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    parsed = to_float(value)
    return int(parsed) if parsed is not None else None


def to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "ok", "success", "succeeded"}:
        return True
    if normalized in {"false", "0", "no", "n", "error", "failed", "failure"}:
        return False
    return None


def first_string(row: dict[str, Any], aliases: list[str]) -> str | None:
    flat = flatten_record(row)
    for alias in aliases:
        source = flat.get(normalize_key(alias))
        if source is None:
            continue
        value = convert_value(source.value, "string")
        if value is not None:
            return value
    return None


def first_int(row: dict[str, Any], aliases: list[str]) -> int | None:
    flat = flatten_record(row)
    for alias in aliases:
        source = flat.get(normalize_key(alias))
        if source is None:
            continue
        value = convert_value(source.value, "int")
        if value is not None:
            return value
    return None


def first_bool(row: dict[str, Any], aliases: list[str]) -> bool | None:
    flat = flatten_record(row)
    for alias in aliases:
        source = flat.get(normalize_key(alias))
        if source is None:
            continue
        value = convert_value(source.value, "bool")
        if value is not None:
            return value
    return None


def raw_field_names(row: dict[str, Any]) -> list[str]:
    """Return raw source field names only, including nested leaf paths."""
    return sorted(source.source_field for source in flatten_record(row).values())


def _available_provenance(metric: ResolvedMetric) -> dict[str, Any]:
    mapping = metric.mapping
    return {
        "source_type": "imported",
        "source_field": metric.source_field,
        "provider_field": metric.source_field,
        "normalized_field": mapping.normalized_field,
        "unit": mapping.unit,
        "measurement_method": mapping.measurement_method,
        "available": True,
        "statistic": mapping.statistic,
        "notes": mapping.loss_or_caveat,
    }


def _missing_provenance(
    mapping: MetricMapping | None,
    normalized_field: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "source_type": "imported",
        "source_field": None,
        "provider_field": None,
        "normalized_field": normalized_field,
        "unit": mapping.unit if mapping else None,
        "measurement_method": (
            mapping.measurement_method if mapping else "Imported benchmark field mapping"
        ),
        "available": False,
        "statistic": mapping.statistic if mapping else "value",
        "missing_reason": reason,
        "notes": mapping.loss_or_caveat if mapping else None,
    }


def _mapping(
    external_tool: str,
    aliases: tuple[str, ...],
    normalized_field: str,
    unit: str | None,
    measurement_method: str,
    *,
    loss_or_caveat: str | None = None,
    required: bool = False,
    granularity: Granularity = "any",
    statistic: str = "value",
    converter: str = "float",
) -> MetricMapping:
    return MetricMapping(
        external_tool=external_tool,
        aliases=aliases,
        normalized_field=normalized_field,
        unit=unit,
        measurement_method=measurement_method,
        loss_or_caveat=loss_or_caveat,
        required=required,
        granularity=granularity,
        statistic=statistic,
        converter=converter,
    )


TOOL_LABELS = {
    "vllm_bench": "vLLM bench",
    "genai_perf": "GenAI-Perf",
    "aiperf": "AIPerf",
}

EXPECTED_METRICS: dict[tuple[str, Granularity], tuple[str, ...]] = {
    (
        "vllm_bench",
        "request",
    ): (
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "tpot_ms",
        "e2e_latency_ms",
        "gpu_memory_used_gb",
        "gpu_memory_peak_gb",
    ),
    ("genai_perf", "request"): (
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "itl_ms",
        "e2e_latency_ms",
    ),
    ("genai_perf", "aggregate"): (
        "requests",
        "ttft_ms",
        "itl_ms",
        "output_tokens_per_second",
        "requests_per_second",
    ),
    ("aiperf", "request"): (
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "itl_ms",
        "e2e_latency_ms",
    ),
    ("aiperf", "aggregate"): (
        "requests",
        "errors",
        "success_rate",
        "ttft_ms",
        "itl_ms",
        "e2e_latency_ms",
        "requests_per_second",
        "output_tokens_per_second",
    ),
}

METRIC_MAPPINGS: list[MetricMapping] = [
    _mapping(
        "vllm_bench",
        ("input_tokens", "num_input_tokens", "prompt_tokens", "input_len", "prompt_len"),
        "input_tokens",
        "tokens",
        "vLLM bench reported input token count",
        loss_or_caveat="Alias source differs by vLLM bench export.",
        required=True,
        granularity="request",
        converter="int",
    ),
    _mapping(
        "vllm_bench",
        (
            "output_tokens",
            "num_output_tokens",
            "completion_tokens",
            "output_len",
            "generated_tokens",
        ),
        "output_tokens",
        "tokens",
        "vLLM bench reported output token count",
        loss_or_caveat="Alias source differs by vLLM bench export.",
        required=True,
        granularity="request",
        converter="int",
    ),
    _mapping(
        "vllm_bench",
        ("ttft_ms", "time_to_first_token_ms", "mean_ttft_ms", "median_ttft_ms"),
        "ttft_ms",
        "ms",
        "vLLM bench reported time to first token",
        loss_or_caveat="Aggregate aliases must be interpreted with source granularity.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "vllm_bench",
        ("tpot_ms", "time_per_output_token_ms", "mean_tpot_ms", "median_tpot_ms"),
        "tpot_ms",
        "ms",
        "vLLM bench reported time per output token",
        loss_or_caveat="Aggregate aliases must be interpreted with source granularity.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "vllm_bench",
        (
            "e2e_latency_ms",
            "latency_ms",
            "request_latency_ms",
            "end_to_end_latency_ms",
            "mean_latency_ms",
        ),
        "e2e_latency_ms",
        "ms",
        "vLLM bench reported end-to-end request latency",
        loss_or_caveat="End-to-end latency definitions vary by exporter.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "vllm_bench",
        ("requests_per_second", "request_throughput"),
        "requests_per_second",
        "requests/s",
        "vLLM bench reported request throughput",
        granularity="request",
    ),
    _mapping(
        "vllm_bench",
        ("input_tokens_per_second", "input_throughput"),
        "input_tokens_per_second",
        "tokens/s",
        "vLLM bench reported input token throughput",
        granularity="request",
    ),
    _mapping(
        "vllm_bench",
        ("output_tokens_per_second", "output_throughput"),
        "output_tokens_per_second",
        "tokens/s",
        "vLLM bench reported output token throughput",
        granularity="request",
    ),
    _mapping(
        "vllm_bench",
        ("gpu_memory_used_gb", "memory_used_gb", "gpu_memory_used"),
        "gpu_memory_used_gb",
        "GB",
        "vLLM bench reported GPU memory snapshot",
        loss_or_caveat="May be absent on client-only exports.",
        granularity="request",
    ),
    _mapping(
        "vllm_bench",
        ("gpu_memory_peak_gb", "peak_gpu_memory_gb", "gpu_memory_peak"),
        "gpu_memory_peak_gb",
        "GB",
        "vLLM bench reported peak GPU memory",
        loss_or_caveat="May be absent on client-only exports.",
        granularity="request",
    ),
    _mapping(
        "genai_perf",
        ("input_tokens", "input_token_count", "prompt_tokens", "input_sequence_length"),
        "input_tokens",
        "tokens",
        "GenAI-Perf reported input token count",
        loss_or_caveat="Compatibility import for historical GenAI-Perf artifacts.",
        required=True,
        granularity="request",
        converter="int",
    ),
    _mapping(
        "genai_perf",
        ("output_tokens", "output_token_count", "completion_tokens", "output_sequence_length"),
        "output_tokens",
        "tokens",
        "GenAI-Perf reported output token count",
        loss_or_caveat="Compatibility import for historical GenAI-Perf artifacts.",
        required=True,
        granularity="request",
        converter="int",
    ),
    _mapping(
        "genai_perf",
        ("ttft_ms", "time_to_first_token_ms", "time_to_first_token_ms_mean"),
        "ttft_ms",
        "ms",
        "GenAI-Perf reported time to first token",
        loss_or_caveat=(
            "Compatibility path only; reasoning-token semantics can differ from AIPerf."
        ),
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "genai_perf",
        ("time_to_first_token_s", "time_to_first_token_seconds"),
        "ttft_ms",
        "ms",
        "GenAI-Perf reported time to first token converted to milliseconds",
        loss_or_caveat=(
            "Compatibility path only; reasoning-token semantics can differ from AIPerf."
        ),
        granularity="request",
        converter="seconds_to_ms",
    ),
    _mapping(
        "genai_perf",
        ("itl_ms", "inter_token_latency_ms", "inter_token_latency_ms_mean"),
        "itl_ms",
        "ms",
        "GenAI-Perf reported inter-token latency",
        loss_or_caveat="Exact statistic depends on export row.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "genai_perf",
        ("request_latency_ms", "latency_ms", "end_to_end_latency_ms"),
        "e2e_latency_ms",
        "ms",
        "GenAI-Perf reported request latency",
        loss_or_caveat="End-to-end definition varies by source exporter.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "genai_perf",
        ("output_token_throughput", "output_tokens_per_second", "output_throughput"),
        "output_tokens_per_second",
        "tokens/s",
        "GenAI-Perf reported output token throughput",
        loss_or_caveat="Aggregate metric when sourced from summary exports.",
        granularity="any",
        statistic="mean",
    ),
    _mapping(
        "genai_perf",
        ("request_throughput", "requests_per_second", "request_throughput_per_sec"),
        "requests_per_second",
        "requests/s",
        "GenAI-Perf reported request throughput",
        loss_or_caveat="Aggregate metric when sourced from summary exports.",
        granularity="any",
        statistic="mean",
    ),
    _mapping(
        "genai_perf",
        ("request_count", "requests", "num_requests"),
        "requests",
        "requests",
        "GenAI-Perf reported request count",
        granularity="aggregate",
        converter="count",
    ),
    _mapping(
        "genai_perf",
        ("ttft_mean_ms", "mean_ttft_ms", "time_to_first_token_mean_ms"),
        "ttft_ms",
        "ms",
        "GenAI-Perf aggregate mean time to first token",
        loss_or_caveat=(
            "Compatibility path only; reasoning-token semantics can differ from AIPerf."
        ),
        granularity="aggregate",
        statistic="mean",
        converter="milliseconds",
    ),
    _mapping(
        "genai_perf",
        ("ttft_p50_ms", "p50_ttft_ms", "time_to_first_token_p50_ms"),
        "ttft_ms",
        "ms",
        "GenAI-Perf aggregate p50 time to first token",
        loss_or_caveat=(
            "Compatibility path only; reasoning-token semantics can differ from AIPerf."
        ),
        granularity="aggregate",
        statistic="p50",
        converter="milliseconds",
    ),
    _mapping(
        "genai_perf",
        ("ttft_p95_ms", "p95_ttft_ms", "time_to_first_token_p95_ms"),
        "ttft_ms",
        "ms",
        "GenAI-Perf aggregate p95 time to first token",
        loss_or_caveat=(
            "Compatibility path only; reasoning-token semantics can differ from AIPerf."
        ),
        granularity="aggregate",
        statistic="p95",
        converter="milliseconds",
    ),
    _mapping(
        "genai_perf",
        ("itl_mean_ms", "mean_itl_ms", "inter_token_latency_mean_ms"),
        "itl_ms",
        "ms",
        "GenAI-Perf aggregate mean inter-token latency",
        loss_or_caveat="Exact statistic depends on export row.",
        granularity="aggregate",
        statistic="mean",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("input_tokens", "input_token_count", "prompt_tokens", "usage_input_tokens"),
        "input_tokens",
        "tokens",
        "AIPerf reported input token count",
        loss_or_caveat="May be tokenizer-derived instead of provider usage.",
        required=True,
        granularity="request",
        converter="int",
    ),
    _mapping(
        "aiperf",
        ("output_tokens", "output_token_count", "completion_tokens", "usage_output_tokens"),
        "output_tokens",
        "tokens",
        "AIPerf reported output token count",
        loss_or_caveat="May exclude reasoning tokens depending on export field.",
        required=True,
        granularity="request",
        converter="int",
    ),
    _mapping(
        "aiperf",
        (
            "time_to_first_token_ms",
            "time_to_first_output_token_ms",
            "ttft_ms",
            "ttfo_ms",
        ),
        "ttft_ms",
        "ms",
        "AIPerf reported first token timing",
        loss_or_caveat=(
            "Compare to GenAI-Perf carefully for reasoning-capable models; TTFO may be "
            "the closer compatibility metric."
        ),
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("time_to_first_token", "time_to_first_output_token"),
        "ttft_ms",
        "ms",
        "AIPerf reported first token timing",
        loss_or_caveat="Source field has no unit suffix; value is treated as milliseconds.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("time_to_first_token_s", "time_to_first_output_token_s"),
        "ttft_ms",
        "ms",
        "AIPerf reported first token timing converted to milliseconds",
        loss_or_caveat=(
            "Compare to GenAI-Perf carefully for reasoning-capable models; TTFO may be "
            "the closer compatibility metric."
        ),
        granularity="request",
        converter="seconds_to_ms",
    ),
    _mapping(
        "aiperf",
        ("inter_token_latency_ms", "itl_ms"),
        "itl_ms",
        "ms",
        "AIPerf reported inter-token latency",
        loss_or_caveat="Streaming or token-producing endpoints only.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("inter_token_latency",),
        "itl_ms",
        "ms",
        "AIPerf reported inter-token latency",
        loss_or_caveat="Source field has no unit suffix; value is treated as milliseconds.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("request_latency_ms", "latency_ms", "end_to_end_latency_ms"),
        "e2e_latency_ms",
        "ms",
        "AIPerf reported request latency",
        loss_or_caveat="Duration conversion depends on export units.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("request_latency",),
        "e2e_latency_ms",
        "ms",
        "AIPerf reported request latency",
        loss_or_caveat="Source field has no unit suffix; value is treated as milliseconds.",
        granularity="request",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("request_latency_s", "latency_s"),
        "e2e_latency_ms",
        "ms",
        "AIPerf reported request latency converted to milliseconds",
        loss_or_caveat="Duration conversion depends on export units.",
        granularity="request",
        converter="seconds_to_ms",
    ),
    _mapping(
        "aiperf",
        ("request_throughput", "requests_per_second"),
        "requests_per_second",
        "requests/s",
        "AIPerf reported request throughput",
        loss_or_caveat="Aggregate metric.",
        granularity="any",
        statistic="mean",
    ),
    _mapping(
        "aiperf",
        ("output_token_throughput", "output_tokens_per_second", "output_throughput"),
        "output_tokens_per_second",
        "tokens/s",
        "AIPerf reported output token throughput",
        loss_or_caveat="Aggregate metric.",
        granularity="any",
        statistic="mean",
    ),
    _mapping(
        "aiperf",
        ("request_count", "requests", "num_requests"),
        "requests",
        "requests",
        "AIPerf reported request count",
        granularity="aggregate",
        converter="count",
    ),
    _mapping(
        "aiperf",
        ("error_count", "errors", "failed_requests"),
        "errors",
        "requests",
        "AIPerf reported error count",
        granularity="aggregate",
        converter="count",
    ),
    _mapping(
        "aiperf",
        ("success_rate", "request_success_rate"),
        "success_rate",
        "ratio",
        "AIPerf reported request success rate",
        granularity="aggregate",
        converter="ratio",
    ),
    _mapping(
        "aiperf",
        ("success_rate_percent", "request_success_rate_percent"),
        "success_rate",
        "ratio",
        "AIPerf reported request success rate converted from percent",
        granularity="aggregate",
        converter="percent_to_ratio",
    ),
    _mapping(
        "aiperf",
        ("ttft_mean_ms", "time_to_first_token_mean_ms", "time_to_first_output_token_mean_ms"),
        "ttft_ms",
        "ms",
        "AIPerf aggregate mean first token timing",
        loss_or_caveat=(
            "Compare to GenAI-Perf carefully for reasoning-capable models; TTFO may be "
            "the closer compatibility metric."
        ),
        granularity="aggregate",
        statistic="mean",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("ttft_p50_ms", "time_to_first_token_p50_ms", "time_to_first_output_token_p50_ms"),
        "ttft_ms",
        "ms",
        "AIPerf aggregate p50 first token timing",
        granularity="aggregate",
        statistic="p50",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("ttft_p95_ms", "time_to_first_token_p95_ms", "time_to_first_output_token_p95_ms"),
        "ttft_ms",
        "ms",
        "AIPerf aggregate p95 first token timing",
        granularity="aggregate",
        statistic="p95",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("itl_mean_ms", "inter_token_latency_mean_ms"),
        "itl_ms",
        "ms",
        "AIPerf aggregate mean inter-token latency",
        loss_or_caveat="Streaming or token-producing endpoints only.",
        granularity="aggregate",
        statistic="mean",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("request_latency_mean_ms", "latency_mean_ms", "e2e_latency_mean_ms"),
        "e2e_latency_ms",
        "ms",
        "AIPerf aggregate mean request latency",
        loss_or_caveat="Duration conversion depends on export units.",
        granularity="aggregate",
        statistic="mean",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("request_latency_p95_ms", "latency_p95_ms", "e2e_latency_p95_ms"),
        "e2e_latency_ms",
        "ms",
        "AIPerf aggregate p95 request latency",
        loss_or_caveat="Duration conversion depends on export units.",
        granularity="aggregate",
        statistic="p95",
        converter="milliseconds",
    ),
    _mapping(
        "aiperf",
        ("error_type", "error.type"),
        "error_type",
        "label",
        "AIPerf error classification",
        loss_or_caveat="Only present for failed requests.",
        granularity="request",
        converter="string",
    ),
]
