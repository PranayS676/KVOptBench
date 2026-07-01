"""Metric provenance helpers for request-level benchmark rows."""

from __future__ import annotations

from kvoptbench.schemas import MetricProvenance, RunEnvironmentSnapshot, TimedResponse


UNAVAILABLE_ENGINE_METRIC_REASON = (
    "The configured endpoint did not expose this engine telemetry metric."
)
UNAVAILABLE_GPU_METRIC_REASON = "GPU telemetry was not collected for this run."


def build_metric_provenance(
    *,
    response: TimedResponse,
    metadata: dict,
    missing_metrics: list[str],
    output_tps_available: bool,
    input_tps_available: bool,
    requests_per_second_available: bool = False,
    environment: RunEnvironmentSnapshot | None = None,
) -> dict[str, MetricProvenance]:
    """Build request-level metric provenance for observed, estimated, and null metrics."""
    missing = set(missing_metrics)
    provenance: dict[str, MetricProvenance] = {
        "input_tokens": MetricProvenance(
            source_type="estimated",
            measurement_method=response.token_count_method,
            unit="tokens",
            notes="Estimated from the prompt when provider prompt token usage is unavailable.",
        ),
        "output_tokens": MetricProvenance(
            source_type="estimated",
            measurement_method=response.token_count_method,
            unit="tokens",
            notes="Estimated from visible response text; reasoning-only output may be zero.",
        ),
        "provider_completion_tokens": MetricProvenance(
            source_type="provider_reported",
            measurement_method="OpenAI-compatible usage.completion_tokens",
            unit="tokens",
            provider_field="usage.completion_tokens",
            available=response.provider_completion_tokens is not None,
            missing_reason=_missing_provider_usage_reason()
            if response.provider_completion_tokens is None
            else None,
        ),
        "finish_reason": MetricProvenance(
            source_type="provider_reported",
            measurement_method="OpenAI-compatible choices.finish_reason",
            provider_field="choices[].finish_reason",
            available=response.finish_reason is not None,
            missing_reason="Provider response did not include finish_reason."
            if response.finish_reason is None
            else None,
        ),
        "reasoning_tokens": MetricProvenance(
            source_type="provider_reported"
            if _has_provider_reasoning_usage(response)
            else "estimated",
            measurement_method="provider usage reasoning_tokens or char approximation",
            unit="tokens",
            provider_field="usage.*_tokens_details.reasoning_tokens",
            available=response.reasoning_tokens is not None,
            missing_reason="No reasoning content or provider reasoning token usage was available."
            if response.reasoning_tokens is None
            else None,
        ),
        "ttft_ms": MetricProvenance(
            source_type="client_observed",
            measurement_method="time_to_first_visible_stream_chunk_or_mock_metadata",
            unit="ms",
            available=response.ttft_ms is not None,
            missing_reason="TTFT requires streaming output or mock timing metadata."
            if response.ttft_ms is None
            else None,
        ),
        "first_reasoning_token_ms": MetricProvenance(
            source_type="client_observed",
            measurement_method="time_to_first_reasoning_stream_chunk",
            unit="ms",
            available=response.first_reasoning_token_ms is not None,
            missing_reason="No reasoning token timing was observed."
            if response.first_reasoning_token_ms is None
            else None,
        ),
        "tpot_ms": MetricProvenance(
            source_type="client_observed",
            measurement_method="mean_inter_token_latency_or_mock_decode_delay",
            unit="ms/token",
            available=response.tpot_ms is not None,
            missing_reason="TPOT requires at least two streamed tokens or mock timing metadata."
            if response.tpot_ms is None
            else None,
        ),
        "itl_ms": MetricProvenance(
            source_type="client_observed",
            measurement_method="average_inter_token_latency",
            unit="ms/token",
            available=response.itl_ms is not None,
            missing_reason="ITL requires at least two streamed token timestamps."
            if response.itl_ms is None
            else None,
        ),
        "e2e_latency_ms": MetricProvenance(
            source_type="client_observed",
            measurement_method="client_wall_clock_request_latency",
            unit="ms",
            available=response.e2e_latency_ms is not None,
            missing_reason="Client request did not complete with latency timing."
            if response.e2e_latency_ms is None
            else None,
        ),
        "input_tokens_per_second": MetricProvenance(
            source_type="derived",
            measurement_method="input_tokens / e2e_latency_seconds",
            unit="tokens/second",
            available=input_tps_available,
            missing_reason="Requires input_tokens and e2e_latency_ms."
            if not input_tps_available
            else None,
        ),
        "output_tokens_per_second": MetricProvenance(
            source_type="derived",
            measurement_method="output_tokens / e2e_latency_seconds",
            unit="tokens/second",
            available=output_tps_available,
            missing_reason="Requires output_tokens and e2e_latency_ms."
            if not output_tps_available
            else None,
        ),
        "requests_per_second": MetricProvenance(
            source_type="derived",
            measurement_method="completed_requests / experiment_elapsed_seconds",
            unit="requests/second",
            available=requests_per_second_available,
            missing_reason="Computed after all scheduled requests finish."
            if not requests_per_second_available
            else None,
        ),
        "cache_hit_rate": MetricProvenance(
            source_type="engine_reported",
            measurement_method="endpoint or mock response metadata",
            unit="ratio",
            available=metadata.get("cache_hit_rate") is not None,
            missing_reason=UNAVAILABLE_ENGINE_METRIC_REASON
            if "cache_hit_rate" in missing
            else None,
        ),
        "cache_hit_proxy": MetricProvenance(
            source_type="derived",
            measurement_method="cache_hit_rate passthrough when engine metric is available",
            unit="ratio",
            available=metadata.get("cache_hit_rate") is not None,
            missing_reason="Cache hit proxy requires cache_hit_rate or a comparison-derived proxy."
            if metadata.get("cache_hit_rate") is None
            else None,
        ),
        "cache_miss_penalty_ms": MetricProvenance(
            source_type="engine_reported",
            measurement_method="endpoint or mock response metadata",
            unit="ms",
            available=metadata.get("cache_miss_penalty_ms") is not None,
            missing_reason=UNAVAILABLE_ENGINE_METRIC_REASON
            if "cache_miss_penalty_ms" in missing
            else None,
        ),
        "engine_version": MetricProvenance(
            source_type="engine_reported",
            measurement_method="experiment config environment metadata",
            available=environment is not None and environment.engine_version is not None,
            missing_reason=UNAVAILABLE_ENGINE_METRIC_REASON
            if "engine_version" in missing
            else None,
        ),
        "gpu_memory_used_gb": MetricProvenance(
            source_type="gpu_reported",
            measurement_method="GPU telemetry adapter",
            unit="GB",
            available=False,
            missing_reason=UNAVAILABLE_GPU_METRIC_REASON,
        ),
        "gpu_memory_peak_gb": MetricProvenance(
            source_type="gpu_reported",
            measurement_method="GPU telemetry adapter",
            unit="GB",
            available=False,
            missing_reason=UNAVAILABLE_GPU_METRIC_REASON,
        ),
        "gpu_type": MetricProvenance(
            source_type="gpu_reported",
            measurement_method="experiment config environment metadata or GPU telemetry adapter",
            available=environment is not None and environment.gpu_type is not None,
            missing_reason=UNAVAILABLE_GPU_METRIC_REASON if "gpu_type" in missing else None,
        ),
        "gpu_count": MetricProvenance(
            source_type="gpu_reported",
            measurement_method="experiment config environment metadata or GPU telemetry adapter",
            available=environment is not None and environment.gpu_count is not None,
            missing_reason=UNAVAILABLE_GPU_METRIC_REASON if "gpu_count" in missing else None,
        ),
        "quality_score": MetricProvenance(
            source_type="derived",
            measurement_method="configured KVOptBench evaluator",
            available=response.success,
            missing_reason="Quality is not evaluated for failed requests."
            if not response.success
            else None,
        ),
    }
    return provenance


def healthcheck_failure_provenance(
    environment: RunEnvironmentSnapshot | None = None,
) -> dict[str, MetricProvenance]:
    """Metric provenance for synthetic failure rows when endpoint healthcheck fails."""
    unavailable = {
        "ttft_ms": ("client_observed", "time_to_first_visible_stream_chunk_or_mock_metadata", "ms"),
        "tpot_ms": ("client_observed", "mean_inter_token_latency_or_mock_decode_delay", "ms/token"),
        "itl_ms": ("client_observed", "average_inter_token_latency", "ms/token"),
        "e2e_latency_ms": ("client_observed", "client_wall_clock_request_latency", "ms"),
        "requests_per_second": ("derived", "completed_requests / experiment_elapsed_seconds", "requests/second"),
        "cache_hit_rate": ("engine_reported", "endpoint or mock response metadata", "ratio"),
        "cache_miss_penalty_ms": ("engine_reported", "endpoint or mock response metadata", "ms"),
        "gpu_memory_used_gb": ("gpu_reported", "GPU telemetry adapter", "GB"),
        "gpu_memory_peak_gb": ("gpu_reported", "GPU telemetry adapter", "GB"),
        "engine_version": ("engine_reported", "experiment config environment metadata", None),
        "gpu_type": (
            "gpu_reported",
            "experiment config environment metadata or GPU telemetry adapter",
            None,
        ),
        "gpu_count": (
            "gpu_reported",
            "experiment config environment metadata or GPU telemetry adapter",
            None,
        ),
    }
    provenance = {
        metric: MetricProvenance(
            source_type=source_type,  # type: ignore[arg-type]
            measurement_method=method,
            unit=unit,
            available=False,
            missing_reason="Endpoint health check failed before requests were sent.",
        )
        for metric, (source_type, method, unit) in unavailable.items()
    }
    if environment is not None:
        if environment.engine_version is not None:
            provenance["engine_version"] = MetricProvenance(
                source_type="engine_reported",
                measurement_method="experiment config environment metadata",
                available=True,
            )
        if environment.gpu_type is not None:
            provenance["gpu_type"] = MetricProvenance(
                source_type="gpu_reported",
                measurement_method="experiment config environment metadata or GPU telemetry adapter",
                available=True,
            )
        if environment.gpu_count is not None:
            provenance["gpu_count"] = MetricProvenance(
                source_type="gpu_reported",
                measurement_method="experiment config environment metadata or GPU telemetry adapter",
                available=True,
            )
    return provenance


def mark_requests_per_second_available(
    provenance: dict[str, MetricProvenance],
) -> dict[str, MetricProvenance]:
    """Mark RPS available after the experiment-level value is computed."""
    updated = dict(provenance)
    updated["requests_per_second"] = MetricProvenance(
        source_type="derived",
        measurement_method="completed_requests / experiment_elapsed_seconds",
        unit="requests/second",
        available=True,
    )
    return updated


def _has_provider_reasoning_usage(response: TimedResponse) -> bool:
    return response.reasoning_tokens is not None and response.reasoning_content_present is False


def _missing_provider_usage_reason() -> str:
    return "Provider response did not include usage completion token counts."
