"""Config-driven experiment runner."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from kvoptbench.client.openai_compat import OpenAICompatClient
from kvoptbench.config import load_config
from kvoptbench.evals.dispatch import evaluate_output
from kvoptbench.runner.environment import capture_environment
from kvoptbench.runner.provenance import (
    build_metric_provenance,
    healthcheck_failure_provenance,
    mark_requests_per_second_available,
)
from kvoptbench.schemas import EndpointHealth, MetricProvenance, RequestResult, WorkloadItem
from kvoptbench.telemetry.runtime import TelemetryRunSummary, build_telemetry_collector


def load_workload(path: str | Path) -> list[WorkloadItem]:
    workload_path = Path(path)
    items: list[WorkloadItem] = []
    with workload_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(WorkloadItem.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid workload row {line_no} in {workload_path}: {exc}") from exc
    return items


async def run_experiment(config_path: str | Path) -> Path:
    config = load_config(config_path)
    items = load_workload(config.workload_file)
    if config.max_tasks:
        items = items[: config.max_tasks]

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}-{config.experiment_id}"
    client = OpenAICompatClient(config)
    environment_metadata = _environment_metadata(config, config_path)
    environment = capture_environment(Path.cwd(), metadata=environment_metadata)
    telemetry = build_telemetry_collector(config, run_id=run_id)
    await telemetry.start()
    endpoint_health = await client.healthcheck()
    if not endpoint_health.ok:
        telemetry_summary = await telemetry.stop()
        results = [
            _failed_healthcheck_result(
                run_id=run_id,
                item=item,
                config=config,
                endpoint_health=endpoint_health,
                environment=environment,
                environment_metadata=environment_metadata,
            )
            for item in items
        ]
        _apply_telemetry_to_results(results, telemetry_summary)
        _write_results(config.output_file, results, requests_per_second=None)
        return config.output_file

    semaphore = asyncio.Semaphore(config.concurrency)
    started = time.perf_counter()

    async def run_one(index: int, item: WorkloadItem) -> RequestResult:
        if config.request_rate:
            await asyncio.sleep(index / config.request_rate)
        async with semaphore:
            response = await client.chat(item)
        quality = (
            evaluate_output(response.content, item, tool_calls=response.tool_calls)
            if response.success
            else None
        )
        e2e_seconds = (response.e2e_latency_ms or 0) / 1000
        output_tps = response.output_tokens / e2e_seconds if e2e_seconds > 0 else None
        input_tps = response.input_tokens / e2e_seconds if e2e_seconds > 0 else None
        metadata = response.response_metadata
        missing_metrics = _missing_metrics(response, metadata, environment)
        metric_provenance = build_metric_provenance(
            response=response,
            metadata=metadata,
            missing_metrics=missing_metrics,
            output_tps_available=output_tps is not None,
            input_tps_available=input_tps is not None,
            environment=environment,
        )
        return RequestResult(
            run_id=run_id,
            experiment_id=config.experiment_id,
            official_run=config.official_run,
            provider=config.provider,
            gpu_type=environment.gpu_type,
            gpu_count=environment.gpu_count,
            engine=config.engine,
            engine_version=environment.engine_version,
            model_id=config.model_id,
            strategy=config.strategy,
            workload=item.workload,
            task_id=item.task_id,
            concurrency=config.concurrency,
            request_rate=config.request_rate,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            provider_completion_tokens=response.provider_completion_tokens,
            reasoning_content_present=response.reasoning_content_present,
            reasoning_tokens=response.reasoning_tokens,
            first_reasoning_token_ms=response.first_reasoning_token_ms,
            visible_answer_missing=response.visible_answer_missing,
            finish_reason=response.finish_reason,
            tool_call_count=len(response.tool_calls),
            tool_call_names=[call.name for call in response.tool_calls if call.name],
            tool_calls=response.tool_calls,
            target_input_tokens=item.target_input_tokens,
            target_output_tokens=item.target_output_tokens,
            shared_prefix_tokens=item.shared_prefix_tokens,
            cache_state=metadata.get("cache_state") or "na",
            cache_hit_rate=metadata.get("cache_hit_rate"),
            cache_hit_proxy=metadata.get("cache_hit_rate"),
            cache_miss_penalty_ms=metadata.get("cache_miss_penalty_ms"),
            ttft_ms=response.ttft_ms,
            tpot_ms=response.tpot_ms,
            itl_ms=response.itl_ms,
            e2e_latency_ms=response.e2e_latency_ms,
            input_tokens_per_second=round(input_tps, 3) if input_tps is not None else None,
            output_tokens_per_second=round(output_tps, 3) if output_tps is not None else None,
            success=response.success,
            error_type=response.error_type,
            error_message=response.error_message,
            quality_score=quality.quality_score if quality else None,
            quality_method=quality.quality_method if quality else None,
            token_count_method=response.token_count_method,
            missing_metrics=missing_metrics,
            metric_provenance=metric_provenance,
            environment=environment,
            metadata={
                "config_metadata": config.metadata,
                "workload_metadata": item.metadata,
                "environment_metadata": _public_environment_metadata(environment_metadata),
                "endpoint_health": endpoint_health.model_dump(),
                "quality_details": quality.details if quality else {},
                "response_metadata": {
                    **metadata,
                    "finish_reason": response.finish_reason,
                    "reasoning_content_captured": response.reasoning_content is not None,
                },
            },
        )

    results = await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
    elapsed = time.perf_counter() - started
    telemetry_summary = await telemetry.stop()
    _apply_telemetry_to_results(results, telemetry_summary)
    rps = len(results) / elapsed if elapsed > 0 else None
    _write_results(config.output_file, results, requests_per_second=rps)
    return config.output_file


def _write_results(
    output_file: Path, results: list[RequestResult], requests_per_second: float | None
) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        for result in results:
            if requests_per_second is not None:
                result.requests_per_second = round(requests_per_second, 3)
                result.metric_provenance = mark_requests_per_second_available(
                    result.metric_provenance
                )
            handle.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")


def _failed_healthcheck_result(
    *,
    run_id: str,
    item: WorkloadItem,
    config,
    endpoint_health: EndpointHealth,
    environment,
    environment_metadata: dict[str, Any],
) -> RequestResult:
    missing_metrics = _failure_missing_metrics(environment)
    return RequestResult(
        run_id=run_id,
        experiment_id=config.experiment_id,
        official_run=config.official_run,
        provider=config.provider,
        gpu_type=environment.gpu_type,
        gpu_count=environment.gpu_count,
        engine=config.engine,
        engine_version=environment.engine_version,
        model_id=config.model_id,
        strategy=config.strategy,
        workload=item.workload,
        task_id=item.task_id,
        concurrency=config.concurrency,
        request_rate=config.request_rate,
        target_input_tokens=item.target_input_tokens,
        target_output_tokens=item.target_output_tokens,
        shared_prefix_tokens=item.shared_prefix_tokens,
        success=False,
        error_type="EndpointHealthcheckFailed",
        error_message=endpoint_health.error_message or "endpoint health check failed",
        missing_metrics=missing_metrics,
        metric_provenance=healthcheck_failure_provenance(environment),
        environment=environment,
        metadata={
            "config_metadata": config.metadata,
            "workload_metadata": item.metadata,
            "environment_metadata": _public_environment_metadata(environment_metadata),
            "endpoint_health": endpoint_health.model_dump(),
            "quality_details": {},
        },
    )


def _missing_metrics(response, metadata: dict, environment=None) -> list[str]:
    missing = []
    if response.ttft_ms is None:
        missing.append("ttft_ms")
    if response.tpot_ms is None:
        missing.append("tpot_ms")
    if response.reasoning_content_present and response.first_reasoning_token_ms is None:
        missing.append("first_reasoning_token_ms")
    if environment is None or environment.engine_version is None:
        missing.append("engine_version")
    if environment is None or environment.gpu_type is None:
        missing.append("gpu_type")
    if environment is None or environment.gpu_count is None:
        missing.append("gpu_count")
    for metric in ["gpu_memory_used_gb", "gpu_memory_peak_gb"]:
        missing.append(metric)
    if metadata.get("cache_hit_rate") is None:
        missing.append("cache_hit_rate")
    if metadata.get("cache_miss_penalty_ms") is None:
        missing.append("cache_miss_penalty_ms")
    return sorted(set(missing))


def _failure_missing_metrics(environment) -> list[str]:
    missing = [
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "e2e_latency_ms",
        "gpu_memory_used_gb",
        "gpu_memory_peak_gb",
        "cache_hit_rate",
        "cache_miss_penalty_ms",
    ]
    if environment is None or environment.engine_version is None:
        missing.append("engine_version")
    if environment is None or environment.gpu_type is None:
        missing.append("gpu_type")
    if environment is None or environment.gpu_count is None:
        missing.append("gpu_count")
    return sorted(set(missing))


def _apply_telemetry_to_results(
    results: list[RequestResult],
    telemetry_summary: TelemetryRunSummary,
) -> None:
    if not telemetry_summary.enabled:
        return
    metrics = telemetry_summary.metrics
    telemetry_missing = {item.metric: item.reason for item in telemetry_summary.missing_metrics}
    for result in results:
        _apply_gpu_telemetry(result, metrics, telemetry_missing, telemetry_summary)
        _apply_cache_telemetry(result, metrics, telemetry_missing, telemetry_summary)
        result.telemetry_run_id = telemetry_summary.run_id
        result.telemetry_summary_path = telemetry_summary.summary_path
        result.telemetry_snapshots_path = telemetry_summary.snapshots_path
        result.metadata["telemetry"] = {
            "run_id": telemetry_summary.run_id,
            "profile": telemetry_summary.telemetry_profile,
            "summary_path": telemetry_summary.summary_path,
            "snapshots_path": telemetry_summary.snapshots_path,
            "snapshot_count": telemetry_summary.snapshot_count,
            "sources": telemetry_summary.sources,
            "missing_metrics": [
                item.model_dump(mode="json") for item in telemetry_summary.missing_metrics
            ],
        }
        result.missing_metrics = sorted(set(result.missing_metrics) - _available_metric_names(metrics))


def _apply_gpu_telemetry(
    result: RequestResult,
    metrics: dict[str, float | None],
    missing: dict[str, str],
    telemetry_summary: TelemetryRunSummary,
) -> None:
    for metric in ["gpu_memory_used_gb", "gpu_memory_peak_gb"]:
        value = metrics.get(metric)
        if value is not None:
            setattr(result, metric, value)
            result.metric_provenance[metric] = MetricProvenance(
                source_type="gpu_reported",
                measurement_method="live GPU telemetry sampler",
                unit="GB",
                available=True,
                provider_field=metric,
                notes=f"Run-level telemetry summary: {telemetry_summary.summary_path}",
            )
        elif metric in missing:
            result.metric_provenance[metric] = MetricProvenance(
                source_type="gpu_reported",
                measurement_method="live GPU telemetry sampler",
                unit="GB",
                available=False,
                missing_reason=missing[metric],
            )


def _apply_cache_telemetry(
    result: RequestResult,
    metrics: dict[str, float | None],
    missing: dict[str, str],
    telemetry_summary: TelemetryRunSummary,
) -> None:
    cache_value = metrics.get("engine_reported_cache_hit_rate")
    cache_field = "engine_reported_cache_hit_rate"
    if cache_value is None:
        cache_value = metrics.get("lmcache_cache_hit_rate")
        cache_field = "lmcache_cache_hit_rate"
    if cache_value is not None and result.cache_hit_rate is None:
        result.cache_hit_rate = cache_value
        result.cache_hit_proxy = cache_value
        result.metric_provenance["cache_hit_rate"] = MetricProvenance(
            source_type="engine_reported",
            measurement_method="run-level telemetry adapter",
            unit="ratio",
            available=True,
            provider_field=cache_field,
            notes=f"Run-level telemetry summary: {telemetry_summary.summary_path}",
        )
        result.metric_provenance["cache_hit_proxy"] = MetricProvenance(
            source_type="derived",
            measurement_method="telemetry cache_hit_rate passthrough",
            unit="ratio",
            available=True,
            provider_field=cache_field,
        )
    elif "cache_hit_rate" in missing:
        result.metric_provenance["cache_hit_rate"] = MetricProvenance(
            source_type="engine_reported",
            measurement_method="run-level telemetry adapter",
            unit="ratio",
            available=False,
            missing_reason=missing["cache_hit_rate"],
        )


def _available_metric_names(metrics: dict[str, float | None]) -> set[str]:
    available = {metric for metric, value in metrics.items() if value is not None}
    if "engine_reported_cache_hit_rate" in available or "lmcache_cache_hit_rate" in available:
        available.update({"cache_hit_rate", "cache_hit_proxy"})
    return available


def _environment_metadata(config, config_path: str | Path | None) -> dict[str, Any]:
    metadata = dict(config.endpoint_metadata or {})
    for field in [
        "engine_version",
        "model_revision",
        "cuda_version",
        "gpu_type",
        "gpu_count",
        "backend_launch_command",
    ]:
        value = getattr(config, field, None)
        if value is not None:
            metadata[field] = value
    if config.config_sha256:
        metadata["config_sha256"] = config.config_sha256
    elif config_path is not None:
        metadata["config_sha256"] = _sha256_file(Path(config_path))
    if config.workload_sha256:
        metadata["workload_sha256"] = config.workload_sha256
    elif config.workload_file.exists():
        metadata["workload_sha256"] = _sha256_file(config.workload_file)
    return metadata


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _public_environment_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "engine_version",
            "model_revision",
            "cuda_version",
            "gpu_type",
            "gpu_count",
            "backend_launch_command",
            "config_sha256",
            "workload_sha256",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a KVOptBench experiment.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    output = asyncio.run(run_experiment(args.config))
    print(f"Wrote results to {output}")


if __name__ == "__main__":
    main()
