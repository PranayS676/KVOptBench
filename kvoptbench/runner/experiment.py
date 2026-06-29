"""Config-driven experiment runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from kvoptbench.client.openai_compat import OpenAICompatClient
from kvoptbench.config import load_config
from kvoptbench.evals.dispatch import evaluate_output
from kvoptbench.schemas import EndpointHealth, RequestResult, WorkloadItem


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
    endpoint_health = await client.healthcheck()
    if not endpoint_health.ok:
        results = [
            _failed_healthcheck_result(
                run_id=run_id,
                item=item,
                config=config,
                endpoint_health=endpoint_health,
            )
            for item in items
        ]
        _write_results(config.output_file, results, requests_per_second=None)
        return config.output_file

    semaphore = asyncio.Semaphore(config.concurrency)
    started = time.perf_counter()

    async def run_one(index: int, item: WorkloadItem) -> RequestResult:
        if config.request_rate:
            await asyncio.sleep(index / config.request_rate)
        async with semaphore:
            response = await client.chat(item)
        quality = evaluate_output(response.content, item) if response.success else None
        e2e_seconds = (response.e2e_latency_ms or 0) / 1000
        output_tps = response.output_tokens / e2e_seconds if e2e_seconds > 0 else None
        input_tps = response.input_tokens / e2e_seconds if e2e_seconds > 0 else None
        metadata = response.response_metadata
        missing_metrics = _missing_metrics(response, metadata)
        return RequestResult(
            run_id=run_id,
            experiment_id=config.experiment_id,
            official_run=config.official_run,
            provider=config.provider,
            engine=config.engine,
            model_id=config.model_id,
            strategy=config.strategy,
            workload=item.workload,
            task_id=item.task_id,
            concurrency=config.concurrency,
            request_rate=config.request_rate,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
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
            metadata={
                "config_metadata": config.metadata,
                "workload_metadata": item.metadata,
                "endpoint_health": endpoint_health.model_dump(),
                "quality_details": quality.details if quality else {},
            },
        )

    results = await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
    elapsed = time.perf_counter() - started
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
            handle.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")


def _failed_healthcheck_result(
    *,
    run_id: str,
    item: WorkloadItem,
    config,
    endpoint_health: EndpointHealth,
) -> RequestResult:
    return RequestResult(
        run_id=run_id,
        experiment_id=config.experiment_id,
        official_run=config.official_run,
        provider=config.provider,
        engine=config.engine,
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
        missing_metrics=[
            "ttft_ms",
            "tpot_ms",
            "itl_ms",
            "e2e_latency_ms",
            "engine_version",
            "gpu_memory_used_gb",
            "gpu_memory_peak_gb",
            "gpu_type",
            "gpu_count",
            "cache_hit_rate",
            "cache_miss_penalty_ms",
        ],
        metadata={
            "config_metadata": config.metadata,
            "workload_metadata": item.metadata,
            "endpoint_health": endpoint_health.model_dump(),
            "quality_details": {},
        },
    )


def _missing_metrics(response, metadata: dict) -> list[str]:
    missing = []
    if response.ttft_ms is None:
        missing.append("ttft_ms")
    if response.tpot_ms is None:
        missing.append("tpot_ms")
    for metric in [
        "engine_version",
        "gpu_memory_used_gb",
        "gpu_memory_peak_gb",
        "gpu_type",
        "gpu_count",
    ]:
        missing.append(metric)
    if metadata.get("cache_hit_rate") is None:
        missing.append("cache_hit_rate")
    if metadata.get("cache_miss_penalty_ms") is None:
        missing.append("cache_miss_penalty_ms")
    return sorted(set(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a KVOptBench experiment.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    output = asyncio.run(run_experiment(args.config))
    print(f"Wrote results to {output}")


if __name__ == "__main__":
    main()
