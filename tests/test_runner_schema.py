import asyncio
import json
import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.mock_server.main import create_app
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import (
    EndpointHealth,
    MetricProvenance,
    RequestResult,
    RunEnvironmentSnapshot,
    TimedResponse,
    WorkloadItem,
)
from kvoptbench.workloads.generate import generate_to_file


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_test_server(port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        pass
    return server


def test_runner_writes_valid_result_rows(tmp_path: Path) -> None:
    port = _free_port()
    server = _start_test_server(port)
    workload_file = tmp_path / "workload.jsonl"
    output_file = tmp_path / "results.jsonl"
    config_file = tmp_path / "config.yaml"

    try:
        generate_to_file(
            profile="partial_prefix",
            out=workload_file,
            count=2,
            target_input_tokens=256,
            target_output_tokens=32,
        )
        config_file.write_text(
            "\n".join(
                [
                    "experiment_id: runner_test",
                    "official_run: false",
                    "provider: mock",
                    "engine: mock",
                    "model_id: mock-frontier-model",
                    "strategy: baseline",
                    f"base_url: http://127.0.0.1:{port}/v1",
                    f"workload_file: {workload_file.as_posix()}",
                    f"output_file: {output_file.as_posix()}",
                    "concurrency: 2",
                    "max_tasks: 2",
                    "max_output_tokens: 16",
                    "timeout_seconds: 10",
                    "stream: true",
                ]
            ),
            encoding="utf-8",
        )

        asyncio.run(run_experiment(config_file))

        rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        for row in rows:
            result = RequestResult.model_validate(row)
            assert result.experiment_id == "runner_test"
            assert result.success is True
            assert result.ttft_ms is not None
            assert result.e2e_latency_ms is not None
            assert result.output_tokens > 0
            assert result.environment is not None
            assert result.metric_provenance["ttft_ms"].source_type == "client_observed"
            assert result.metric_provenance["input_tokens"].source_type == "estimated"
            assert result.metric_provenance["cache_hit_rate"].source_type == "engine_reported"
            assert "shared_prefix_ratio" in result.metadata["workload_metadata"]
    finally:
        server.should_exit = True


def test_runner_writes_reasoning_compatibility_fields(monkeypatch, tmp_path: Path) -> None:
    workload_file = tmp_path / "workload.jsonl"
    output_file = tmp_path / "results.jsonl"
    config_file = tmp_path / "config.yaml"
    item = WorkloadItem(
        task_id="reasoning-1",
        workload="reasoning_smoke",
        category="reasoning",
        prompt="Think, then answer EXPECTED_ANSWER: endpoint-ok",
        expected_answer="endpoint-ok",
        target_input_tokens=64,
        target_output_tokens=16,
    )
    workload_file.write_text(json.dumps(item.model_dump()) + "\n", encoding="utf-8")
    config_file.write_text(
        "\n".join(
            [
                "experiment_id: reasoning_runner_test",
                "official_run: false",
                "provider: local",
                "engine: openai_compatible",
                "endpoint_type: openai_compatible",
                "model_id: reasoning-model",
                "strategy: baseline",
                "base_url: http://testserver/v1",
                f"workload_file: {workload_file.as_posix()}",
                f"output_file: {output_file.as_posix()}",
                "concurrency: 1",
                "max_tasks: 1",
                "stream: true",
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def healthcheck(self) -> EndpointHealth:
            return EndpointHealth(ok=True, url="http://testserver/v1/models")

        async def chat(self, item: WorkloadItem) -> TimedResponse:
            return TimedResponse(
                content="",
                input_tokens=12,
                output_tokens=0,
                provider_completion_tokens=9,
                reasoning_content_present=True,
                reasoning_tokens=9,
                first_reasoning_token_ms=42.0,
                visible_answer_missing=True,
                finish_reason="stop",
                e2e_latency_ms=100.0,
                success=True,
            )

    monkeypatch.setattr("kvoptbench.runner.experiment.OpenAICompatClient", FakeClient)

    asyncio.run(run_experiment(config_file))

    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    result = RequestResult.model_validate(rows[0])
    assert result.reasoning_content_present is True
    assert result.reasoning_tokens == 9
    assert result.first_reasoning_token_ms == 42.0
    assert result.visible_answer_missing is True
    assert result.provider_completion_tokens == 9
    assert result.metric_provenance["provider_completion_tokens"].source_type == "provider_reported"
    assert result.metric_provenance["reasoning_tokens"].source_type == "estimated"
    assert result.metric_provenance["ttft_ms"].available is False
    assert result.metadata["response_metadata"]["finish_reason"] == "stop"


def test_runner_records_configured_environment_metadata(monkeypatch, tmp_path: Path) -> None:
    workload_file = tmp_path / "workload.jsonl"
    output_file = tmp_path / "results.jsonl"
    config_file = tmp_path / "config.yaml"
    item = WorkloadItem(
        task_id="env-1",
        workload="decode_heavy",
        category="decode",
        prompt="Return endpoint-ok",
        expected_answer="endpoint-ok",
        target_input_tokens=32,
        target_output_tokens=16,
    )
    workload_file.write_text(json.dumps(item.model_dump()) + "\n", encoding="utf-8")
    config_file.write_text(
        "\n".join(
            [
                "experiment_id: env_runner_test",
                "official_run: false",
                "provider: local",
                "engine: vllm",
                "endpoint_type: vllm",
                "model_id: example/model",
                "strategy: baseline",
                "base_url: http://testserver/v1",
                f"workload_file: {workload_file.as_posix()}",
                f"output_file: {output_file.as_posix()}",
                "concurrency: 1",
                "max_tasks: 1",
                "engine_version: 0.8.0",
                "model_revision: rev-1",
                "cuda_version: '12.4'",
                "gpu_type: NVIDIA L40S",
                "gpu_count: 1",
                "backend_launch_command: vllm serve example/model --api-key secret",
            ]
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def healthcheck(self) -> EndpointHealth:
            return EndpointHealth(ok=True, url="http://testserver/v1/models")

        async def chat(self, item: WorkloadItem) -> TimedResponse:
            return TimedResponse(
                content="endpoint-ok",
                input_tokens=8,
                output_tokens=2,
                provider_completion_tokens=2,
                ttft_ms=10.0,
                tpot_ms=5.0,
                itl_ms=5.0,
                e2e_latency_ms=20.0,
                success=True,
                response_metadata={"cache_hit_rate": 0.5, "cache_miss_penalty_ms": 12.0},
            )

    monkeypatch.setattr("kvoptbench.runner.experiment.OpenAICompatClient", FakeClient)

    asyncio.run(run_experiment(config_file))

    row = json.loads(output_file.read_text(encoding="utf-8").strip())
    result = RequestResult.model_validate(row)
    assert result.engine_version == "0.8.0"
    assert result.gpu_type == "NVIDIA L40S"
    assert result.gpu_count == 1
    assert result.environment is not None
    assert result.environment.model_revision == "rev-1"
    assert result.environment.cuda_version == "12.4"
    assert result.environment.backend_launch_command.endswith("--api-key <redacted>")
    assert result.environment.config_sha256 is not None
    assert result.environment.workload_sha256 is not None
    assert "engine_version" not in result.missing_metrics
    assert "gpu_type" not in result.missing_metrics
    assert "gpu_count" not in result.missing_metrics
    assert result.metric_provenance["engine_version"].available is True
    assert result.metric_provenance["gpu_type"].available is True


def test_request_result_accepts_metric_provenance_and_environment_snapshot() -> None:
    result = RequestResult(
        run_id="run",
        experiment_id="exp",
        provider="mock",
        engine="mock",
        model_id="model",
        strategy="baseline",
        workload="shared_prefix",
        task_id="task",
        concurrency=1,
        metric_provenance={
            "ttft_ms": MetricProvenance(
                source_type="client_observed",
                measurement_method="time_to_first_stream_chunk",
                unit="ms",
            )
        },
        environment=RunEnvironmentSnapshot(
            python_version="3.11.0",
            platform="Windows",
            kvoptbench_version="0.1.0",
        ),
    )

    payload = result.model_dump(mode="json")

    assert payload["metric_provenance"]["ttft_ms"]["source_type"] == "client_observed"
    assert payload["environment"]["platform"] == "Windows"

