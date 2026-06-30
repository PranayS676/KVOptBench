import asyncio
import json
import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.mock_server.main import create_app
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import EndpointHealth, RequestResult, TimedResponse, WorkloadItem
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
    assert result.metadata["response_metadata"]["finish_reason"] == "stop"

