import asyncio
import json
import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.mock_server.main import create_app
from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import RequestResult
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
            profile="shared_prefix",
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
    finally:
        server.should_exit = True

