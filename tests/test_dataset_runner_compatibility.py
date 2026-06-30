import asyncio
import json
import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.datasets.manifest import DatasetPrepareOptions
from kvoptbench.datasets.qasper import QasperAdapter
from kvoptbench.mock_server.main import create_app
from kvoptbench.runner.experiment import run_experiment


def test_dataset_workload_runs_against_mock_server_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    port = _free_port()
    server = _start_test_server(port)
    workload_path = tmp_path / "qasper_shared.jsonl"
    manifest_path = tmp_path / "qasper_manifest.json"
    output_path = tmp_path / "results.jsonl"
    config_path = tmp_path / "config.yaml"

    try:
        QasperAdapter().prepare(
            DatasetPrepareOptions(
                source="qasper",
                mode="shared_prefix",
                split="validation",
                source_path=Path("tests/fixtures/datasets/qasper_tiny.json"),
                out=workload_path,
                manifest=manifest_path,
                max_items=2,
                target_input_tokens=512,
                target_output_tokens=32,
            )
        )
        config_path.write_text(
            "\n".join(
                [
                    "experiment_id: dataset_mock_smoke",
                    "provider: mock",
                    "engine: mock",
                    "model_id: mock-frontier-model",
                    "strategy: cache_on",
                    f"base_url: http://127.0.0.1:{port}/v1",
                    "endpoint_type: mock",
                    f"workload_file: {workload_path}",
                    f"output_file: {output_path}",
                    "max_tasks: 2",
                    "max_output_tokens: 32",
                    "concurrency: 1",
                    "stream: true",
                ]
            ),
            encoding="utf-8",
        )

        asyncio.run(run_experiment(config_path))

        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 2
        assert all(row["success"] for row in rows)
        workload_metadata = rows[0]["metadata"]["workload_metadata"]
        assert workload_metadata["dataset"] == "qasper"
        assert workload_metadata["prefix_hash"]
        assert workload_metadata["prompt_hash"]
        assert workload_metadata["measured_input_tokens"] > 0
    finally:
        server.should_exit = True


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
