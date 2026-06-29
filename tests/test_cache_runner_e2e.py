import json
import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.experiments.cache import run_cache_plan, write_cache_plan_configs
from kvoptbench.mock_server.main import create_app
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


def test_cache_plan_runs_against_mock_server_and_records_cold_warm(tmp_path: Path) -> None:
    port = _free_port()
    server = _start_test_server(port)
    plan_dir = tmp_path / "plan"
    raw_dir = tmp_path / "raw"
    shared_workload = tmp_path / "shared.jsonl"
    random_workload = tmp_path / "random.jsonl"

    try:
        generate_to_file(
            profile="shared_prefix",
            out=shared_workload,
            count=2,
            target_input_tokens=256,
            target_output_tokens=16,
        )
        generate_to_file(
            profile="random_prefix",
            out=random_workload,
            count=2,
            target_input_tokens=256,
            target_output_tokens=16,
        )
        write_cache_plan_configs(
            plan_dir=plan_dir,
            experiment_prefix="mock_cache",
            provider="mock",
            engine="vllm",
            model_id="mock-frontier-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            shared_workload_file=shared_workload,
            random_workload_file=random_workload,
            output_dir=raw_dir,
            strategies=("cache_on",),
        )

        outputs = run_cache_plan(plan_dir)

        assert len(outputs) == 4
        rows = [
            json.loads(line)
            for output in outputs
            for line in output.read_text(encoding="utf-8").splitlines()
        ]
        shared_rows = [
            row
            for row in rows
            if row["metadata"]["config_metadata"]["workload_profile"] == "shared_prefix"
        ]
        assert {row["cache_state"] for row in shared_rows} == {"cold", "warm"}
        assert all(row["metadata"]["config_metadata"]["cache_experiment"] for row in rows)
    finally:
        server.should_exit = True
