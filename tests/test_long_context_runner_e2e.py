import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.analysis.long_context import compare_long_context_results
from kvoptbench.analysis.summarize import summarize_results
from kvoptbench.experiments.long_context import (
    run_long_context_plan,
    write_long_context_plan_configs,
)
from kvoptbench.mock_server.main import create_app
from kvoptbench.reports.generate import generate_report
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


def test_long_context_plan_runs_against_mock_server_and_reports(tmp_path: Path) -> None:
    port = _free_port()
    server = _start_test_server(port)
    plan_dir = tmp_path / "plan"
    raw_dir = tmp_path / "raw"
    workload = tmp_path / "long_context_pressure.jsonl"

    try:
        generate_to_file(
            profile="long_context_pressure",
            out=workload,
            count=2,
            target_input_tokens=512,
            target_output_tokens=32,
            context_buckets=(128, 256),
        )
        write_long_context_plan_configs(
            plan_dir=plan_dir,
            experiment_prefix="long_context",
            provider="mock",
            engine="vllm",
            model_id="mock-frontier-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            workload_file=workload,
            output_dir=raw_dir,
            strategies=("baseline",),
            max_output_tokens=16,
        )

        outputs = run_long_context_plan(plan_dir)

        assert len(outputs) == 1
        summary = summarize_results(input_path=raw_dir, output_path=tmp_path / "summary.csv")
        long_context = compare_long_context_results(
            input_path=raw_dir,
            output_path=tmp_path / "long_context.csv",
        )
        report = generate_report(
            input_path=summary,
            output_path=tmp_path / "report.md",
            long_context_input_path=long_context,
        )

        report_text = report.read_text(encoding="utf-8")
        assert "## Long Context Pressure" in report_text
        assert "long_context_pressure" in outputs[0].read_text(encoding="utf-8")
        assert "context_token_bucket" in long_context.read_text(encoding="utf-8")
    finally:
        server.should_exit = True
