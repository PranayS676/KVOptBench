import socket
import threading
from pathlib import Path

import uvicorn

from kvoptbench.analysis.speculative_decoding import compare_speculative_decoding_results
from kvoptbench.analysis.summarize import summarize_results
from kvoptbench.experiments.speculative_decoding import (
    run_speculative_decoding_plan,
    write_speculative_decoding_plan_configs,
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


def test_speculative_decoding_plan_runs_against_mock_server_and_reports(tmp_path: Path) -> None:
    port = _free_port()
    server = _start_test_server(port)
    plan_dir = tmp_path / "plan"
    raw_dir = tmp_path / "raw"
    workload = tmp_path / "decode_heavy.jsonl"

    try:
        generate_to_file(
            profile="decode_heavy",
            out=workload,
            count=2,
            target_input_tokens=128,
            target_output_tokens=64,
        )
        write_speculative_decoding_plan_configs(
            plan_dir=plan_dir,
            experiment_prefix="spec_decode",
            provider="mock",
            engine="vllm",
            model_id="mock-frontier-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            workload_file=workload,
            output_dir=raw_dir,
            max_output_tokens=16,
        )

        outputs = run_speculative_decoding_plan(plan_dir)

        assert len(outputs) == 2
        summary = summarize_results(input_path=raw_dir, output_path=tmp_path / "summary.csv")
        speculative_decoding = compare_speculative_decoding_results(
            input_path=raw_dir,
            output_path=tmp_path / "speculative_decoding.csv",
        )
        report = generate_report(
            input_path=summary,
            output_path=tmp_path / "report.md",
            spec_decoding_input_path=speculative_decoding,
        )

        report_text = report.read_text(encoding="utf-8")
        assert "## Speculative Decoding" in report_text
        assert "speculative_decoding" in "\n".join(
            output.read_text(encoding="utf-8") for output in outputs
        )
        assert "speculative_decoding_interpretation" in speculative_decoding.read_text(
            encoding="utf-8"
        )
    finally:
        server.should_exit = True
