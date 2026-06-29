import asyncio
import json
from pathlib import Path

from kvoptbench.runner.experiment import run_experiment
from kvoptbench.schemas import RequestResult
from kvoptbench.workloads.generate import generate_to_file


def test_runner_writes_failed_row_when_healthcheck_fails(tmp_path: Path) -> None:
    workload_file = tmp_path / "workload.jsonl"
    output_file = tmp_path / "results.jsonl"
    config_file = tmp_path / "config.yaml"
    generate_to_file(
        profile="shared_prefix",
        out=workload_file,
        count=2,
        target_input_tokens=128,
        target_output_tokens=16,
    )
    config_file.write_text(
        "\n".join(
            [
                "experiment_id: failed_healthcheck",
                "official_run: false",
                "provider: local",
                "engine: vllm",
                "endpoint_type: vllm",
                "model_id: model",
                "strategy: baseline",
                "base_url: http://127.0.0.1:9/v1",
                "healthcheck_path: /v1/models",
                f"workload_file: {workload_file.as_posix()}",
                f"output_file: {output_file.as_posix()}",
                "concurrency: 1",
                "max_tasks: 2",
                "request_timeout_seconds: 0.2",
                "retries: 0",
            ]
        ),
        encoding="utf-8",
    )

    asyncio.run(run_experiment(config_file))

    rows = [RequestResult.model_validate(json.loads(line)) for line in output_file.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row.success is False for row in rows)
    assert all(row.error_type == "EndpointHealthcheckFailed" for row in rows)
    assert all(row.metadata["endpoint_health"]["ok"] is False for row in rows)


def test_public_real_endpoint_example_configs_validate() -> None:
    from kvoptbench.config import load_config

    vllm = load_config("examples/vllm_openai_compatible_config.yaml")
    sglang = load_config("examples/sglang_openai_compatible_config.yaml")

    assert vllm.endpoint_type == "vllm"
    assert sglang.endpoint_type == "sglang"
    assert vllm.retries >= 1
    assert sglang.retries >= 1
