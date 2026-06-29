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

    configs = [
        ("examples/vllm_openai_compatible_config.yaml", "local", "vllm", "vllm"),
        ("examples/sglang_openai_compatible_config.yaml", "local", "sglang", "sglang"),
        ("examples/runpod_vllm_openai_compatible_config.yaml", "runpod", "vllm", "vllm"),
        ("examples/runpod_sglang_openai_compatible_config.yaml", "runpod", "sglang", "sglang"),
        (
            "examples/lambda_cloud_vllm_openai_compatible_config.yaml",
            "lambda_cloud",
            "vllm",
            "vllm",
        ),
        (
            "examples/generic_openai_compatible_config.yaml",
            "other",
            "openai_compatible",
            "openai_compatible",
        ),
    ]

    for path, provider, engine, endpoint_type in configs:
        config = load_config(path)
        assert config.provider == provider
        assert config.engine == engine
        assert config.endpoint_type == endpoint_type
        assert config.retries >= 1
        assert config.base_url.endswith("/v1")
        assert config.output_file.as_posix().startswith("results/raw/")
