from pathlib import Path

import pytest

from kvoptbench.config import ConfigError, load_config


def _write_config(path: Path, extra: str = "") -> Path:
    path.write_text(
        "\n".join(
            [
                "experiment_id: test_exp",
                "official_run: false",
                "provider: mock",
                "engine: mock",
                "model_id: mock-frontier-model",
                "strategy: baseline",
                "base_url: http://127.0.0.1:8000/v1",
                "workload_file: workloads/generated/test.jsonl",
                "output_file: results/raw/test.jsonl",
                "concurrency: 2",
                "max_output_tokens: 16",
                "timeout_seconds: 5",
                "stream: true",
                extra,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_validates_yaml(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "experiment.yaml")

    config = load_config(config_path)

    assert config.experiment_id == "test_exp"
    assert config.provider == "mock"
    assert config.concurrency == 2
    assert config.workload_file == Path("workloads/generated/test.jsonl")
    assert config.capture_reasoning_content is False
    assert config.capture_tool_calls is True


def test_load_config_rejects_missing_required_field(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "experiment.yaml")
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("experiment_id: test_exp\n", ""), encoding="utf-8")

    with pytest.raises(ConfigError, match="experiment_id"):
        load_config(config_path)


def test_load_config_rejects_invalid_concurrency(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "experiment.yaml")
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("concurrency: 2", "concurrency: 0"), encoding="utf-8")

    with pytest.raises(ConfigError, match="concurrency"):
        load_config(config_path)


def test_load_config_supports_real_endpoint_fields(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "real_endpoint.yaml",
        "\n".join(
            [
                "endpoint_type: vllm",
                "healthcheck_path: /v1/models",
                "retries: 2",
                "retry_backoff_seconds: 0.25",
                "request_timeout_seconds: 30",
                "capture_response_headers: true",
                "capture_reasoning_content: true",
                "capture_tool_calls: false",
                "endpoint_metadata:",
                "  deployment: local-vllm",
            ]
        ),
    )

    config = load_config(config_path)

    assert config.endpoint_type == "vllm"
    assert config.healthcheck_path == "/v1/models"
    assert config.retries == 2
    assert config.retry_backoff_seconds == 0.25
    assert config.request_timeout_seconds == 30
    assert config.capture_response_headers is True
    assert config.capture_reasoning_content is True
    assert config.capture_tool_calls is False
    assert config.endpoint_metadata["deployment"] == "local-vllm"


def test_load_config_supports_environment_capture_fields(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "environment.yaml",
        "\n".join(
            [
                "engine_version: 0.8.0",
                "model_revision: abc123",
                "cuda_version: '12.4'",
                "gpu_type: NVIDIA L40S",
                "gpu_count: 1",
                "backend_launch_command: vllm serve example/model --api-key secret",
                "config_sha256: config-hash",
                "workload_sha256: workload-hash",
            ]
        ),
    )

    config = load_config(config_path)

    assert config.engine_version == "0.8.0"
    assert config.model_revision == "abc123"
    assert config.cuda_version == "12.4"
    assert config.gpu_type == "NVIDIA L40S"
    assert config.gpu_count == 1
    assert config.backend_launch_command == "vllm serve example/model --api-key secret"
    assert config.config_sha256 == "config-hash"
    assert config.workload_sha256 == "workload-hash"


def test_load_config_rejects_unknown_endpoint_type(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "bad_endpoint.yaml", "endpoint_type: custom_engine")

    with pytest.raises(ConfigError, match="endpoint_type"):
        load_config(config_path)

