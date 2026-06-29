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

