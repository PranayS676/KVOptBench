from typer.testing import CliRunner

from kvoptbench.cli import app


def test_engine_command_cli_prints_preview_without_launching() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "engine-command",
            "--engine",
            "vllm",
            "--strategy",
            "cache_on",
            "--model-id",
            "example/model",
        ],
    )

    assert result.exit_code == 0
    assert "--enable-prefix-caching" in result.stdout
    assert "does not launch" in result.stdout


def test_cache_plan_cli_writes_yaml_configs(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cache-plan",
            "--plan-dir",
            str(tmp_path / "plan"),
            "--experiment-prefix",
            "cache_exp",
            "--provider",
            "mock",
            "--engine",
            "sglang",
            "--model-id",
            "mock-frontier-model",
            "--base-url",
            "http://127.0.0.1:30000/v1",
            "--shared-workload-file",
            "workloads/generated/shared.jsonl",
            "--random-workload-file",
            "workloads/generated/random.jsonl",
            "--output-dir",
            "results/raw",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 8 cache experiment configs" in result.stdout
    assert len(list((tmp_path / "plan").glob("*.yaml"))) == 8
