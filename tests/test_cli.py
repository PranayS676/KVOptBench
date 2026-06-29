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
