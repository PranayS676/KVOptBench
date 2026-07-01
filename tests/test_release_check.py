import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from kvoptbench.cli import app
from kvoptbench.release import run_release_check


def test_release_check_passes_for_current_repo() -> None:
    report = run_release_check(Path("."))

    assert report.ok is True
    check_names = {check.name for check in report.checks}
    assert {
        "version_consistency",
        "schema_snapshots",
        "bundled_resources",
        "public_files",
        "public_text_safety",
    }.issubset(check_names)


def test_release_check_cli_emits_json() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["release-check", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert any(check["name"] == "schema_snapshots" for check in payload["checks"])


def test_version_command_and_module_entrypoint() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "KVOptBench" in result.stdout

    module_result = subprocess.run(
        [sys.executable, "-m", "kvoptbench", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert module_result.returncode == 0
    assert "KVOptBench" in module_result.stdout
