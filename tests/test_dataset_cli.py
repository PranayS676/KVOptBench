import json
from pathlib import Path

from typer.testing import CliRunner

from kvoptbench.cli import app
from kvoptbench.runner.experiment import load_workload


def test_dataset_prepare_cli_writes_qasper_workload_and_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "qasper_shared.jsonl"
    manifest = tmp_path / "qasper_manifest.json"

    result = runner.invoke(
        app,
        [
            "dataset",
            "prepare",
            "--source",
            "qasper",
            "--mode",
            "shared_prefix",
            "--split",
            "validation",
            "--source-path",
            "tests/fixtures/datasets/qasper_tiny.json",
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--max-items",
            "2",
            "--target-input-tokens",
            "512",
            "--target-output-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 dataset workload rows" in result.stdout
    assert len(load_workload(out)) == 2
    assert json.loads(manifest.read_text(encoding="utf-8"))["adapter_name"] == "qasper"


def test_dataset_prepare_cli_writes_gutenberg_workload_and_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "gutenberg_needle.jsonl"
    manifest = tmp_path / "gutenberg_manifest.json"

    result = runner.invoke(
        app,
        [
            "dataset",
            "prepare",
            "--source",
            "gutenberg",
            "--mode",
            "needle",
            "--source-path",
            "tests/fixtures/datasets/gutenberg",
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--context-buckets",
            "128,256",
            "--book-ids",
            "1342",
            "--max-items",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 2 dataset workload rows" in result.stdout
    assert len(load_workload(out)) == 2
    assert json.loads(manifest.read_text(encoding="utf-8"))["adapter_name"] == "gutenberg"


def test_dataset_prepare_cli_writes_longbench_workload_with_subset(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "longbench.jsonl"
    manifest = tmp_path / "longbench_manifest.json"

    result = runner.invoke(
        app,
        [
            "dataset",
            "prepare",
            "--source",
            "longbench",
            "--mode",
            "long_context_qa",
            "--source-path",
            "tests/fixtures/datasets/longbench_tiny.json",
            "--subset",
            "qasper",
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--target-input-tokens",
            "256",
            "--target-output-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 dataset workload rows" in result.stdout
    assert len(load_workload(out)) == 1
    assert json.loads(manifest.read_text(encoding="utf-8"))["adapter_name"] == "longbench"


def test_dataset_prepare_cli_rejects_unknown_source(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "dataset",
            "prepare",
            "--source",
            "missing",
            "--mode",
            "shared_prefix",
            "--source-path",
            "tests/fixtures/datasets/qasper_tiny.jsonl",
            "--out",
            str(tmp_path / "out.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code != 0
    assert "Unknown dataset adapter" in result.stdout
