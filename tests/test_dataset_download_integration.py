import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kvoptbench.cli import app
from kvoptbench.runner.experiment import load_workload

pytestmark = pytest.mark.skipif(
    os.getenv("KVOPTBENCH_DATASET_DOWNLOAD") != "1",
    reason="set KVOPTBENCH_DATASET_DOWNLOAD=1 to exercise real public dataset downloads",
)


def test_public_dataset_download_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    cases = [
        ("qasper", "shared_prefix", ["--split", "validation", "--max-items", "1"]),
        (
            "longbench",
            "long_context_qa",
            ["--subset", "qasper", "--split", "test", "--max-items", "1"],
        ),
        (
            "gutenberg",
            "needle",
            ["--book-ids", "1342", "--context-buckets", "512", "--max-items", "1"],
        ),
        ("beir_scifact", "rag", ["--max-items", "1"]),
        ("bfcl", "tool_calling", ["--subset", "BFCL_v3_simple", "--max-items", "1"]),
    ]
    for source, mode, extra_args in cases:
        out = tmp_path / f"{source}.jsonl"
        manifest = tmp_path / f"{source}_manifest.json"
        result = runner.invoke(
            app,
            [
                "dataset",
                "prepare",
                "--source",
                source,
                "--mode",
                mode,
                "--download",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--out",
                str(out),
                "--manifest",
                str(manifest),
                "--target-input-tokens",
                "512",
                "--target-output-tokens",
                "64",
                *extra_args,
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert len(load_workload(out)) == 1
        assert manifest.exists()
