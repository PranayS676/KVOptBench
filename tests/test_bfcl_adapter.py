import json
from pathlib import Path

from kvoptbench.datasets.bfcl import BfclAdapter
from kvoptbench.datasets.download import DownloadResult
from kvoptbench.datasets.manifest import DatasetPrepareOptions
from kvoptbench.runner.experiment import load_workload


FIXTURE = Path("tests/fixtures/datasets/bfcl_tiny.json")


def test_bfcl_adapter_prepares_tool_calling_fixture(tmp_path: Path) -> None:
    options = DatasetPrepareOptions(
        source="bfcl",
        mode="tool_calling",
        source_path=FIXTURE,
        out=tmp_path / "bfcl.jsonl",
        manifest=tmp_path / "manifest.json",
        max_items=1,
        target_input_tokens=256,
        target_output_tokens=64,
    )

    result = BfclAdapter().prepare(options)
    items = load_workload(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.row_count == 1
    assert items[0].eval_type == "tool_calling_placeholder"
    assert items[0].expected_answer == "lookup_order"
    assert items[0].expected_schema["required"] == ["tool", "arguments"]
    assert manifest["adapter_name"] == "bfcl"


def test_bfcl_download_uses_huggingface_file_cache(monkeypatch, tmp_path: Path) -> None:
    def fake_download_file(*args, **kwargs):
        return DownloadResult(
            path=FIXTURE,
            url="https://example.test/BFCL_v3_simple.json",
            sha256="sha",
            downloaded_at="2026-06-30T00:00:00+00:00",
        )

    monkeypatch.setattr("kvoptbench.datasets.bfcl.download_file", fake_download_file)
    options = DatasetPrepareOptions(
        source="bfcl",
        mode="tool_calling",
        download=True,
        cache_dir=tmp_path / "cache",
        out=tmp_path / "bfcl_download.jsonl",
        manifest=tmp_path / "manifest.json",
        target_input_tokens=256,
        target_output_tokens=64,
    )

    result = BfclAdapter().prepare(options)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.row_count == 2
    assert manifest["download_method"] == "huggingface_file"
    assert manifest["max_items_requested"] == 100
