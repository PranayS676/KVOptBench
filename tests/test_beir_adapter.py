import json
import zipfile
from pathlib import Path

from kvoptbench.datasets.beir import BeirScifactAdapter
from kvoptbench.datasets.download import DownloadResult
from kvoptbench.datasets.manifest import DatasetPrepareOptions
from kvoptbench.runner.experiment import load_workload


FIXTURE = Path("tests/fixtures/datasets/beir_scifact")


def test_beir_scifact_adapter_prepares_rag_fixture(tmp_path: Path) -> None:
    options = DatasetPrepareOptions(
        source="beir_scifact",
        mode="rag",
        source_path=FIXTURE,
        out=tmp_path / "beir.jsonl",
        manifest=tmp_path / "manifest.json",
        max_items=1,
        target_input_tokens=256,
        target_output_tokens=64,
    )

    result = BeirScifactAdapter().prepare(options)
    items = load_workload(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.row_count == 1
    assert items[0].eval_type == "rag_placeholder"
    assert items[0].expected_answer == "doc-1"
    assert "Source doc-1" in items[0].prompt
    assert manifest["adapter_name"] == "beir_scifact"


def test_beir_scifact_download_extracts_cached_zip(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive, "w") as zip_handle:
        for file_name in ("corpus.json", "queries.json", "qrels.json"):
            zip_handle.write(FIXTURE / file_name, arcname=f"scifact/{file_name}")

    def fake_download_file(*args, **kwargs):
        return DownloadResult(
            path=archive,
            url="https://example.test/scifact.zip",
            sha256="sha",
            downloaded_at="2026-06-30T00:00:00+00:00",
        )

    monkeypatch.setattr("kvoptbench.datasets.beir.download_file", fake_download_file)
    options = DatasetPrepareOptions(
        source="beir_scifact",
        mode="rag",
        download=True,
        cache_dir=tmp_path / "cache",
        out=tmp_path / "beir_download.jsonl",
        manifest=tmp_path / "manifest.json",
        max_items=1,
        target_input_tokens=256,
        target_output_tokens=64,
    )

    result = BeirScifactAdapter().prepare(options)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.row_count == 1
    assert manifest["download_method"] == "beir_public_zip"
    assert manifest["cache_path"]
