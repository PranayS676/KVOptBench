import json
from pathlib import Path

from kvoptbench.datasets.manifest import DatasetPrepareOptions
from kvoptbench.datasets.longbench import LongBenchAdapter
from kvoptbench.runner.experiment import load_workload


FIXTURE = Path("tests/fixtures/datasets/longbench_tiny.json")


def test_longbench_adapter_prepares_local_fixture(tmp_path: Path) -> None:
    options = DatasetPrepareOptions(
        source="longbench",
        mode="long_context_qa",
        source_path=FIXTURE,
        out=tmp_path / "longbench.jsonl",
        manifest=tmp_path / "manifest.json",
        subset=("qasper",),
        target_input_tokens=256,
        target_output_tokens=64,
    )

    result = LongBenchAdapter().prepare(options)
    items = load_workload(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.row_count == 1
    assert items[0].eval_type == "longbench_answer"
    assert items[0].expected_answer == "time to first token"
    assert items[0].metadata["subset"] == "qasper"
    assert items[0].metadata["evaluator"] == "longbench_answer"
    assert manifest["adapter_name"] == "longbench"
    assert manifest["license_review_status"] == "fixture_only"


def test_longbench_download_uses_cache_and_small_default(monkeypatch, tmp_path: Path) -> None:
    def fake_load_hf_dataset_records(dataset_name, **kwargs):
        assert dataset_name == "THUDM/LongBench"
        assert kwargs["max_items"] == 100
        assert kwargs["trust_remote_code"] is True
        return [
            {
                "id": "downloaded-1",
                "subset": kwargs["subset"],
                "context": "A downloaded context says the expected answer is cached rows.",
                "input": "What is the expected answer?",
                "answers": ["cached rows"],
            }
        ]

    monkeypatch.setattr(
        "kvoptbench.datasets.longbench.load_hf_dataset_records",
        fake_load_hf_dataset_records,
    )
    options = DatasetPrepareOptions(
        source="longbench",
        mode="long_context_qa",
        download=True,
        cache_dir=tmp_path / "cache",
        out=tmp_path / "longbench_download.jsonl",
        manifest=tmp_path / "manifest.json",
        target_input_tokens=256,
        target_output_tokens=64,
    )

    result = LongBenchAdapter().prepare(options)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.row_count == 3
    assert manifest["download_method"] == "huggingface.datasets"
    assert manifest["cache_path"]
    assert manifest["max_items_requested"] == 100
