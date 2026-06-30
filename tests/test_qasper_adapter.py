import json
from pathlib import Path

from kvoptbench.datasets.manifest import DatasetPrepareOptions
from kvoptbench.datasets.qasper import QasperAdapter
from kvoptbench.runner.experiment import load_workload


FIXTURE = Path("tests/fixtures/datasets/qasper_tiny.jsonl")


def _prepare(tmp_path: Path, mode: str, max_items: int | None = None):
    options = DatasetPrepareOptions(
        source="qasper",
        mode=mode,
        split="validation",
        source_path=FIXTURE,
        out=tmp_path / f"qasper_{mode}.jsonl",
        manifest=tmp_path / f"qasper_{mode}_manifest.json",
        max_items=max_items,
        target_input_tokens=512,
        target_output_tokens=64,
    )
    result = QasperAdapter().prepare(options)
    items = load_workload(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    return result, items, manifest


def test_qasper_shared_prefix_adapter_writes_valid_workload_and_manifest(tmp_path: Path) -> None:
    result, items, manifest = _prepare(tmp_path, "shared_prefix")

    assert result.row_count == 4
    assert len(items) == 4
    assert manifest["adapter_name"] == "qasper"
    assert manifest["mode"] == "shared_prefix"
    assert manifest["row_count"] == 4
    assert manifest["workload_sha256"] == result.workload_sha256
    assert manifest["prefix_group_count"] == 2
    assert manifest["license_review_status"] == "fixture_only"
    assert manifest["redistribution_policy"] == "tiny_fixture_allowed"

    alpha_items = [item for item in items if item.metadata["source_document_id"] == "paper-alpha"]
    assert len({item.prefix_group_id for item in alpha_items}) == 1
    assert len({item.metadata["prefix_hash"] for item in alpha_items}) == 1
    assert all(item.shared_prefix_tokens > 0 for item in items)
    assert all(item.metadata["prompt_hash"] for item in items)
    assert all(item.metadata["measured_input_tokens"] > 0 for item in items)
    assert all(item.metadata["redistributable_prompt"] is False for item in items)


def test_qasper_random_prefix_control_avoids_intentional_prefix_reuse(tmp_path: Path) -> None:
    _, items, manifest = _prepare(tmp_path, "random_prefix")

    assert manifest["mode"] == "random_prefix"
    assert all(item.prefix_group_id is None for item in items)
    assert all(item.shared_prefix_tokens == 0 for item in items)
    assert all(
        item.metadata["source_document_id"] != item.metadata["question_document_id"]
        for item in items
    )
    assert len({item.metadata["prefix_hash"] for item in items}) == len(items)


def test_qasper_partial_prefix_sweep_records_overlap_ratios(tmp_path: Path) -> None:
    _, items, manifest = _prepare(tmp_path, "partial_prefix_sweep")

    ratios = [item.metadata["prefix_overlap_ratio"] for item in items]
    assert ratios == [0.0, 0.25, 0.5, 0.75, 0.9]
    assert manifest["mode"] == "partial_prefix_sweep"
    assert items[0].prefix_group_id is None
    assert all(item.prefix_group_id == "qasper_partial_paper-alpha" for item in items[1:])
    assert [item.shared_prefix_tokens for item in items] == sorted(
        item.shared_prefix_tokens for item in items
    )


def test_qasper_adapter_honors_max_items(tmp_path: Path) -> None:
    result, items, manifest = _prepare(tmp_path, "shared_prefix", max_items=2)

    assert result.row_count == 2
    assert len(items) == 2
    assert manifest["max_items_requested"] == 2
    assert manifest["max_items_emitted"] == 2
