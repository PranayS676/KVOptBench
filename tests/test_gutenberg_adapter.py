import json
from pathlib import Path

import pytest

from kvoptbench.datasets.gutenberg import GutenbergAdapter
from kvoptbench.datasets.manifest import DatasetPrepareOptions
from kvoptbench.runner.experiment import load_workload


FIXTURE = Path("tests/fixtures/datasets/gutenberg")


def _prepare(tmp_path: Path, mode: str, max_items: int | None = None):
    options = DatasetPrepareOptions(
        source="gutenberg",
        mode=mode,
        source_path=FIXTURE,
        out=tmp_path / f"gutenberg_{mode}.jsonl",
        manifest=tmp_path / f"gutenberg_{mode}_manifest.json",
        max_items=max_items,
        target_input_tokens=256,
        target_output_tokens=64,
        context_buckets=(128, 256),
        book_ids=("1342", "84"),
        seed=7,
    )
    result = GutenbergAdapter().prepare(options)
    items = load_workload(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    return result, items, manifest


def test_gutenberg_needle_adapter_writes_deterministic_workload(tmp_path: Path) -> None:
    result, items, manifest = _prepare(tmp_path, "needle", max_items=4)
    first_output = result.output_path.read_text(encoding="utf-8")
    result_again, _, _ = _prepare(tmp_path, "needle", max_items=4)

    assert first_output == result_again.output_path.read_text(encoding="utf-8")
    assert len(items) == 4
    assert manifest["adapter_name"] == "gutenberg"
    assert manifest["mode"] == "needle"
    assert manifest["context_buckets"] == [128, 256]
    assert all(item.eval_type == "needle" for item in items)
    assert all(item.expected_answer for item in items)
    assert all(item.expected_answer in item.prompt for item in items)
    assert all(item.metadata["needle_id"] for item in items)
    assert all(item.metadata["rights_note"] for item in items)


@pytest.mark.parametrize(
    ("mode", "expected_eval"),
    [
        ("no_needle_control", "contains_expected"),
        ("multi_needle", "needle"),
        ("conflicting_needle", "needle"),
    ],
)
def test_gutenberg_control_modes_emit_valid_rows(
    tmp_path: Path, mode: str, expected_eval: str
) -> None:
    _, items, manifest = _prepare(tmp_path, mode, max_items=2)

    assert len(items) == 2
    assert manifest["mode"] == mode
    assert all(item.eval_type == expected_eval for item in items)
    assert all(item.metadata["mode"] == mode for item in items)
    assert all(item.metadata["prompt_hash"] for item in items)
    assert all(item.metadata["measured_input_tokens"] > 0 for item in items)


def test_gutenberg_manifest_records_redistribution_policy(tmp_path: Path) -> None:
    _, _, manifest = _prepare(tmp_path, "needle", max_items=1)

    assert manifest["license_review_status"] == "fixture_only"
    assert manifest["redistribution_policy"] == "tiny_fixture_allowed"
    assert manifest["workload_sha256"]
    assert manifest["prompt_template_hash"]
