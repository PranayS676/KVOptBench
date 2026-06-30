import json
from pathlib import Path

from kvoptbench.datasets.hashing import sha256_json, sha256_text
from kvoptbench.datasets.manifest import DatasetManifest, DatasetPrepareOptions


def test_hash_helpers_are_stable_and_order_independent() -> None:
    assert sha256_text("prefix") == sha256_text("prefix")
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_dataset_manifest_requires_reproducibility_fields() -> None:
    manifest = DatasetManifest(
        adapter_name="qasper",
        adapter_version="0.1.0",
        dataset_name="QASPER",
        dataset="qasper",
        dataset_source_url="https://huggingface.co/datasets/allenai/qasper",
        source_url="https://huggingface.co/datasets/allenai/qasper",
        license="cc-by-4.0",
        rights_note="fixture",
        license_review_status="fixture_only",
        redistribution_policy="tiny_fixture_allowed",
        adapter="qasper",
        mode="shared_prefix",
        generation_command="kvoptbench dataset prepare ...",
        row_count=2,
        workload_sha256="abc",
        token_count_method="char_approx_4",
        prompt_template="qasper_shared_prefix_v1",
        prompt_template_hash="def",
    )

    payload = manifest.model_dump()
    assert payload["schema_version"] == "1"
    assert payload["kvoptbench_version"]
    assert payload["git_commit"]
    assert payload["workload_sha256"] == "abc"


def test_dataset_prepare_options_parse_context_and_book_ids(tmp_path: Path) -> None:
    options = DatasetPrepareOptions(
        source="gutenberg",
        mode="needle",
        source_path=tmp_path,
        out=tmp_path / "out.jsonl",
        manifest=tmp_path / "manifest.json",
        context_buckets=(128, 256),
        book_ids=("1342", "84"),
        cache_dir=tmp_path / "cache",
        dataset_revision="abc123",
        subset=("qasper",),
        force=True,
    )

    assert options.context_buckets == (128, 256)
    assert options.book_ids == ("1342", "84")
    assert options.cache_dir == tmp_path / "cache"
    assert options.dataset_revision == "abc123"
    assert options.subset == ("qasper",)
    assert options.force is True


def test_manifest_json_is_serializable(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        adapter_name="fixture",
        adapter_version="0.1.0",
        dataset_name="Fixture",
        dataset="fixture",
        dataset_source_url="https://example.test",
        source_url="https://example.test",
        rights_note="fixture",
        license_review_status="fixture_only",
        redistribution_policy="tiny_fixture_allowed",
        adapter="fixture",
        mode="test",
        generation_command="test",
        row_count=0,
        workload_sha256="hash",
        token_count_method="char_approx_4",
        prompt_template="fixture_v1",
        prompt_template_hash="template_hash",
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["dataset_name"] == "Fixture"
