import json
from pathlib import Path

import pytest

from kvoptbench.schemas import WorkloadItem
from kvoptbench.workloads.generate import generate_items, generate_to_file


@pytest.mark.parametrize(
    "profile",
    [
        "shared_prefix",
        "random_prefix",
        "decode_heavy",
        "long_context_needle",
        "rag",
        "tool_calling",
        "agentic_coding",
        "partial_prefix",
    ],
)
def test_generate_profiles_return_valid_items(profile: str) -> None:
    items = generate_items(
        profile=profile,
        count=3,
        target_input_tokens=512,
        target_output_tokens=64,
    )

    assert len(items) == 3
    for item in items:
        validated = WorkloadItem.model_validate(item.model_dump())
        assert validated.task_id
        assert validated.prompt
        assert validated.target_input_tokens == 512
        assert validated.target_output_tokens == 64


def test_shared_prefix_items_share_prefix_group() -> None:
    items = generate_items(
        profile="shared_prefix",
        count=4,
        target_input_tokens=1024,
        target_output_tokens=128,
    )

    assert {item.prefix_group_id for item in items} == {"shared_doc_001"}
    assert all(item.shared_prefix_tokens > 0 for item in items)
    assert all(item.category == "prefix_cache" for item in items)


def test_random_prefix_control_has_no_shared_prefix() -> None:
    items = generate_items(
        profile="random_prefix",
        count=4,
        target_input_tokens=1024,
        target_output_tokens=128,
    )

    assert all(item.prefix_group_id is None for item in items)
    assert all(item.shared_prefix_tokens == 0 for item in items)
    assert len({item.prompt for item in items}) == 4


def test_partial_prefix_workload_covers_expected_ratios() -> None:
    items = generate_items(
        profile="partial_prefix",
        count=6,
        target_input_tokens=1000,
        target_output_tokens=64,
    )

    ratios = [item.metadata["shared_prefix_ratio"] for item in items]
    assert ratios == [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    assert [item.shared_prefix_tokens for item in items] == [0, 250, 500, 750, 900, 1000]
    assert items[0].prefix_group_id is None
    assert all(item.prefix_group_id == "partial_prefix_doc_001" for item in items[1:])


def test_prefill_decode_grid_covers_input_output_buckets() -> None:
    items = generate_items(
        profile="prefill_decode_grid",
        count=12,
        target_input_tokens=32768,
        target_output_tokens=512,
    )

    input_buckets = {item.metadata["input_token_bucket"] for item in items}
    output_buckets = {item.metadata["output_token_bucket"] for item in items}
    expected_bottlenecks = {item.metadata["expected_bottleneck"] for item in items}

    assert input_buckets == {512, 2048, 8192, 32768}
    assert output_buckets == {32, 128, 512}
    assert expected_bottlenecks == {"prefill_bound", "decode_bound", "mixed"}
    assert all(item.workload == "prefill_decode_grid" for item in items)
    assert all(item.category == "prefill_decode" for item in items)


def test_needle_workload_contains_expected_answer() -> None:
    items = generate_items(
        profile="long_context_needle",
        count=2,
        target_input_tokens=1024,
        target_output_tokens=64,
    )

    assert all(item.expected_answer for item in items)
    assert all(item.expected_answer in item.prompt for item in items)
    assert all(item.eval_type == "needle" for item in items)


def test_generate_to_file_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "workload.jsonl"

    count = generate_to_file(
        profile="shared_prefix",
        out=out,
        count=2,
        target_input_tokens=256,
        target_output_tokens=32,
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    assert count == 2
    assert len(lines) == 2
    assert json.loads(lines[0])["workload"] == "shared_prefix_long_doc"

