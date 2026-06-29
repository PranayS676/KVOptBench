"""Shared helpers for synthetic workload generation."""

from __future__ import annotations

from kvoptbench.schemas import WorkloadItem


def filler_words(target_tokens: int, seed: str) -> str:
    """Create deterministic filler text with roughly target_tokens whitespace tokens."""
    if target_tokens <= 0:
        return ""
    base = [
        seed,
        "cache",
        "aware",
        "frontier",
        "inference",
        "benchmark",
        "prefill",
        "decode",
        "latency",
        "quality",
    ]
    words = [base[index % len(base)] for index in range(target_tokens)]
    return " ".join(words)


def make_item(
    *,
    task_id: str,
    workload: str,
    category: str,
    prompt: str,
    expected_answer: str | None,
    target_input_tokens: int,
    target_output_tokens: int,
    eval_type: str,
    expected_schema: dict | None = None,
    prefix_group_id: str | None = None,
    shared_prefix_tokens: int = 0,
    metadata: dict | None = None,
) -> WorkloadItem:
    """Construct a validated workload item."""
    return WorkloadItem(
        task_id=task_id,
        workload=workload,
        category=category,
        prompt=prompt,
        expected_answer=expected_answer,
        expected_schema=expected_schema,
        target_input_tokens=target_input_tokens,
        target_output_tokens=target_output_tokens,
        prefix_group_id=prefix_group_id,
        shared_prefix_tokens=shared_prefix_tokens,
        eval_type=eval_type,
        metadata=metadata or {},
    )

