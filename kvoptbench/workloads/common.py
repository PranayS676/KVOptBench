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


def lifecycle_metadata(
    *,
    lifecycle_pattern: str,
    workload_profile: str,
    request_group_id: str,
    ordering: str = "fixed",
    session_boundary: str = "request_group",
    reuse_hint: str = "none",
    required_evaluators: list[str] | None = None,
    required_metrics: list[str] | None = None,
    recommended_metrics: list[str] | None = None,
) -> dict:
    """Describe cache-lifecycle workload intent without controlling backend KV state."""
    return {
        "lifecycle_mode": {
            "mode_source": "scbench_inspired_workload_lifecycle_metadata",
            "kvoptbench_implements_scbench": False,
            "lifecycle_pattern": lifecycle_pattern,
            "workload_profile": workload_profile,
            "request_group": {
                "group_id": request_group_id,
                "ordering": ordering,
                "session_boundary": session_boundary,
                "reuse_hint": reuse_hint,
            },
            "evaluation": {
                "required_evaluators": required_evaluators or [],
                "required_metrics": required_metrics or [],
                "recommended_metrics": recommended_metrics or [],
            },
        }
    }


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

