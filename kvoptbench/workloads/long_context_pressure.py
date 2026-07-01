"""Long-context pressure sweep workload."""

from __future__ import annotations

from collections.abc import Iterable

from kvoptbench.workloads.common import filler_words, lifecycle_metadata, make_item

DEFAULT_CONTEXT_BUCKETS = (4096, 16384, 32768, 65536, 131072)


def generate(
    count: int,
    target_input_tokens: int,
    target_output_tokens: int,
    *,
    context_buckets: Iterable[int] | None = None,
):
    """Generate deterministic prompts across increasing context-token buckets."""
    _ = target_input_tokens
    buckets = _normalize_context_buckets(context_buckets)
    items = []
    for index in range(count):
        bucket = buckets[index % len(buckets)]
        pressure_level = _pressure_level(bucket)
        expected_pressure = _expected_pressure(bucket)
        answer = f"long-context-marker-{bucket}-{index + 1:04d}"
        prompt = (
            "You are measuring long-context inference pressure.\n"
            f"{filler_words(bucket, f'long-context-{bucket}')}\n"
            "Return the exact marker after reading the full context.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"long_context_{bucket}_{index + 1:04d}",
                workload="long_context_pressure",
                category="long_context",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=bucket,
                target_output_tokens=target_output_tokens,
                prefix_group_id=None,
                shared_prefix_tokens=0,
                eval_type="contains_expected",
                metadata={
                    **lifecycle_metadata(
                        lifecycle_pattern="compression",
                        workload_profile="long_context_qa",
                        request_group_id="long_context_pressure_group_001",
                        reuse_hint="none",
                        required_evaluators=["answer_correctness", "factuality"],
                        required_metrics=["input_tokens", "output_tokens", "latency_ms"],
                        recommended_metrics=["gpu_memory_peak_gb", "cache_hit_rate"],
                    ),
                    "context_token_bucket": bucket,
                    "pressure_level": pressure_level,
                    "expected_pressure": expected_pressure,
                },
            )
        )
    return items


def _normalize_context_buckets(context_buckets: Iterable[int] | None) -> tuple[int, ...]:
    if context_buckets is None:
        return DEFAULT_CONTEXT_BUCKETS
    buckets = tuple(int(bucket) for bucket in context_buckets)
    if not buckets:
        raise ValueError("context_buckets must contain at least one bucket")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("context_buckets must be positive integers")
    return buckets


def _pressure_level(bucket: int) -> str:
    if bucket <= 4096:
        return "baseline"
    if bucket <= 16384:
        return "moderate"
    if bucket <= 32768:
        return "high"
    if bucket <= 65536:
        return "extreme"
    return "frontier"


def _expected_pressure(bucket: int) -> str:
    if bucket <= 4096:
        return "stable"
    if bucket <= 32768:
        return "prefill_latency_growth"
    return "memory_pressure_candidate"
