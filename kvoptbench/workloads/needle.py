"""Long-context needle retrieval workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, lifecycle_metadata, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    items = []
    half = max(1, target_input_tokens // 2)
    for idx in range(1, count + 1):
        answer = f"needle-code-{idx:04d}"
        prompt = (
            "Find the hidden needle in the long context and return it exactly.\n"
            f"{filler_words(half, f'needle-before-{idx}')}\n"
            f"The hidden needle is {answer}.\n"
            f"{filler_words(max(1, target_input_tokens - half), f'needle-after-{idx}')}\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"needle_{idx:04d}",
                workload="long_context_needle",
                category="long_context",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                eval_type="needle",
                metadata=lifecycle_metadata(
                    lifecycle_pattern="loading",
                    workload_profile="long_context_qa",
                    request_group_id="needle_loading_group_001",
                    reuse_hint="shared_prefix",
                    required_evaluators=["answer_correctness", "factuality"],
                    required_metrics=["input_tokens", "output_tokens", "latency_ms"],
                    recommended_metrics=["cache_load_ms", "cache_hit_rate"],
                ),
            )
        )
    return items

