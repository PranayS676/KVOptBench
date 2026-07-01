"""Decode-heavy generation workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, lifecycle_metadata, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    items = []
    for idx in range(1, count + 1):
        answer = f"decode-plan-{idx:04d}"
        prompt = (
            "Generate a long technical implementation plan for a Python service.\n"
            f"{filler_words(max(1, target_input_tokens - 30), f'decode-context-{idx}')}\n"
            f"Include this marker in the final paragraph: {answer}\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"decode_heavy_{idx:04d}",
                workload="decode_heavy_generation",
                category="decode",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                eval_type="contains_expected",
                metadata=lifecycle_metadata(
                    lifecycle_pattern="kv_generation",
                    workload_profile="decode_heavy",
                    request_group_id="decode_generation_group_001",
                    reuse_hint="none",
                    required_evaluators=["output_validity"],
                    required_metrics=["output_tokens_per_second", "latency_ms", "error_rate"],
                    recommended_metrics=["input_tokens", "output_tokens"],
                ),
            )
        )
    return items

