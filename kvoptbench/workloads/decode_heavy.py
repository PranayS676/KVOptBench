"""Decode-heavy generation workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, make_item


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
            )
        )
    return items

