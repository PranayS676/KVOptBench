"""Shared-prefix long-document QA workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    shared_prefix_tokens = max(1, int(target_input_tokens * 0.85))
    prefix = filler_words(shared_prefix_tokens, "shared-document")
    items = []
    for idx in range(1, count + 1):
        answer = f"shared-answer-{idx:04d}"
        prompt = (
            "You are answering questions over a shared long document.\n"
            f"{prefix}\n"
            f"Question {idx}: Return the exact marker.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"shared_prefix_32k_{idx:04d}",
                workload="shared_prefix_long_doc",
                category="prefix_cache",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                prefix_group_id="shared_doc_001",
                shared_prefix_tokens=shared_prefix_tokens,
                eval_type="contains_expected",
            )
        )
    return items

