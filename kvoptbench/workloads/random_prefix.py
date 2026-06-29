"""Random-prefix control workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    items = []
    for idx in range(1, count + 1):
        answer = f"random-answer-{idx:04d}"
        prefix = filler_words(max(1, target_input_tokens - 20), f"random-doc-{idx}")
        prompt = (
            "This is a random-prefix control prompt with no intentional reuse.\n"
            f"{prefix}\n"
            f"Question {idx}: Return the exact marker.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"random_prefix_{idx:04d}",
                workload="random_prefix_control",
                category="prefix_cache_control",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                eval_type="contains_expected",
            )
        )
    return items

