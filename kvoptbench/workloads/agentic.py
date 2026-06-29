"""Agentic coding workflow workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    items = []
    for idx in range(1, count + 1):
        answer = f"agentic-success-{idx:04d}"
        prompt = (
            "Simulate an agentic coding workflow: plan, implement, review, and summarize.\n"
            f"{filler_words(max(1, target_input_tokens - 40), f'agent-context-{idx}')}\n"
            f"The final summary must include marker {answer}.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"agentic_coding_{idx:04d}",
                workload="agentic_coding",
                category="agentic",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                eval_type="llm_judge_placeholder",
            )
        )
    return items

