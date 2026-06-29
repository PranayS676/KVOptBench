"""Tool-calling placeholder workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    items = []
    schema = {
        "type": "object",
        "required": ["tool", "arguments"],
        "properties": {
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
    }
    for idx in range(1, count + 1):
        answer = f"lookup_order_{idx:04d}"
        prompt = (
            "Return a JSON tool call with tool and arguments fields.\n"
            f"{filler_words(max(1, target_input_tokens - 30), f'tool-context-{idx}')}\n"
            f"The correct tool is {answer}.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"tool_calling_{idx:04d}",
                workload="tool_calling",
                category="structured_output",
                prompt=prompt,
                expected_answer=answer,
                expected_schema=schema,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                eval_type="tool_calling_placeholder",
            )
        )
    return items

