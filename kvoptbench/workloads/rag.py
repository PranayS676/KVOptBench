"""RAG faithfulness placeholder workload."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, make_item


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    items = []
    for idx in range(1, count + 1):
        answer = f"rag-fact-{idx:04d}"
        prompt = (
            "Answer only from the provided source chunks and cite the source id.\n"
            f"Source chunk A{idx}: The supported answer is {answer}.\n"
            f"{filler_words(max(1, target_input_tokens - 30), f'rag-context-{idx}')}\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"rag_{idx:04d}",
                workload="rag_faithfulness",
                category="rag",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                eval_type="rag_placeholder",
                metadata={"source_id": f"A{idx}"},
            )
        )
    return items

