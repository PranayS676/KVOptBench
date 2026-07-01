"""Partial-prefix reuse workload for cache ratio sweeps."""

from __future__ import annotations

from kvoptbench.workloads.common import filler_words, lifecycle_metadata, make_item

DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    ratios = _ratios_for_count(count)
    shared_seed = filler_words(target_input_tokens, "partial-shared-document")
    items = []
    for idx, ratio in enumerate(ratios, start=1):
        shared_prefix_tokens = int(target_input_tokens * ratio)
        unique_tokens = max(1, target_input_tokens - shared_prefix_tokens)
        shared_part = " ".join(shared_seed.split()[:shared_prefix_tokens])
        unique_part = filler_words(unique_tokens, f"partial-unique-{idx}")
        answer = f"partial-prefix-answer-{int(ratio * 100):03d}"
        prompt = (
            "You are measuring partial prefix reuse.\n"
            f"{shared_part}\n"
            f"{unique_part}\n"
            f"Shared prefix ratio: {ratio:.2f}.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"partial_prefix_{int(ratio * 100):03d}_{idx:04d}",
                workload="partial_prefix_reuse",
                category="prefix_cache_ratio",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=target_input_tokens,
                target_output_tokens=target_output_tokens,
                prefix_group_id="partial_prefix_doc_001" if shared_prefix_tokens > 0 else None,
                shared_prefix_tokens=shared_prefix_tokens,
                eval_type="contains_expected",
                metadata={
                    **lifecycle_metadata(
                        lifecycle_pattern="retrieval",
                        workload_profile="long_context_qa",
                        request_group_id="partial_prefix_doc_001",
                        reuse_hint="shared_prefix" if shared_prefix_tokens > 0 else "none",
                        required_evaluators=["answer_correctness", "factuality"],
                        required_metrics=["input_tokens", "output_tokens", "latency_ms"],
                        recommended_metrics=["cache_hit_rate", "cache_load_ms"],
                    ),
                    "shared_prefix_ratio": ratio,
                },
            )
        )
    return items


def _ratios_for_count(count: int) -> list[float]:
    if count <= len(DEFAULT_RATIOS):
        return list(DEFAULT_RATIOS[:count])
    ratios = list(DEFAULT_RATIOS)
    while len(ratios) < count:
        ratios.append(DEFAULT_RATIOS[len(ratios) % len(DEFAULT_RATIOS)])
    return ratios
