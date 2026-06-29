"""Prefill/decode decomposition workload grid."""

from __future__ import annotations

from itertools import product

from kvoptbench.workloads.common import filler_words, make_item

INPUT_TOKEN_BUCKETS = (512, 2048, 8192, 32768)
OUTPUT_TOKEN_BUCKETS = (32, 128, 512)


def generate(count: int, target_input_tokens: int, target_output_tokens: int):
    """Generate a controlled input/output grid for prefill vs decode analysis."""
    _ = (target_input_tokens, target_output_tokens)
    combinations = list(product(INPUT_TOKEN_BUCKETS, OUTPUT_TOKEN_BUCKETS))
    items = []
    for index, (input_bucket, output_bucket) in enumerate(combinations[:count], start=1):
        expected_bottleneck = _expected_bottleneck(input_bucket, output_bucket)
        answer = f"prefill-decode-answer-{input_bucket}-{output_bucket}"
        prompt = (
            "You are measuring prefill and decode pressure.\n"
            f"{filler_words(input_bucket, f'prefill-decode-{input_bucket}')}\n"
            f"Return the exact marker after producing about {output_bucket} output tokens.\n"
            f"EXPECTED_ANSWER: {answer}"
        )
        items.append(
            make_item(
                task_id=f"prefill_decode_{input_bucket}_{output_bucket}_{index:04d}",
                workload="prefill_decode_grid",
                category="prefill_decode",
                prompt=prompt,
                expected_answer=answer,
                target_input_tokens=input_bucket,
                target_output_tokens=output_bucket,
                prefix_group_id=None,
                shared_prefix_tokens=0,
                eval_type="contains_expected",
                metadata={
                    "input_token_bucket": input_bucket,
                    "output_token_bucket": output_bucket,
                    "expected_bottleneck": expected_bottleneck,
                },
            )
        )
    return items


def _expected_bottleneck(input_bucket: int, output_bucket: int) -> str:
    if input_bucket >= 8192 and output_bucket <= 128:
        return "prefill_bound"
    if input_bucket <= 2048 and output_bucket >= 512:
        return "decode_bound"
    if input_bucket >= 8192 and output_bucket >= 512:
        return "mixed"
    if output_bucket >= 512:
        return "decode_bound"
    return "mixed"
