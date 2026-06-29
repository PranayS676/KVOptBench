"""Workload generation CLI and dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvoptbench.schemas import WorkloadItem
from kvoptbench.workloads import (
    agentic,
    decode_heavy,
    needle,
    rag,
    random_prefix,
    shared_prefix,
    tool_calling,
)

GENERATORS = {
    "shared_prefix": shared_prefix.generate,
    "random_prefix": random_prefix.generate,
    "decode_heavy": decode_heavy.generate,
    "long_context_needle": needle.generate,
    "needle": needle.generate,
    "rag": rag.generate,
    "tool_calling": tool_calling.generate,
    "agentic_coding": agentic.generate,
}


def generate_items(
    *,
    profile: str,
    count: int = 10,
    target_input_tokens: int = 32768,
    target_output_tokens: int = 256,
) -> list[WorkloadItem]:
    """Generate workload items by profile name."""
    if profile not in GENERATORS:
        valid = ", ".join(sorted(GENERATORS))
        raise ValueError(f"Unknown workload profile '{profile}'. Valid profiles: {valid}")
    return list(GENERATORS[profile](count, target_input_tokens, target_output_tokens))


def generate_to_file(
    *,
    profile: str,
    out: str | Path,
    count: int = 10,
    target_input_tokens: int = 32768,
    target_output_tokens: int = 256,
) -> int:
    """Generate a workload JSONL file and return the number of rows written."""
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = generate_items(
        profile=profile,
        count=count,
        target_input_tokens=target_input_tokens,
        target_output_tokens=target_output_tokens,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KVOptBench workload JSONL.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--target-input-tokens", type=int, default=32768)
    parser.add_argument("--target-output-tokens", type=int, default=256)
    args = parser.parse_args()
    written = generate_to_file(
        profile=args.profile,
        out=args.out,
        count=args.count,
        target_input_tokens=args.target_input_tokens,
        target_output_tokens=args.target_output_tokens,
    )
    print(f"Wrote {written} tasks to {args.out}")


if __name__ == "__main__":
    main()

