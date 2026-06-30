"""Aggregate request-level JSONL results to CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.jsonl"))


def _read_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in _jsonl_files(input_path):
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSONL result rows found under {input_path}")
    return rows


def summarize_results(input_path: str | Path, output_path: str | Path) -> Path:
    """Write a grouped summary CSV from raw JSONL result rows."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = _read_rows(input_path)
    frame = pd.DataFrame(rows)
    group_cols = [
        "experiment_id",
        "provider",
        "engine",
        "model_id",
        "strategy",
        "workload",
        "concurrency",
    ]

    summaries: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        summary = dict(zip(group_cols, keys, strict=True))
        summary["requests"] = int(len(group))
        summary["successes"] = int(group["success"].fillna(False).sum())
        summary["errors"] = int(summary["requests"] - summary["successes"])
        summary["success_rate"] = round(summary["successes"] / summary["requests"], 4)
        for col in [
            "ttft_ms",
            "tpot_ms",
            "itl_ms",
            "e2e_latency_ms",
            "requests_per_second",
            "input_tokens_per_second",
            "output_tokens_per_second",
            "provider_completion_tokens",
            "reasoning_tokens",
            "first_reasoning_token_ms",
            "tool_call_count",
            "quality_score",
            "cache_hit_rate",
            "cache_miss_penalty_ms",
        ]:
            values = pd.to_numeric(group.get(col), errors="coerce")
            if values.notna().any():
                summary[f"{col}_mean"] = round(float(values.mean()), 3)
                summary[f"{col}_p50"] = round(float(values.quantile(0.50)), 3)
                summary[f"{col}_p95"] = round(float(values.quantile(0.95)), 3)
            else:
                summary[f"{col}_mean"] = None
                summary[f"{col}_p50"] = None
                summary[f"{col}_p95"] = None

        for col in ["reasoning_content_present", "visible_answer_missing"]:
            if col in group:
                values = group[col].fillna(False).astype(bool)
                summary[f"{col}_rate"] = round(float(values.mean()), 4)
            else:
                summary[f"{col}_rate"] = None

        missing: set[str] = set()
        for value in group.get("missing_metrics", []):
            if isinstance(value, list):
                missing.update(str(item) for item in value)
            elif isinstance(value, str) and value:
                missing.update(value.split(";"))
        summary["missing_metrics"] = ";".join(sorted(missing))
        summaries.append(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KVOptBench JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize_results(input_path=args.input, output_path=args.output)
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()

