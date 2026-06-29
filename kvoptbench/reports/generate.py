"""Generate markdown reports from summary CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fmt(value) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate_report(input_path: str | Path, output_path: str | Path) -> Path:
    """Generate a markdown report from a summary CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError(f"Summary CSV is empty: {input_path}")

    total_requests = int(frame["requests"].sum()) if "requests" in frame else 0
    avg_success = frame["success_rate"].mean() if "success_rate" in frame else None
    missing_values = sorted(
        {
            metric
            for value in frame.get("missing_metrics", pd.Series(dtype=str)).fillna("")
            for metric in str(value).split(";")
            if metric
        }
    )

    lines = [
        "# KVOptBench Mock Benchmark Report",
        "",
        "## Run Summary",
        "",
        f"- Summary source: `{input_path}`",
        f"- Experiment groups: {len(frame)}",
        f"- Total requests: {total_requests}",
        f"- Mean success rate: {_fmt(avg_success)}",
        "",
        "## Workload Summary",
        "",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "- "
            f"`{row.get('workload', 'unknown')}` on `{row.get('engine', 'unknown')}`/"
            f"`{row.get('strategy', 'unknown')}`: {int(row.get('requests', 0))} requests"
        )

    lines.extend(
        [
            "",
            "## Latency Summary",
            "",
            "| workload | p50 E2E ms | p95 E2E ms |",
            "|---|---:|---:|",
        ]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.get('workload', 'unknown')} | "
            f"{_fmt(row.get('e2e_latency_ms_p50'))} | {_fmt(row.get('e2e_latency_ms_p95'))} |"
        )

    lines.extend(
        [
            "",
            "## TTFT Summary",
            "",
            "| workload | p50 TTFT ms | p95 TTFT ms |",
            "|---|---:|---:|",
        ]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.get('workload', 'unknown')} | "
            f"{_fmt(row.get('ttft_ms_p50'))} | {_fmt(row.get('ttft_ms_p95'))} |"
        )

    lines.extend(
        [
            "",
            "## Throughput Summary",
            "",
            "| workload | requests/sec | output tokens/sec |",
            "|---|---:|---:|",
        ]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.get('workload', 'unknown')} | "
            f"{_fmt(row.get('requests_per_second_mean'))} | "
            f"{_fmt(row.get('output_tokens_per_second_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Quality Summary",
            "",
            "| workload | mean quality score | success rate |",
            "|---|---:|---:|",
        ]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.get('workload', 'unknown')} | "
            f"{_fmt(row.get('quality_score_mean'))} | {_fmt(row.get('success_rate'))} |"
        )

    lines.extend(["", "## Missing Metrics Warning", ""])
    if missing_values:
        lines.append(
            "The following metrics were unavailable or intentionally null in this run: "
            + ", ".join(f"`{metric}`" for metric in missing_values)
            + "."
        )
    else:
        lines.append("No missing metrics were reported.")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- Use this report to validate the local/mock harness only.",
            "- Do not treat mock metrics as real engine benchmark results.",
            "- Add real endpoint validation in Milestone 2 after Milestone 1 tests pass.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a KVOptBench markdown report.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate_report(input_path=args.input, output_path=args.output)
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()

