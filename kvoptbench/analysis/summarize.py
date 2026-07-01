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
        metric_provenance = _collect_metric_provenance(group)
        summary["metric_provenance"] = json.dumps(metric_provenance, sort_keys=True)
        summary["metric_source_types"] = _render_metric_sources(metric_provenance)
        summary["unavailable_metric_reasons"] = _render_unavailable_reasons(
            metric_provenance
        )
        summaries.append(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_path, index=False)
    return output_path


def _collect_metric_provenance(group: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    collected: dict[str, dict[str, set[str]]] = {}
    for value in group.get("metric_provenance", []):
        if not isinstance(value, dict):
            continue
        for metric, details in value.items():
            if not isinstance(details, dict):
                continue
            entry = collected.setdefault(
                str(metric),
                {
                    "source_types": set(),
                    "measurement_methods": set(),
                    "unavailable_reasons": set(),
                },
            )
            if details.get("source_type"):
                entry["source_types"].add(str(details["source_type"]))
            if details.get("measurement_method"):
                entry["measurement_methods"].add(str(details["measurement_method"]))
            if details.get("available") is False and details.get("missing_reason"):
                entry["unavailable_reasons"].add(str(details["missing_reason"]))

    return {
        metric: {
            "source_types": sorted(values["source_types"]),
            "measurement_methods": sorted(values["measurement_methods"]),
            "unavailable_reasons": sorted(values["unavailable_reasons"]),
        }
        for metric, values in sorted(collected.items())
    }


def _render_metric_sources(metric_provenance: dict[str, dict[str, list[str]]]) -> str:
    return ";".join(
        f"{metric}:{','.join(details['source_types'])}"
        for metric, details in metric_provenance.items()
        if details.get("source_types")
    )


def _render_unavailable_reasons(metric_provenance: dict[str, dict[str, list[str]]]) -> str:
    return ";".join(
        f"{metric}:{' | '.join(details['unavailable_reasons'])}"
        for metric, details in metric_provenance.items()
        if details.get("unavailable_reasons")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KVOptBench JSONL results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize_results(input_path=args.input, output_path=args.output)
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()

