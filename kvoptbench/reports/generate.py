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


def generate_report(
    input_path: str | Path,
    output_path: str | Path,
    cache_input_path: str | Path | None = None,
    prefix_sweep_input_path: str | Path | None = None,
    prefill_decode_input_path: str | Path | None = None,
    long_context_input_path: str | Path | None = None,
    kv_quant_input_path: str | Path | None = None,
    kv_offload_input_path: str | Path | None = None,
) -> Path:
    """Generate a markdown report from a summary CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError(f"Summary CSV is empty: {input_path}")
    cache_frame = _read_cache_summary(cache_input_path)
    prefix_sweep_frame = _read_prefix_sweep(prefix_sweep_input_path)
    prefill_decode_frame = _read_prefill_decode(prefill_decode_input_path)
    long_context_frame = _read_long_context(long_context_input_path)
    kv_quant_frame = _read_kv_quantization(kv_quant_input_path)
    kv_offload_frame = _read_kv_offload(kv_offload_input_path)

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

    lines.extend(
        [
            "",
            "## Cache Summary",
            "",
            "| workload | strategy | cache hit rate | cache miss penalty ms |",
            "|---|---|---:|---:|",
        ]
    )
    if {"cache_hit_rate_mean", "cache_miss_penalty_ms_mean"}.intersection(frame.columns):
        for _, row in frame.iterrows():
            lines.append(
                f"| {row.get('workload', 'unknown')} | {row.get('strategy', 'unknown')} | "
                f"{_fmt(row.get('cache_hit_rate_mean'))} | "
                f"{_fmt(row.get('cache_miss_penalty_ms_mean'))} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a |")

    _append_cache_comparison(lines, cache_frame)
    _append_prefix_overlap_sweep(lines, prefix_sweep_frame)
    _append_prefill_decode(lines, prefill_decode_frame)
    _append_long_context(lines, long_context_frame)
    _append_kv_quantization(lines, kv_quant_frame)
    _append_kv_offload(lines, kv_offload_frame)

    lines.extend(["", "## Cache Interpretation", ""])
    if "cache_interpretation" in frame.columns:
        interpretations = sorted(
            {
                str(value)
                for value in frame["cache_interpretation"].dropna()
                if str(value).strip()
            }
        )
        if interpretations:
            for interpretation in interpretations:
                lines.append(f"- `{interpretation}`")
        else:
            lines.append("No cache interpretation was available.")
    else:
        lines.append("No cache interpretation was available.")

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
            "- Use this report to validate benchmark wiring and experiment shape.",
            "- Do not treat mock metrics as real engine benchmark results.",
            "- For real endpoint runs, verify engine flags, model revision, workload hash, and missing telemetry before publishing.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _read_cache_summary(cache_input_path: str | Path | None) -> pd.DataFrame | None:
    if cache_input_path is None:
        return None
    cache_path = Path(cache_input_path)
    cache_frame = pd.read_csv(cache_path)
    if cache_frame.empty:
        return pd.DataFrame()
    return cache_frame


def _read_prefix_sweep(prefix_sweep_input_path: str | Path | None) -> pd.DataFrame | None:
    if prefix_sweep_input_path is None:
        return None
    prefix_sweep_path = Path(prefix_sweep_input_path)
    prefix_sweep_frame = pd.read_csv(prefix_sweep_path)
    if prefix_sweep_frame.empty:
        return pd.DataFrame()
    return prefix_sweep_frame


def _read_prefill_decode(prefill_decode_input_path: str | Path | None) -> pd.DataFrame | None:
    if prefill_decode_input_path is None:
        return None
    prefill_decode_path = Path(prefill_decode_input_path)
    prefill_decode_frame = pd.read_csv(prefill_decode_path)
    if prefill_decode_frame.empty:
        return pd.DataFrame()
    return prefill_decode_frame


def _read_long_context(long_context_input_path: str | Path | None) -> pd.DataFrame | None:
    if long_context_input_path is None:
        return None
    long_context_path = Path(long_context_input_path)
    long_context_frame = pd.read_csv(long_context_path)
    if long_context_frame.empty:
        return pd.DataFrame()
    return long_context_frame


def _read_kv_quantization(kv_quant_input_path: str | Path | None) -> pd.DataFrame | None:
    if kv_quant_input_path is None:
        return None
    kv_quant_path = Path(kv_quant_input_path)
    kv_quant_frame = pd.read_csv(kv_quant_path)
    if kv_quant_frame.empty:
        return pd.DataFrame()
    return kv_quant_frame


def _read_kv_offload(kv_offload_input_path: str | Path | None) -> pd.DataFrame | None:
    if kv_offload_input_path is None:
        return None
    kv_offload_path = Path(kv_offload_input_path)
    kv_offload_frame = pd.read_csv(kv_offload_path)
    if kv_offload_frame.empty:
        return pd.DataFrame()
    return kv_offload_frame


def _append_cache_comparison(lines: list[str], cache_frame: pd.DataFrame | None) -> None:
    lines.extend(
        [
            "",
            "## Cache Comparison",
            "",
        ]
    )
    if cache_frame is None:
        lines.append("No cache comparison CSV was provided.")
        return
    if cache_frame.empty:
        lines.append("No cache comparison rows were available.")
        return

    lines.extend(
        [
            "Mock cache timings validate benchmark wiring only; use real endpoint runs for engine claims.",
            "",
            "| engine | strategy | shared cold TTFT ms | shared warm TTFT ms | "
            "random penalty ms | control-adjusted gain ms | interpretation |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in cache_frame.iterrows():
        lines.append(
            f"| {row.get('engine', 'unknown')} | {row.get('strategy', 'unknown')} | "
            f"{_fmt(row.get('shared_cold_ttft_ms'))} | "
            f"{_fmt(row.get('shared_warm_ttft_ms'))} | "
            f"{_fmt(row.get('random_cache_miss_penalty_ms'))} | "
            f"{_fmt(row.get('control_adjusted_cache_gain_ms'))} | "
            f"{row.get('interpretation', 'unknown')} |"
        )


def _append_prefill_decode(lines: list[str], prefill_decode_frame: pd.DataFrame | None) -> None:
    lines.extend(["", "## Prefill vs Decode", ""])
    if prefill_decode_frame is None:
        lines.append("No prefill/decode CSV was provided.")
        return
    if prefill_decode_frame.empty:
        lines.append("No prefill/decode rows were available.")
        return

    classifications = sorted(
        {
            str(value)
            for value in prefill_decode_frame.get(
                "bottleneck_classification", pd.Series(dtype=str)
            ).dropna()
            if str(value).strip()
        }
    )
    if classifications:
        lines.append(
            "Observed bottleneck classifications: "
            + ", ".join(f"`{classification}`" for classification in classifications)
            + "."
        )
    else:
        lines.append("No bottleneck classifications were available.")
    lines.extend(
        [
            "Mock prefill/decode timings validate benchmark wiring only; use real endpoint runs for engine claims.",
            "",
            "| engine | strategy | input bucket | output bucket | expected | "
            "p50 TTFT ms | mean TPOT ms | mean ITL ms | output tok/sec | classification |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in prefill_decode_frame.iterrows():
        lines.append(
            f"| {row.get('engine', 'unknown')} | {row.get('strategy', 'unknown')} | "
            f"{_fmt(row.get('input_token_bucket'))} | "
            f"{_fmt(row.get('output_token_bucket'))} | "
            f"{row.get('expected_bottleneck', 'unknown')} | "
            f"{_fmt(row.get('ttft_ms_p50'))} | "
            f"{_fmt(row.get('tpot_ms_mean'))} | "
            f"{_fmt(row.get('itl_ms_mean'))} | "
            f"{_fmt(row.get('output_tokens_per_second_mean'))} | "
            f"{row.get('bottleneck_classification', 'unknown')} |"
        )


def _append_long_context(lines: list[str], long_context_frame: pd.DataFrame | None) -> None:
    lines.extend(["", "## Long Context Pressure", ""])
    if long_context_frame is None:
        lines.append("No long-context CSV was provided.")
        return
    if long_context_frame.empty:
        lines.append("No long-context rows were available.")
        return

    classifications = sorted(
        {
            str(value)
            for value in long_context_frame.get(
                "pressure_classification", pd.Series(dtype=str)
            ).dropna()
            if str(value).strip()
        }
    )
    if classifications:
        lines.append(
            "Observed pressure classifications: "
            + ", ".join(f"`{classification}`" for classification in classifications)
            + "."
        )
    else:
        lines.append("No pressure classifications were available.")
    lines.extend(
        [
            "Mock long-context timings validate benchmark wiring only; use real endpoint runs for engine claims.",
            "",
            "| engine | strategy | context bucket | pressure level | expected | "
            "p50 TTFT ms | p50 E2E ms | input tok/sec | output tok/sec | "
            "success rate | classification |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in long_context_frame.iterrows():
        lines.append(
            f"| {row.get('engine', 'unknown')} | {row.get('strategy', 'unknown')} | "
            f"{_fmt(row.get('context_token_bucket'))} | "
            f"{row.get('pressure_level', 'unknown')} | "
            f"{row.get('expected_pressure', 'unknown')} | "
            f"{_fmt(row.get('ttft_ms_p50'))} | "
            f"{_fmt(row.get('e2e_latency_ms_p50'))} | "
            f"{_fmt(row.get('input_tokens_per_second_mean'))} | "
            f"{_fmt(row.get('output_tokens_per_second_mean'))} | "
            f"{_fmt(row.get('success_rate'))} | "
            f"{row.get('pressure_classification', 'unknown')} |"
        )


def _append_kv_quantization(lines: list[str], kv_quant_frame: pd.DataFrame | None) -> None:
    lines.extend(["", "## KV Cache Quantization", ""])
    if kv_quant_frame is None:
        lines.append("No KV quantization CSV was provided.")
        return
    if kv_quant_frame.empty:
        lines.append("No KV quantization rows were available.")
        return

    interpretations = sorted(
        {
            str(value)
            for value in kv_quant_frame.get(
                "quantization_interpretation", pd.Series(dtype=str)
            ).dropna()
            if str(value).strip()
        }
    )
    if interpretations:
        lines.append(
            "Observed quantization interpretations: "
            + ", ".join(f"`{interpretation}`" for interpretation in interpretations)
            + "."
        )
    else:
        lines.append("No quantization interpretations were available.")
    lines.extend(
        [
            "Mock KV quantization timings validate benchmark wiring only; use real endpoint runs for engine claims.",
            "",
            "| engine | workload | context bucket | baseline | quantized | "
            "TTFT delta % | E2E delta % | throughput delta % | quality delta | "
            "memory delta % | interpretation |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in kv_quant_frame.iterrows():
        lines.append(
            f"| {row.get('engine', 'unknown')} | {row.get('workload', 'unknown')} | "
            f"{_fmt(row.get('context_token_bucket'))} | "
            f"{row.get('baseline_strategy', 'baseline')} | "
            f"{row.get('quantized_strategy', 'unknown')} | "
            f"{_fmt(row.get('ttft_delta_pct'))} | "
            f"{_fmt(row.get('e2e_delta_pct'))} | "
            f"{_fmt(row.get('throughput_delta_pct'))} | "
            f"{_fmt(row.get('quality_delta'))} | "
            f"{_fmt(row.get('memory_delta_pct'))} | "
            f"{row.get('quantization_interpretation', 'unknown')} |"
        )


def _append_kv_offload(lines: list[str], kv_offload_frame: pd.DataFrame | None) -> None:
    lines.extend(["", "## KV Offload", ""])
    if kv_offload_frame is None:
        lines.append("No KV offload CSV was provided.")
        return
    if kv_offload_frame.empty:
        lines.append("No KV offload rows were available.")
        return

    interpretations = sorted(
        {
            str(value)
            for value in kv_offload_frame.get(
                "offload_interpretation", pd.Series(dtype=str)
            ).dropna()
            if str(value).strip()
        }
    )
    if interpretations:
        lines.append(
            "Observed offload interpretations: "
            + ", ".join(f"`{interpretation}`" for interpretation in interpretations)
            + "."
        )
    else:
        lines.append("No offload interpretations were available.")
    lines.extend(
        [
            "Mock KV offload timings validate benchmark wiring only; use real endpoint runs for engine claims.",
            "",
            "| engine | workload | context bucket | baseline | offload | "
            "TTFT delta % | E2E delta % | throughput delta % | quality delta | "
            "memory delta % | interpretation |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in kv_offload_frame.iterrows():
        lines.append(
            f"| {row.get('engine', 'unknown')} | {row.get('workload', 'unknown')} | "
            f"{_fmt(row.get('context_token_bucket'))} | "
            f"{row.get('baseline_strategy', 'baseline')} | "
            f"{row.get('offload_strategy', 'unknown')} | "
            f"{_fmt(row.get('ttft_delta_pct'))} | "
            f"{_fmt(row.get('e2e_delta_pct'))} | "
            f"{_fmt(row.get('throughput_delta_pct'))} | "
            f"{_fmt(row.get('quality_delta'))} | "
            f"{_fmt(row.get('memory_delta_pct'))} | "
            f"{row.get('offload_interpretation', 'unknown')} |"
        )


def _append_prefix_overlap_sweep(lines: list[str], prefix_sweep_frame: pd.DataFrame | None) -> None:
    lines.extend(["", "## Prefix Overlap Sweep", ""])
    if prefix_sweep_frame is None:
        lines.append("No prefix sweep CSV was provided.")
        return
    if prefix_sweep_frame.empty:
        lines.append("No prefix sweep rows were available.")
        return

    if "interpretation" in prefix_sweep_frame:
        meaningful = prefix_sweep_frame[
            prefix_sweep_frame["interpretation"] == "meaningful_prefix_cache_gain"
        ]
    else:
        meaningful = pd.DataFrame()
    if meaningful.empty:
        lines.append("No meaningful cache gain threshold was observed in this sweep.")
    else:
        first_ratio = float(
            meaningful.sort_values("shared_prefix_ratio").iloc[0]["shared_prefix_ratio"]
        )
        lines.append(
            "First meaningful cache gain appears at shared-prefix ratio "
            f"`{first_ratio:.3f}`."
        )
    lines.extend(
        [
            "Mock prefix-overlap timings validate benchmark wiring only; use real endpoint runs for engine claims.",
            "",
            "| engine | strategy | shared prefix ratio | shared prefix tokens | "
            "cold TTFT ms | warm TTFT ms | cache gain ms | interpretation |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in prefix_sweep_frame.iterrows():
        lines.append(
            f"| {row.get('engine', 'unknown')} | {row.get('strategy', 'unknown')} | "
            f"{_fmt(row.get('shared_prefix_ratio'))} | "
            f"{_fmt(row.get('shared_prefix_tokens'))} | "
            f"{_fmt(row.get('cold_ttft_ms'))} | "
            f"{_fmt(row.get('warm_ttft_ms'))} | "
            f"{_fmt(row.get('cache_gain_ms'))} | "
            f"{row.get('interpretation', 'unknown')} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a KVOptBench markdown report.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-input", type=Path, default=None)
    parser.add_argument("--prefix-sweep-input", type=Path, default=None)
    parser.add_argument("--prefill-decode-input", type=Path, default=None)
    parser.add_argument("--long-context-input", type=Path, default=None)
    parser.add_argument("--kv-quant-input", type=Path, default=None)
    parser.add_argument("--kv-offload-input", type=Path, default=None)
    args = parser.parse_args()
    generate_report(
        input_path=args.input,
        output_path=args.output,
        cache_input_path=args.cache_input,
        prefix_sweep_input_path=args.prefix_sweep_input,
        prefill_decode_input_path=args.prefill_decode_input,
        long_context_input_path=args.long_context_input,
        kv_quant_input_path=args.kv_quant_input,
        kv_offload_input_path=args.kv_offload_input,
    )
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()

