from pathlib import Path

import pandas as pd

from kvoptbench.reports.generate import generate_report


def test_report_generator_creates_required_sections(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "mock",
                "model_id": "model",
                "strategy": "baseline",
                "workload": "shared_prefix_long_doc",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "ttft_ms_count": 3,
                "ttft_ms_ci95_low": 95.0,
                "ttft_ms_ci95_high": 125.0,
                "ttft_ms_stats_status": "ok",
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "e2e_latency_ms_count": 3,
                "e2e_latency_ms_ci95_low": 190.0,
                "e2e_latency_ms_ci95_high": 250.0,
                "e2e_latency_ms_stats_status": "ok",
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "cache_hit_rate_mean": 0.9,
                "cache_miss_penalty_ms_mean": 250.0,
                "gpu_memory_used_gb_mean": None,
                "gpu_memory_peak_gb_mean": None,
                "cache_interpretation": "credible_cache_reuse_signal",
                "missing_metrics": "gpu_memory_used_gb;gpu_memory_peak_gb",
                "metric_provenance": (
                    '{"gpu_memory_peak_gb":{"source_types":["gpu_reported"],'
                    '"measurement_methods":["GPU telemetry adapter"],'
                    '"unavailable_reasons":["GPU telemetry was not collected for this run."]},'
                    '"ttft_ms":{"source_types":["client_observed"],'
                    '"measurement_methods":["stream timing"],"unavailable_reasons":[]}}'
                ),
            }
        ]
    ).to_csv(summary, index=False)

    generate_report(input_path=summary, output_path=output)

    report = output.read_text(encoding="utf-8")
    assert "# KVOptBench Mock Benchmark Report" in report
    assert "## Run Summary" in report
    assert "## Workload Summary" in report
    assert "## Latency Summary" in report
    assert "## TTFT Summary" in report
    assert "## Repetition Statistics" in report
    assert "95.000 to 125.000" in report
    assert "## Throughput Summary" in report
    assert "## Quality Summary" in report
    assert "## Cache Summary" in report
    assert "cache miss penalty ms" in report
    assert "## Telemetry Summary" in report
    assert "GPU peak GB" in report
    assert "## Metric Provenance" in report
    assert "`ttft_ms`" in report
    assert "`client_observed`" in report
    assert "GPU telemetry was not collected" in report
    assert "## Cache Interpretation" in report
    assert "credible_cache_reuse_signal" in report
    assert "## Missing Metrics Warning" in report
    assert "## Next Steps" in report
    assert "Milestone" not in report


def test_report_generator_includes_reasoning_and_tool_call_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "local",
                "engine": "openai_compatible",
                "model_id": "reasoning-model",
                "strategy": "baseline",
                "workload": "tool_calling",
                "concurrency": 1,
                "requests": 2,
                "success_rate": 1.0,
                "ttft_ms_p50": None,
                "ttft_ms_p95": None,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 0.0,
                "quality_score_mean": 0.5,
                "reasoning_content_present_rate": 1.0,
                "visible_answer_missing_rate": 0.5,
                "reasoning_tokens_mean": 12.0,
                "first_reasoning_token_ms_p50": 80.0,
                "tool_call_count_mean": 1.0,
                "missing_metrics": "ttft_ms",
            }
        ]
    ).to_csv(summary, index=False)

    generate_report(input_path=summary, output_path=output)

    report = output.read_text(encoding="utf-8")
    assert "## Reasoning & Tool Calls" in report
    assert "reasoning output rate" in report
    assert "visible answer missing rate" in report
    assert "tool calls/request" in report


def test_report_generator_includes_optional_cache_comparison(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    cache_summary = tmp_path / "cache_summary.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "concurrency": 1,
                "requests": 4,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_cold_ttft_ms": 320.0,
                "shared_warm_ttft_ms": 110.0,
                "random_cold_ttft_ms": 295.0,
                "random_warm_ttft_ms": 280.0,
                "shared_cache_miss_penalty_ms": 210.0,
                "random_cache_miss_penalty_ms": 15.0,
                "control_adjusted_cache_gain_ms": 195.0,
                "shared_prefix_tokens": 12000,
                "miss_penalty_per_1k_tokens": 17.5,
                "interpretation": "credible_cache_reuse_signal",
            }
        ]
    ).to_csv(cache_summary, index=False)

    generate_report(input_path=summary, output_path=output, cache_input_path=cache_summary)

    report = output.read_text(encoding="utf-8")
    assert "## Cache Comparison" in report
    assert "control-adjusted gain ms" in report
    assert "credible_cache_reuse_signal" in report
    assert "Mock cache timings validate benchmark wiring only" in report


def test_report_generator_includes_optional_prefix_sweep(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    prefix_sweep = tmp_path / "prefix_sweep.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "workload": "partial_prefix_reuse",
                "concurrency": 1,
                "requests": 12,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_prefix_ratio": 0.0,
                "shared_prefix_tokens": 0,
                "cold_ttft_ms": 100.0,
                "warm_ttft_ms": 100.0,
                "cache_gain_ms": 0.0,
                "miss_penalty_per_1k_tokens": None,
                "requests": 2,
                "success_rate": 1.0,
                "interpretation": "no_prefix_overlap",
            },
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_prefix_ratio": 0.5,
                "shared_prefix_tokens": 500,
                "cold_ttft_ms": 260.0,
                "warm_ttft_ms": 160.0,
                "cache_gain_ms": 100.0,
                "miss_penalty_per_1k_tokens": 200.0,
                "requests": 2,
                "success_rate": 1.0,
                "interpretation": "meaningful_prefix_cache_gain",
            },
        ]
    ).to_csv(prefix_sweep, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        prefix_sweep_input_path=prefix_sweep,
    )

    report = output.read_text(encoding="utf-8")
    assert "## Prefix Overlap Sweep" in report
    assert "shared prefix ratio" in report
    assert "First meaningful cache gain appears at shared-prefix ratio `0.500`" in report
    assert "meaningful_prefix_cache_gain" in report


def test_report_generator_includes_optional_prefill_decode(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    prefill_decode = tmp_path / "prefill_decode.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "baseline",
                "workload": "prefill_decode_grid",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "baseline",
                "input_token_bucket": 32768,
                "output_token_bucket": 32,
                "expected_bottleneck": "prefill_bound",
                "ttft_ms_p50": 900.0,
                "ttft_ms_p95": 900.0,
                "tpot_ms_mean": 12.0,
                "itl_ms_mean": 11.0,
                "output_tokens_per_second_mean": 80.0,
                "bottleneck_classification": "prefill_bound",
                "missing_metrics": "",
                "requests": 1,
                "success_rate": 1.0,
            }
        ]
    ).to_csv(prefill_decode, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        prefill_decode_input_path=prefill_decode,
    )

    report = output.read_text(encoding="utf-8")
    assert "## Prefill vs Decode" in report
    assert "input bucket" in report
    assert "prefill_bound" in report
    assert "Mock prefill/decode timings validate benchmark wiring only" in report


def test_report_generator_includes_optional_long_context(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    long_context = tmp_path / "long_context.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "baseline",
                "workload": "long_context_pressure",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "baseline",
                "context_token_bucket": 32768,
                "pressure_level": "high",
                "expected_pressure": "prefill_latency_growth",
                "ttft_ms_p50": 700.0,
                "ttft_ms_p95": 720.0,
                "e2e_latency_ms_p50": 920.0,
                "e2e_latency_ms_p95": 960.0,
                "input_tokens_per_second_mean": 12000.0,
                "output_tokens_per_second_mean": 60.0,
                "pressure_classification": "prefill_latency_growth",
                "missing_metrics": "",
                "requests": 1,
                "success_rate": 1.0,
                "error_rate": 0.0,
            }
        ]
    ).to_csv(long_context, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        long_context_input_path=long_context,
    )

    report = output.read_text(encoding="utf-8")
    assert "## Long Context Pressure" in report
    assert "context bucket" in report
    assert "prefill_latency_growth" in report
    assert "Mock long-context timings validate benchmark wiring only" in report


def test_report_generator_includes_optional_kv_quantization(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    kv_quantization = tmp_path / "kv_quantization.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "kv_fp8",
                "workload": "long_context_pressure",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 0.99,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "workload": "long_context_pressure",
                "context_token_bucket": 32768,
                "baseline_strategy": "baseline",
                "quantized_strategy": "kv_fp8",
                "baseline_ttft_ms_p50": 500.0,
                "quantized_ttft_ms_p50": 520.0,
                "ttft_delta_pct": 4.0,
                "baseline_e2e_latency_ms_p50": 1000.0,
                "quantized_e2e_latency_ms_p50": 930.0,
                "e2e_delta_pct": -7.0,
                "baseline_output_tokens_per_second_mean": 60.0,
                "quantized_output_tokens_per_second_mean": 68.0,
                "throughput_delta_pct": 13.333,
                "baseline_quality_score_mean": 1.0,
                "quantized_quality_score_mean": 0.99,
                "quality_delta": -0.01,
                "baseline_gpu_memory_peak_gb": 20.0,
                "quantized_gpu_memory_peak_gb": 12.0,
                "memory_delta_pct": -40.0,
                "missing_metrics": "",
                "requests": 2,
                "baseline_success_rate": 1.0,
                "quantized_success_rate": 1.0,
                "quantization_interpretation": "quantization_promising",
            }
        ]
    ).to_csv(kv_quantization, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        kv_quant_input_path=kv_quantization,
    )

    report = output.read_text(encoding="utf-8")
    assert "## KV Cache Quantization" in report
    assert "quantization_promising" in report
    assert "memory delta %" in report
    assert "Mock KV quantization timings validate benchmark wiring only" in report


def test_report_generator_includes_optional_kv_offload(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    kv_offload = tmp_path / "kv_offload.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "kv_offload",
                "workload": "long_context_pressure",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 0.99,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "workload": "long_context_pressure",
                "context_token_bucket": 32768,
                "baseline_strategy": "baseline",
                "offload_strategy": "kv_offload",
                "baseline_ttft_ms_p50": 500.0,
                "offload_ttft_ms_p50": 520.0,
                "ttft_delta_pct": 4.0,
                "baseline_e2e_latency_ms_p50": 1000.0,
                "offload_e2e_latency_ms_p50": 1050.0,
                "e2e_delta_pct": 5.0,
                "baseline_output_tokens_per_second_mean": 60.0,
                "offload_output_tokens_per_second_mean": 61.0,
                "throughput_delta_pct": 1.667,
                "baseline_quality_score_mean": 1.0,
                "offload_quality_score_mean": 0.99,
                "quality_delta": -0.01,
                "baseline_gpu_memory_peak_gb": 24.0,
                "offload_gpu_memory_peak_gb": 14.0,
                "memory_delta_pct": -41.667,
                "missing_metrics": "",
                "requests": 2,
                "baseline_success_rate": 1.0,
                "offload_success_rate": 1.0,
                "offload_interpretation": "offload_promising",
            }
        ]
    ).to_csv(kv_offload, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        kv_offload_input_path=kv_offload,
    )

    report = output.read_text(encoding="utf-8")
    assert "## KV Offload" in report
    assert "offload_promising" in report
    assert "memory delta %" in report
    assert "Mock KV offload timings validate benchmark wiring only" in report


def test_report_generator_includes_optional_speculative_decoding(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    speculative_decoding = tmp_path / "speculative_decoding.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "speculative_decoding",
                "workload": "decode_heavy",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 80.0,
                "quality_score_mean": 0.99,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "workload": "decode_heavy",
                "output_token_bucket": 256,
                "baseline_strategy": "baseline",
                "speculative_strategy": "speculative_decoding",
                "baseline_ttft_ms_p50": 120.0,
                "speculative_ttft_ms_p50": 125.0,
                "ttft_delta_pct": 4.167,
                "baseline_e2e_latency_ms_p50": 1000.0,
                "speculative_e2e_latency_ms_p50": 760.0,
                "e2e_delta_pct": -24.0,
                "baseline_output_tokens_per_second_mean": 60.0,
                "speculative_output_tokens_per_second_mean": 80.0,
                "throughput_delta_pct": 33.333,
                "baseline_quality_score_mean": 1.0,
                "speculative_quality_score_mean": 0.99,
                "quality_delta": -0.01,
                "missing_metrics": "speculative_acceptance_rate",
                "requests": 2,
                "baseline_success_rate": 1.0,
                "speculative_success_rate": 1.0,
                "speculative_decoding_interpretation": "speculative_decoding_promising",
            }
        ]
    ).to_csv(speculative_decoding, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        spec_decoding_input_path=speculative_decoding,
    )

    report = output.read_text(encoding="utf-8")
    assert "## Speculative Decoding" in report
    assert "speculative_decoding_promising" in report
    assert "throughput delta %" in report
    assert "Mock speculative decoding timings validate benchmark wiring only" in report


def test_report_generator_includes_optional_disaggregation(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    disaggregation = tmp_path / "disaggregation.csv"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "sglang",
                "model_id": "model",
                "strategy": "prefill_decode_disaggregation",
                "workload": "prefill_decode_grid",
                "concurrency": 1,
                "requests": 3,
                "success_rate": 1.0,
                "ttft_ms_p50": 700.0,
                "ttft_ms_p95": 720.0,
                "e2e_latency_ms_p50": 1040.0,
                "e2e_latency_ms_p95": 1080.0,
                "output_tokens_per_second_mean": 80.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "sglang",
                "model_id": "model",
                "workload": "prefill_decode_grid",
                "input_token_bucket": 32768,
                "output_token_bucket": 32,
                "baseline_strategy": "baseline",
                "disaggregated_strategy": "prefill_decode_disaggregation",
                "baseline_ttft_ms_p50": 900.0,
                "disaggregated_ttft_ms_p50": 700.0,
                "ttft_delta_pct": -22.222,
                "baseline_tpot_ms_mean": 12.0,
                "disaggregated_tpot_ms_mean": 12.5,
                "tpot_delta_pct": 4.167,
                "baseline_itl_ms_mean": 12.0,
                "disaggregated_itl_ms_mean": 12.5,
                "itl_delta_pct": 4.167,
                "baseline_e2e_latency_ms_p50": 1220.0,
                "disaggregated_e2e_latency_ms_p50": 1040.0,
                "e2e_delta_pct": -14.754,
                "baseline_output_tokens_per_second_mean": 80.0,
                "disaggregated_output_tokens_per_second_mean": 82.0,
                "throughput_delta_pct": 2.5,
                "baseline_quality_score_mean": 1.0,
                "disaggregated_quality_score_mean": 1.0,
                "quality_delta": 0.0,
                "missing_metrics": "",
                "requests": 2,
                "baseline_success_rate": 1.0,
                "disaggregated_success_rate": 1.0,
                "disaggregation_interpretation": "disaggregation_promising",
            }
        ]
    ).to_csv(disaggregation, index=False)

    generate_report(
        input_path=summary,
        output_path=output,
        disagg_input_path=disaggregation,
    )

    report = output.read_text(encoding="utf-8")
    assert "## Prefill/Decode Disaggregation" in report
    assert "disaggregation_promising" in report
    assert "TPOT delta %" in report
    assert "Mock disaggregation timings validate benchmark wiring only" in report


def test_report_generator_includes_strategy_advisor(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    strategy_advisor = tmp_path / "strategy_advisor.json"
    output = tmp_path / "report.md"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "concurrency": 1,
                "requests": 4,
                "success_rate": 1.0,
                "ttft_ms_p50": 100.0,
                "ttft_ms_p95": 120.0,
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    strategy_advisor.write_text(
        """{
  "generated_at": "2026-06-29T00:00:00+00:00",
  "overall_recommendation": "prefix_caching",
  "recommendations": [
    {
      "strategy": "prefix_caching",
      "decision": "recommend",
      "confidence": "high",
      "rank": 1,
      "score": 5.0,
      "evidence": [
        {
          "message": "Observed control-adjusted cache gain of 195.000 ms.",
          "metric": "control_adjusted_cache_gain_ms",
          "value": 195.0
        }
      ],
      "caveats": [],
      "next_experiments": ["Run prefix-sweep-compare on production-shaped prompts."],
      "source": "cache comparison CSV"
    }
  ]
}
""",
        encoding="utf-8",
    )

    generate_report(
        input_path=summary,
        output_path=output,
        strategy_input_path=strategy_advisor,
    )

    report = output.read_text(encoding="utf-8")
    assert "## Strategy Advisor" in report
    assert "Overall recommendation: `prefix_caching`" in report
    assert "| 1 | `prefix_caching` | `recommend` | `high` |" in report
    assert "Observed control-adjusted cache gain" in report
    assert "Run prefix-sweep-compare" in report
