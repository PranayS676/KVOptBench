import json
from pathlib import Path

import pandas as pd

from kvoptbench.strategy.advisor import build_strategy_advisor_report
from kvoptbench.strategy.report import render_strategy_advisor_markdown


def test_strategy_advisor_recommends_prefix_caching_with_threshold_evidence(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    cache = tmp_path / "cache_summary.csv"
    prefix_sweep = tmp_path / "prefix_sweep.csv"
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_cold_ttft_ms": 320.0,
                "shared_warm_ttft_ms": 110.0,
                "random_cache_miss_penalty_ms": 15.0,
                "control_adjusted_cache_gain_ms": 195.0,
                "shared_prefix_tokens": 12000,
                "interpretation": "credible_cache_reuse_signal",
            }
        ]
    ).to_csv(cache, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_prefix_ratio": 0.5,
                "cache_gain_ms": 100.0,
                "interpretation": "meaningful_prefix_cache_gain",
                "success_rate": 1.0,
            }
        ]
    ).to_csv(prefix_sweep, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        cache_input_path=cache,
        prefix_sweep_input_path=prefix_sweep,
    )

    recommendation = _by_strategy(report, "prefix_caching")
    assert recommendation.decision == "recommend"
    assert recommendation.confidence == "low"
    assert "mock_source" in recommendation.reason_codes
    assert "no_randomization" in recommendation.reason_codes
    assert recommendation.rank == 1
    assert recommendation.workload_profile == "long_context_qa"
    assert recommendation.quality_gate_status in {"warn", "unknown"}
    assert any("control-adjusted cache gain" in item.message for item in recommendation.evidence)
    assert any("shared-prefix ratio" in item.message for item in recommendation.evidence)


def test_strategy_advisor_flags_kv_offload_missing_memory_as_inconclusive(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    kv_offload = tmp_path / "kv_offload.csv"
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
                "memory_delta_pct": None,
                "e2e_delta_pct": 1.0,
                "ttft_delta_pct": 1.0,
                "throughput_delta_pct": 0.5,
                "quality_delta": 0.0,
                "missing_metrics": "gpu_memory_peak_gb",
                "offload_success_rate": 1.0,
                "offload_interpretation": "memory_telemetry_missing",
            }
        ]
    ).to_csv(kv_offload, index=False)

    report = build_strategy_advisor_report(summary_path=summary, kv_offload_input_path=kv_offload)

    recommendation = _by_strategy(report, "kv_offload")
    assert recommendation.decision == "inconclusive"
    assert recommendation.confidence == "low"
    assert any("memory telemetry" in caveat for caveat in recommendation.caveats)
    assert any("GPU memory" in item for item in recommendation.next_experiments)
    assert recommendation.next_experiment_plans
    assert recommendation.next_experiment_plans[0].command.startswith(
        "kvoptbench kv-offload-plan"
    )
    assert "<workload_file>" in recommendation.next_experiment_plans[0].command
    assert recommendation.next_experiment_plans[0].options["workload_profile"] == (
        "long_context_qa"
    )


def test_strategy_advisor_recommends_speculative_decoding_when_decode_metrics_improve(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)
    speculative = tmp_path / "speculative_decoding.csv"
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
                "e2e_delta_pct": -24.0,
                "throughput_delta_pct": 33.333,
                "quality_delta": -0.01,
                "speculative_success_rate": 1.0,
                "missing_metrics": "speculative_acceptance_rate",
                "speculative_decoding_interpretation": "speculative_decoding_promising",
            }
        ]
    ).to_csv(speculative, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        spec_decoding_input_path=speculative,
    )

    recommendation = _by_strategy(report, "speculative_decoding")
    assert recommendation.decision == "recommend"
    assert recommendation.confidence == "low"
    assert recommendation.workload_profile == "decode_heavy"
    assert any("throughput improved" in item.message for item in recommendation.evidence)
    assert any("acceptance" in caveat for caveat in recommendation.caveats)


def test_strategy_advisor_downgrades_confidence_for_missing_required_telemetry(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=12)
    kv_quant = tmp_path / "kv_quantization.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "workload": "long_context_pressure",
                "context_token_bucket": 32768,
                "baseline_strategy": "baseline",
                "quantized_strategy": "kv_fp8",
                "e2e_delta_pct": -8.0,
                "throughput_delta_pct": 12.0,
                "quality_delta": 0.0,
                "memory_delta_pct": None,
                "missing_metrics": "gpu_memory_peak_gb",
                "requests": 12,
                "quantized_success_rate": 1.0,
                "quantization_interpretation": "quantization_promising",
            }
        ]
    ).to_csv(kv_quant, index=False)

    report = build_strategy_advisor_report(summary_path=summary, kv_quant_input_path=kv_quant)

    recommendation = _by_strategy(report, "kv_quantization")
    assert recommendation.decision == "recommend"
    assert recommendation.confidence == "low"
    assert recommendation.confidence_score < 0.8
    assert any("missing telemetry" in reason for reason in recommendation.confidence_reasons)
    assert any("GPU memory telemetry" in item for item in recommendation.next_experiment_priority)
    assert "timeout_rate" in recommendation.missing_required_metrics


def test_strategy_advisor_downgrades_confidence_for_tiny_sample_support(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=2)
    kv_quant = tmp_path / "kv_quantization.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "workload": "long_context_pressure",
                "context_token_bucket": 32768,
                "baseline_strategy": "baseline",
                "quantized_strategy": "kv_fp8",
                "e2e_delta_pct": -8.0,
                "throughput_delta_pct": 12.0,
                "quality_delta": 0.0,
                "memory_delta_pct": -20.0,
                "missing_metrics": "",
                "requests": 2,
                "quantized_success_rate": 1.0,
                "quantization_interpretation": "quantization_promising",
            }
        ]
    ).to_csv(kv_quant, index=False)

    report = build_strategy_advisor_report(summary_path=summary, kv_quant_input_path=kv_quant)

    recommendation = _by_strategy(report, "kv_quantization")
    assert recommendation.decision == "recommend"
    assert recommendation.confidence == "low"
    assert recommendation.confidence_score < 0.8
    assert "tiny_sample_support" in recommendation.reason_codes
    assert any("sample support" in reason for reason in recommendation.confidence_reasons)
    assert any("Repeat trials" in item for item in recommendation.next_experiment_priority)


def test_strategy_advisor_quality_guardrail_overrides_promising_interpretation(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=12)
    speculative = tmp_path / "speculative_decoding.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "workload": "decode_heavy",
                "output_token_bucket": 256,
                "baseline_strategy": "baseline",
                "speculative_strategy": "speculative_decoding",
                "e2e_delta_pct": -24.0,
                "throughput_delta_pct": 33.333,
                "quality_delta": -0.08,
                "speculative_success_rate": 1.0,
                "missing_metrics": "",
                "requests": 12,
                "speculative_decoding_interpretation": "speculative_decoding_promising",
            }
        ]
    ).to_csv(speculative, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        spec_decoding_input_path=speculative,
    )

    recommendation = _by_strategy(report, "speculative_decoding")
    assert recommendation.decision == "do_not_recommend"
    assert recommendation.quality_guardrail == "failed"
    assert "quality_guardrail_failed" in recommendation.reason_codes
    assert any("quality guardrail" in reason for reason in recommendation.confidence_reasons)
    assert any("quality-sensitive" in item for item in recommendation.next_experiment_priority)


def test_strategy_advisor_marks_mock_source_without_real_engine_claims(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path, provider="mock", source_type="mock", requests=8)
    cache = tmp_path / "cache_summary.csv"
    prefix_sweep = tmp_path / "prefix_sweep.csv"
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "source_type": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_cold_ttft_ms": 320.0,
                "shared_warm_ttft_ms": 110.0,
                "random_cache_miss_penalty_ms": 15.0,
                "control_adjusted_cache_gain_ms": 195.0,
                "shared_prefix_tokens": 12000,
                "interpretation": "credible_cache_reuse_signal",
            }
        ]
    ).to_csv(cache, index=False)
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "source_type": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "shared_prefix_ratio": 0.5,
                "cache_gain_ms": 100.0,
                "requests": 8,
                "interpretation": "meaningful_prefix_cache_gain",
                "success_rate": 1.0,
            }
        ]
    ).to_csv(prefix_sweep, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        cache_input_path=cache,
        prefix_sweep_input_path=prefix_sweep,
    )

    recommendation = _by_strategy(report, "prefix_caching")
    assert recommendation.decision == "recommend"
    assert "mock_source" in recommendation.reason_codes
    assert any("mock" in caveat and "real engine" in caveat for caveat in recommendation.caveats)
    assert any("mock source" in reason for reason in recommendation.confidence_reasons)


def test_strategy_advisor_rejects_disaggregation_decode_regression(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    disagg = tmp_path / "disaggregation.csv"
    pd.DataFrame(
        [
            {
                "provider": "mock",
                "engine": "sglang",
                "model_id": "model",
                "workload": "prefill_decode_grid",
                "input_token_bucket": 4096,
                "output_token_bucket": 512,
                "baseline_strategy": "baseline",
                "disaggregated_strategy": "prefill_decode_disaggregation",
                "ttft_delta_pct": 0.0,
                "tpot_delta_pct": 30.0,
                "itl_delta_pct": 30.0,
                "e2e_delta_pct": 26.923,
                "throughput_delta_pct": -21.053,
                "quality_delta": 0.0,
                "disaggregated_success_rate": 1.0,
                "missing_metrics": "",
                "disaggregation_interpretation": "decode_regression",
            }
        ]
    ).to_csv(disagg, index=False)

    report = build_strategy_advisor_report(summary_path=summary, disagg_input_path=disagg)

    recommendation = _by_strategy(report, "prefill_decode_disaggregation")
    assert recommendation.decision == "do_not_recommend"
    assert recommendation.confidence == "medium"
    assert any("decode latency regressed" in item.message for item in recommendation.evidence)


def test_strategy_advisor_serializes_json_and_markdown(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    report = build_strategy_advisor_report(summary_path=summary)
    payload = report.model_dump(mode="json")
    markdown = render_strategy_advisor_markdown(report)

    assert json.loads(json.dumps(payload))["overall_recommendation"] == "needs_more_data"
    assert "# Strategy Advisor" in markdown
    assert "Needs More Data" in markdown
    assert "Next experiment command plans:" in markdown


def test_strategy_advisor_workload_gate_keeps_rag_performance_only_win_as_consider(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=40)
    kv_quant = tmp_path / "kv_quantization.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "workload": "rag_faithfulness",
                "context_token_bucket": 4096,
                "baseline_strategy": "baseline",
                "quantized_strategy": "kv_fp8",
                "e2e_delta_pct": -10.0,
                "throughput_delta_pct": 18.0,
                "memory_delta_pct": -20.0,
                "quality_delta": None,
                "missing_metrics": "",
                "requests": 40,
                "repeated_trials": 2,
                "quantized_success_rate": 1.0,
                "quantization_interpretation": "quantization_promising",
            }
        ]
    ).to_csv(kv_quant, index=False)

    report = build_strategy_advisor_report(summary_path=summary, kv_quant_input_path=kv_quant)

    recommendation = _by_strategy(report, "kv_quantization")
    assert recommendation.workload_profile == "rag"
    assert recommendation.decision == "consider"
    assert recommendation.quality_gate_status == "warn"
    assert any("partial coverage" in reason for reason in recommendation.confidence_reasons)


def test_strategy_advisor_inconclusive_recommendations_include_command_plans(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=8)
    speculative = tmp_path / "speculative_decoding.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "workload": "decode_heavy",
                "baseline_strategy": "baseline",
                "speculative_strategy": "speculative_decoding",
                "missing_metrics": "output_tokens_per_second;error_rate",
                "requests": 8,
                "speculative_decoding_interpretation": "no_observed_benefit",
            }
        ]
    ).to_csv(speculative, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        spec_decoding_input_path=speculative,
    )

    recommendation = _by_strategy(report, "speculative_decoding")
    assert recommendation.decision == "inconclusive"
    assert recommendation.next_experiment_plans
    plan = recommendation.next_experiment_plans[0]
    assert plan.reason == "missing_required_metrics"
    assert plan.command.startswith("kvoptbench spec-decoding-plan")
    assert "<workload_file>" in plan.command
    assert plan.options["required_metrics"] == [
        "output_tokens_per_second",
        "latency_ms",
        "error_rate",
        "output_tokens",
    ]
    assert any(item.startswith("Command template: kvoptbench") for item in recommendation.next_experiments)


def test_strategy_advisor_loads_workload_thresholds_from_yaml(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=20)
    config = tmp_path / "advisor_thresholds.yaml"
    config.write_text(
        """
workload_thresholds:
  long_context_qa:
    workload_profile: long_context_qa
    primary_focus: custom long context policy
    minimum_samples: 99
    minimum_repeated_trials: 5
    required_quality_evaluators:
      - answer_correctness
    required_metrics:
      - ttft_ms
    recommended_metrics:
      - gpu_memory_peak_gb
    threshold_posture: custom strict policy
    blocking_quality_regression: true
""",
        encoding="utf-8",
    )
    cache = tmp_path / "cache_summary.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "shared_cold_ttft_ms": 320.0,
                "shared_warm_ttft_ms": 110.0,
                "random_cache_miss_penalty_ms": 15.0,
                "control_adjusted_cache_gain_ms": 195.0,
                "shared_prefix_tokens": 12000,
                "requests": 20,
                "randomized_order": True,
                "quality_method": "qasper_answer",
                "ttft_ms": 100.0,
                "interpretation": "credible_cache_reuse_signal",
            }
        ]
    ).to_csv(cache, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        cache_input_path=cache,
        advisor_config_path=config,
    )

    recommendation = _by_strategy(report, "prefix_caching")
    assert recommendation.workload_threshold is not None
    assert recommendation.workload_threshold.minimum_samples == 99
    assert "threshold_sample_support_below_target" in recommendation.reason_codes
    assert any("99 target" in reason for reason in recommendation.confidence_reasons)


def test_strategy_advisor_markdown_renders_confidence_rationale(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path, provider="local", requests=2)
    speculative = tmp_path / "speculative_decoding.csv"
    pd.DataFrame(
        [
            {
                "provider": "local",
                "engine": "vllm",
                "model_id": "model",
                "workload": "decode_heavy",
                "output_token_bucket": 256,
                "baseline_strategy": "baseline",
                "speculative_strategy": "speculative_decoding",
                "e2e_delta_pct": -24.0,
                "throughput_delta_pct": 33.333,
                "quality_delta": -0.01,
                "speculative_success_rate": 1.0,
                "missing_metrics": "speculative_acceptance_rate",
                "requests": 2,
                "speculative_decoding_interpretation": "speculative_decoding_promising",
            }
        ]
    ).to_csv(speculative, index=False)

    report = build_strategy_advisor_report(
        summary_path=summary,
        spec_decoding_input_path=speculative,
    )
    markdown = render_strategy_advisor_markdown(report)

    assert "Confidence rationale:" in markdown
    assert "confidence score:" in markdown
    assert "missing telemetry" in markdown
    assert "Next experiment priority:" in markdown


def _write_summary(
    tmp_path: Path,
    *,
    provider: str = "mock",
    requests: int = 4,
    missing_metrics: str = "",
    source_type: str | None = None,
    endpoint_type: str | None = None,
) -> Path:
    summary = tmp_path / "summary.csv"
    row = {
        "experiment_id": "exp",
        "provider": provider,
        "engine": "vllm",
        "model_id": "model",
        "strategy": "baseline",
        "workload": "shared_prefix_long_doc",
        "concurrency": 1,
        "requests": requests,
        "success_rate": 1.0,
        "quality_score_mean": 1.0,
        "missing_metrics": missing_metrics,
    }
    if source_type is not None:
        row["source_type"] = source_type
    if endpoint_type is not None:
        row["endpoint_type"] = endpoint_type
    pd.DataFrame([row]).to_csv(summary, index=False)
    return summary


def _by_strategy(report, strategy: str):
    matches = [item for item in report.recommendations if item.strategy == strategy]
    assert matches, f"Missing strategy {strategy}"
    return matches[0]
