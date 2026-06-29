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
    assert recommendation.confidence == "high"
    assert recommendation.rank == 1
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
    assert recommendation.confidence == "medium"
    assert any("throughput improved" in item.message for item in recommendation.evidence)
    assert any("acceptance" in caveat for caveat in recommendation.caveats)


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
    assert recommendation.confidence == "high"
    assert any("decode latency regressed" in item.message for item in recommendation.evidence)


def test_strategy_advisor_serializes_json_and_markdown(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    report = build_strategy_advisor_report(summary_path=summary)
    payload = report.model_dump(mode="json")
    markdown = render_strategy_advisor_markdown(report)

    assert json.loads(json.dumps(payload))["overall_recommendation"] == "needs_more_data"
    assert "# Strategy Advisor" in markdown
    assert "Needs More Data" in markdown


def _write_summary(tmp_path: Path) -> Path:
    summary = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "provider": "mock",
                "engine": "vllm",
                "model_id": "model",
                "strategy": "baseline",
                "workload": "shared_prefix_long_doc",
                "concurrency": 1,
                "requests": 4,
                "success_rate": 1.0,
                "quality_score_mean": 1.0,
                "missing_metrics": "",
            }
        ]
    ).to_csv(summary, index=False)
    return summary


def _by_strategy(report, strategy: str):
    matches = [item for item in report.recommendations if item.strategy == strategy]
    assert matches, f"Missing strategy {strategy}"
    return matches[0]
