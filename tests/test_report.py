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
                "e2e_latency_ms_p50": 200.0,
                "e2e_latency_ms_p95": 240.0,
                "output_tokens_per_second_mean": 20.0,
                "quality_score_mean": 1.0,
                "cache_hit_rate_mean": 0.9,
                "cache_miss_penalty_ms_mean": 250.0,
                "cache_interpretation": "credible_cache_reuse_signal",
                "missing_metrics": "gpu_memory_used_gb;gpu_memory_peak_gb",
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
    assert "## Throughput Summary" in report
    assert "## Quality Summary" in report
    assert "## Cache Summary" in report
    assert "cache miss penalty ms" in report
    assert "## Cache Interpretation" in report
    assert "credible_cache_reuse_signal" in report
    assert "## Missing Metrics Warning" in report
    assert "## Next Steps" in report
    assert "Milestone" not in report


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

