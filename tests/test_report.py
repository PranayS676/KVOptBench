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
    assert "## Missing Metrics Warning" in report
    assert "## Next Steps" in report

