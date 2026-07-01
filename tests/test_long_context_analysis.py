import json

import pandas as pd

from kvoptbench.analysis.long_context import compare_long_context_results


def test_compare_long_context_results_classifies_pressure(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "long_context.csv"
    rows = [
        _row(
            task_id="stable",
            context_bucket=4096,
            pressure_level="baseline",
            expected_pressure="stable",
            ttft_ms=100.0,
            e2e_ms=180.0,
            input_tps=22000.0,
            output_tps=80.0,
        ),
        _row(
            task_id="prefill",
            context_bucket=65536,
            pressure_level="extreme",
            expected_pressure="prefill_latency_growth",
            ttft_ms=750.0,
            e2e_ms=900.0,
            input_tps=18000.0,
            output_tps=76.0,
        ),
        _row(
            task_id="throughput",
            context_bucket=131072,
            pressure_level="frontier",
            expected_pressure="throughput_degradation",
            ttft_ms=130.0,
            e2e_ms=600.0,
            input_tps=6000.0,
            output_tps=24.0,
        ),
        _row(
            task_id="failure",
            context_bucket=262144,
            pressure_level="frontier",
            expected_pressure="memory_pressure_candidate",
            ttft_ms=None,
            e2e_ms=None,
            input_tps=None,
            output_tps=None,
            success=False,
            missing_metrics=["ttft_ms", "e2e_latency_ms"],
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_long_context_results(input_path=raw_dir, output_path=output)

    assert written == output
    frame = pd.read_csv(output)
    by_bucket = {row["context_token_bucket"]: row for _, row in frame.iterrows()}
    assert by_bucket[4096]["pressure_classification"] == "stable_long_context"
    assert by_bucket[65536]["pressure_classification"] == "prefill_latency_growth"
    assert by_bucket[131072]["pressure_classification"] == "throughput_degradation"
    assert by_bucket[262144]["pressure_classification"] == "failure_pressure"
    assert by_bucket[65536]["ttft_ms_p50"] == 750.0
    assert by_bucket[131072]["output_tokens_per_second_mean"] == 24.0
    assert "ttft_ms_stats_status" in frame.columns
    assert "ttft_ms_effect_size_vs_baseline" in frame.columns


def test_compare_long_context_results_keeps_unavailable_metrics_null(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "long_context.csv"
    rows = [
        _row(
            task_id="missing",
            context_bucket=32768,
            pressure_level="high",
            expected_pressure="prefill_latency_growth",
            ttft_ms=None,
            e2e_ms=None,
            input_tps=None,
            output_tps=None,
            missing_metrics=["ttft_ms", "e2e_latency_ms", "input_tokens_per_second"],
        )
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    compare_long_context_results(input_path=raw_dir, output_path=output)

    row = pd.read_csv(output).iloc[0]
    assert pd.isna(row["ttft_ms_p50"])
    assert pd.isna(row["input_tokens_per_second_mean"])
    assert row["pressure_classification"] == "insufficient_long_context_signal"
    assert row["missing_metrics"] == "e2e_latency_ms;input_tokens_per_second;ttft_ms"


def _row(
    *,
    task_id: str,
    context_bucket: int,
    pressure_level: str,
    expected_pressure: str,
    ttft_ms: float | None,
    e2e_ms: float | None,
    input_tps: float | None,
    output_tps: float | None,
    success: bool = True,
    missing_metrics: list[str] | None = None,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": "long_context_vllm_baseline_pressure",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": "baseline",
        "workload": "long_context_pressure",
        "task_id": task_id,
        "concurrency": 1,
        "input_tokens": context_bucket,
        "output_tokens": 32,
        "target_input_tokens": context_bucket,
        "target_output_tokens": 32,
        "shared_prefix_tokens": 0,
        "cache_state": "na",
        "ttft_ms": ttft_ms,
        "tpot_ms": 12.0 if success else None,
        "itl_ms": 11.0 if success else None,
        "e2e_latency_ms": e2e_ms,
        "input_tokens_per_second": input_tps,
        "output_tokens_per_second": output_tps,
        "success": success,
        "missing_metrics": missing_metrics or [],
        "metadata": {
            "config_metadata": {
                "long_context_experiment": True,
                "workload_profile": "long_context_pressure",
            },
            "workload_metadata": {
                "context_token_bucket": context_bucket,
                "pressure_level": pressure_level,
                "expected_pressure": expected_pressure,
            },
        },
    }
