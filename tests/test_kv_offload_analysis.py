import json

import pandas as pd

from kvoptbench.analysis.kv_offload import compare_kv_offload_results


def test_compare_kv_offload_results_computes_deltas_and_interpretation(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "kv_offload.csv"
    rows = [
        _row(
            strategy="baseline",
            context_bucket=32768,
            ttft_ms=500.0,
            e2e_ms=1000.0,
            output_tps=60.0,
            quality=1.0,
            gpu_peak=24.0,
        ),
        _row(
            strategy="kv_offload",
            context_bucket=32768,
            ttft_ms=520.0,
            e2e_ms=1050.0,
            output_tps=61.0,
            quality=0.99,
            gpu_peak=14.0,
        ),
        _row(
            strategy="baseline",
            context_bucket=65536,
            ttft_ms=700.0,
            e2e_ms=1200.0,
            output_tps=40.0,
            quality=1.0,
            gpu_peak=30.0,
        ),
        _row(
            strategy="kv_offload",
            context_bucket=65536,
            ttft_ms=710.0,
            e2e_ms=1800.0,
            output_tps=30.0,
            quality=1.0,
            gpu_peak=20.0,
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_kv_offload_results(input_path=raw_dir, output_path=output)

    assert written == output
    frame = pd.read_csv(output)
    by_bucket = {row["context_token_bucket"]: row for _, row in frame.iterrows()}
    assert by_bucket[32768]["memory_delta_pct"] == -41.667
    assert by_bucket[32768]["quality_delta"] == -0.01
    assert by_bucket[32768]["offload_interpretation"] == "offload_promising"
    assert "baseline_ttft_ms_stats_status" in frame.columns
    assert "ttft_ms_effect_size" in frame.columns
    assert by_bucket[65536]["offload_interpretation"] == "latency_regression"


def test_compare_kv_offload_results_flags_missing_memory_telemetry(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "kv_offload.csv"
    rows = [
        _row(
            strategy="baseline",
            context_bucket=32768,
            ttft_ms=500.0,
            e2e_ms=1000.0,
            output_tps=60.0,
            quality=1.0,
            gpu_peak=None,
            missing_metrics=["gpu_memory_peak_gb"],
        ),
        _row(
            strategy="kv_offload",
            context_bucket=32768,
            ttft_ms=520.0,
            e2e_ms=1010.0,
            output_tps=61.0,
            quality=1.0,
            gpu_peak=None,
            missing_metrics=["gpu_memory_peak_gb"],
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    compare_kv_offload_results(input_path=raw_dir, output_path=output)

    row = pd.read_csv(output).iloc[0]
    assert pd.isna(row["baseline_gpu_memory_peak_gb"])
    assert pd.isna(row["offload_gpu_memory_peak_gb"])
    assert pd.isna(row["memory_delta_pct"])
    assert row["missing_metrics"] == "gpu_memory_peak_gb"
    assert row["offload_interpretation"] == "memory_telemetry_missing"


def _row(
    *,
    strategy: str,
    context_bucket: int,
    ttft_ms: float,
    e2e_ms: float,
    output_tps: float,
    quality: float,
    gpu_peak: float | None,
    missing_metrics: list[str] | None = None,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": f"kv_offload_vllm_{strategy}_long_context_pressure",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": strategy,
        "workload": "long_context_pressure",
        "task_id": f"{strategy}_{context_bucket}",
        "concurrency": 1,
        "input_tokens": context_bucket,
        "output_tokens": 32,
        "target_input_tokens": context_bucket,
        "target_output_tokens": 32,
        "shared_prefix_tokens": 0,
        "cache_state": "na",
        "ttft_ms": ttft_ms,
        "tpot_ms": 10.0,
        "itl_ms": 10.0,
        "e2e_latency_ms": e2e_ms,
        "output_tokens_per_second": output_tps,
        "gpu_memory_peak_gb": gpu_peak,
        "success": True,
        "quality_score": quality,
        "missing_metrics": missing_metrics or [],
        "metadata": {
            "config_metadata": {
                "kv_offload_experiment": True,
                "workload_profile": "long_context_pressure",
            },
            "workload_metadata": {
                "context_token_bucket": context_bucket,
                "pressure_level": "high",
                "expected_pressure": "prefill_latency_growth",
            },
        },
    }
