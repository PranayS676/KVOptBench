import json

import pandas as pd

from kvoptbench.analysis.speculative_decoding import compare_speculative_decoding_results


def test_compare_speculative_decoding_results_computes_deltas_and_interpretation(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "speculative_decoding.csv"
    rows = [
        _row(
            strategy="baseline",
            output_bucket=256,
            ttft_ms=120.0,
            e2e_ms=1000.0,
            output_tps=60.0,
            quality=1.0,
        ),
        _row(
            strategy="speculative_decoding",
            output_bucket=256,
            ttft_ms=125.0,
            e2e_ms=760.0,
            output_tps=80.0,
            quality=0.99,
        ),
        _row(
            strategy="baseline",
            output_bucket=512,
            ttft_ms=130.0,
            e2e_ms=1800.0,
            output_tps=45.0,
            quality=1.0,
        ),
        _row(
            strategy="speculative_decoding",
            output_bucket=512,
            ttft_ms=135.0,
            e2e_ms=1700.0,
            output_tps=47.0,
            quality=0.80,
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_speculative_decoding_results(input_path=raw_dir, output_path=output)

    assert written == output
    frame = pd.read_csv(output)
    by_bucket = {row["output_token_bucket"]: row for _, row in frame.iterrows()}
    assert by_bucket[256]["e2e_delta_pct"] == -24.0
    assert by_bucket[256]["throughput_delta_pct"] == 33.333
    assert by_bucket[256]["quality_delta"] == -0.01
    assert by_bucket[256]["speculative_decoding_interpretation"] == (
        "speculative_decoding_promising"
    )
    assert "baseline_ttft_ms_stats_status" in frame.columns
    assert "ttft_ms_effect_size" in frame.columns
    assert by_bucket[512]["speculative_decoding_interpretation"] == "quality_regression"


def test_compare_speculative_decoding_results_preserves_missing_backend_metrics(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "speculative_decoding.csv"
    rows = [
        _row(
            strategy="baseline",
            output_bucket=256,
            ttft_ms=120.0,
            e2e_ms=1000.0,
            output_tps=60.0,
            quality=1.0,
            missing_metrics=["speculative_acceptance_rate"],
        ),
        _row(
            strategy="speculative_decoding",
            output_bucket=256,
            ttft_ms=121.0,
            e2e_ms=995.0,
            output_tps=60.5,
            quality=1.0,
            missing_metrics=["speculative_acceptance_rate"],
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    compare_speculative_decoding_results(input_path=raw_dir, output_path=output)

    row = pd.read_csv(output).iloc[0]
    assert row["missing_metrics"] == "speculative_acceptance_rate"
    assert row["speculative_decoding_interpretation"] == "no_observed_benefit"


def _row(
    *,
    strategy: str,
    output_bucket: int,
    ttft_ms: float,
    e2e_ms: float,
    output_tps: float,
    quality: float,
    missing_metrics: list[str] | None = None,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": f"spec_decode_vllm_{strategy}_decode_heavy",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": strategy,
        "workload": "decode_heavy",
        "task_id": f"{strategy}_{output_bucket}",
        "concurrency": 1,
        "input_tokens": 128,
        "output_tokens": output_bucket,
        "target_input_tokens": 128,
        "target_output_tokens": output_bucket,
        "shared_prefix_tokens": 0,
        "cache_state": "na",
        "ttft_ms": ttft_ms,
        "tpot_ms": 8.0,
        "itl_ms": 8.0,
        "e2e_latency_ms": e2e_ms,
        "output_tokens_per_second": output_tps,
        "success": True,
        "quality_score": quality,
        "missing_metrics": missing_metrics or [],
        "metadata": {
            "config_metadata": {
                "speculative_decoding_experiment": True,
                "workload_profile": "decode_heavy",
            },
            "workload_metadata": {
                "output_token_bucket": output_bucket,
            },
        },
    }
