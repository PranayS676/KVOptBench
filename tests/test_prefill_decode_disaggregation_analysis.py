import json

import pandas as pd

from kvoptbench.analysis.prefill_decode_disaggregation import (
    compare_prefill_decode_disaggregation_results,
)


def test_compare_disaggregation_results_computes_deltas_and_interpretation(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "disaggregation.csv"
    rows = [
        _row(
            strategy="baseline",
            input_bucket=32768,
            output_bucket=32,
            ttft_ms=900.0,
            tpot_ms=12.0,
            e2e_ms=1220.0,
            output_tps=80.0,
            quality=1.0,
        ),
        _row(
            strategy="prefill_decode_disaggregation",
            input_bucket=32768,
            output_bucket=32,
            ttft_ms=700.0,
            tpot_ms=12.5,
            e2e_ms=1040.0,
            output_tps=82.0,
            quality=1.0,
        ),
        _row(
            strategy="baseline",
            input_bucket=4096,
            output_bucket=512,
            ttft_ms=180.0,
            tpot_ms=10.0,
            e2e_ms=5200.0,
            output_tps=95.0,
            quality=1.0,
        ),
        _row(
            strategy="prefill_decode_disaggregation",
            input_bucket=4096,
            output_bucket=512,
            ttft_ms=180.0,
            tpot_ms=13.0,
            e2e_ms=6600.0,
            output_tps=75.0,
            quality=1.0,
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_prefill_decode_disaggregation_results(
        input_path=raw_dir,
        output_path=output,
    )

    assert written == output
    frame = pd.read_csv(output)
    by_bucket = {
        (row["input_token_bucket"], row["output_token_bucket"]): row
        for _, row in frame.iterrows()
    }
    assert by_bucket[(32768, 32)]["ttft_delta_pct"] == -22.222
    assert by_bucket[(32768, 32)]["tpot_delta_pct"] == 4.167
    assert by_bucket[(32768, 32)]["disaggregation_interpretation"] == (
        "disaggregation_promising"
    )
    assert "baseline_ttft_ms_stats_status" in frame.columns
    assert "ttft_ms_effect_size" in frame.columns
    assert by_bucket[(4096, 512)]["disaggregation_interpretation"] == "decode_regression"


def test_compare_disaggregation_results_preserves_missing_backend_metrics(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "disaggregation.csv"
    rows = [
        _row(
            strategy="baseline",
            input_bucket=32768,
            output_bucket=32,
            ttft_ms=900.0,
            tpot_ms=12.0,
            e2e_ms=1220.0,
            output_tps=80.0,
            quality=1.0,
            missing_metrics=["prefill_decode_split_ms"],
        ),
        _row(
            strategy="prefill_decode_disaggregation",
            input_bucket=32768,
            output_bucket=32,
            ttft_ms=895.0,
            tpot_ms=12.0,
            e2e_ms=1215.0,
            output_tps=80.0,
            quality=1.0,
            missing_metrics=["prefill_decode_split_ms"],
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    compare_prefill_decode_disaggregation_results(input_path=raw_dir, output_path=output)

    row = pd.read_csv(output).iloc[0]
    assert row["missing_metrics"] == "prefill_decode_split_ms"
    assert row["disaggregation_interpretation"] == "no_observed_benefit"


def _row(
    *,
    strategy: str,
    input_bucket: int,
    output_bucket: int,
    ttft_ms: float,
    tpot_ms: float,
    e2e_ms: float,
    output_tps: float,
    quality: float,
    missing_metrics: list[str] | None = None,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": f"disagg_vllm_{strategy}_prefill_decode_grid",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": strategy,
        "workload": "prefill_decode_grid",
        "task_id": f"{strategy}_{input_bucket}_{output_bucket}",
        "concurrency": 1,
        "input_tokens": input_bucket,
        "output_tokens": output_bucket,
        "target_input_tokens": input_bucket,
        "target_output_tokens": output_bucket,
        "shared_prefix_tokens": 0,
        "cache_state": "na",
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itl_ms": tpot_ms,
        "e2e_latency_ms": e2e_ms,
        "output_tokens_per_second": output_tps,
        "success": True,
        "quality_score": quality,
        "missing_metrics": missing_metrics or [],
        "metadata": {
            "config_metadata": {
                "prefill_decode_disaggregation_experiment": True,
                "workload_profile": "prefill_decode_grid",
            },
            "workload_metadata": {
                "input_token_bucket": input_bucket,
                "output_token_bucket": output_bucket,
                "expected_bottleneck": "prefill_bound",
            },
        },
    }
