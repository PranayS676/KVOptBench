import json

import pandas as pd

from kvoptbench.analysis.prefill_decode import compare_prefill_decode_results


def test_compare_prefill_decode_results_classifies_bottlenecks(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "prefill_decode.csv"
    rows = [
        _row(
            task_id="prefill",
            input_bucket=32768,
            output_bucket=32,
            expected_bottleneck="prefill_bound",
            ttft_ms=900.0,
            tpot_ms=12.0,
            itl_ms=11.0,
            output_tps=80.0,
        ),
        _row(
            task_id="decode",
            input_bucket=512,
            output_bucket=512,
            expected_bottleneck="decode_bound",
            ttft_ms=120.0,
            tpot_ms=65.0,
            itl_ms=66.0,
            output_tps=12.0,
        ),
        _row(
            task_id="mixed",
            input_bucket=32768,
            output_bucket=512,
            expected_bottleneck="mixed",
            ttft_ms=950.0,
            tpot_ms=70.0,
            itl_ms=72.0,
            output_tps=10.0,
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_prefill_decode_results(input_path=raw_dir, output_path=output)

    assert written == output
    frame = pd.read_csv(output)
    by_expected = {row["expected_bottleneck"]: row for _, row in frame.iterrows()}
    assert by_expected["prefill_bound"]["bottleneck_classification"] == "prefill_bound"
    assert by_expected["decode_bound"]["bottleneck_classification"] == "decode_bound"
    assert by_expected["mixed"]["bottleneck_classification"] == "mixed"
    assert by_expected["prefill_bound"]["ttft_ms_p50"] == 900.0
    assert by_expected["decode_bound"]["tpot_ms_mean"] == 65.0
    assert "ttft_ms_stats_status" in frame.columns
    assert "ttft_ms_effect_size_vs_baseline" in frame.columns


def test_compare_prefill_decode_results_keeps_unavailable_metrics_null(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "prefill_decode.csv"
    rows = [
        _row(
            task_id="missing",
            input_bucket=2048,
            output_bucket=128,
            expected_bottleneck="mixed",
            ttft_ms=None,
            tpot_ms=None,
            itl_ms=None,
            output_tps=None,
            missing_metrics=["ttft_ms", "tpot_ms", "itl_ms"],
        )
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    compare_prefill_decode_results(input_path=raw_dir, output_path=output)

    row = pd.read_csv(output).iloc[0]
    assert pd.isna(row["ttft_ms_p50"])
    assert pd.isna(row["tpot_ms_mean"])
    assert row["bottleneck_classification"] == "insufficient_prefill_decode_signal"
    assert row["missing_metrics"] == "itl_ms;tpot_ms;ttft_ms"


def _row(
    *,
    task_id: str,
    input_bucket: int,
    output_bucket: int,
    expected_bottleneck: str,
    ttft_ms: float | None,
    tpot_ms: float | None,
    itl_ms: float | None,
    output_tps: float | None,
    missing_metrics: list[str] | None = None,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": "prefill_decode_vllm_baseline_grid",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": "baseline",
        "workload": "prefill_decode_grid",
        "task_id": task_id,
        "concurrency": 1,
        "input_tokens": input_bucket,
        "output_tokens": output_bucket,
        "target_input_tokens": input_bucket,
        "target_output_tokens": output_bucket,
        "shared_prefix_tokens": 0,
        "cache_state": "na",
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itl_ms": itl_ms,
        "e2e_latency_ms": None if ttft_ms is None else ttft_ms + 100.0,
        "output_tokens_per_second": output_tps,
        "success": ttft_ms is not None,
        "missing_metrics": missing_metrics or [],
        "metadata": {
            "config_metadata": {
                "prefill_decode_experiment": True,
                "workload_profile": "prefill_decode_grid",
            },
            "workload_metadata": {
                "input_token_bucket": input_bucket,
                "output_token_bucket": output_bucket,
                "expected_bottleneck": expected_bottleneck,
            },
        },
    }
