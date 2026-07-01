import json

import pandas as pd

from kvoptbench.analysis.prefix_sweep import compare_prefix_sweep_results


def test_compare_prefix_sweep_results_groups_by_shared_prefix_ratio(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "prefix_sweep.csv"
    rows = []
    for ratio, cold_ttft, warm_ttft in [
        (0.0, 100.0, 100.0),
        (0.25, 180.0, 130.0),
        (0.5, 260.0, 160.0),
        (0.75, 340.0, 190.0),
        (0.9, 400.0, 220.0),
        (1.0, 440.0, 240.0),
    ]:
        rows.append(_prefix_row(ratio=ratio, cache_state="cold", ttft_ms=cold_ttft))
        rows.append(_prefix_row(ratio=ratio, cache_state="warm", ttft_ms=warm_ttft))
    (raw_dir / "prefix.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_prefix_sweep_results(input_path=raw_dir, output_path=output)

    assert written == output
    frame = pd.read_csv(output)
    assert frame["shared_prefix_ratio"].tolist() == [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    assert frame["cache_gain_ms"].tolist() == [0.0, 50.0, 100.0, 150.0, 180.0, 200.0]
    assert frame.iloc[0]["interpretation"] == "no_prefix_overlap"
    assert set(frame.iloc[1:]["interpretation"]) == {"meaningful_prefix_cache_gain"}
    assert frame.iloc[-1]["miss_penalty_per_1k_tokens"] == 200.0
    assert "cold_ttft_ms_stats_status" in frame.columns
    assert "ttft_ms_effect_size" in frame.columns


def test_compare_prefix_sweep_results_keeps_missing_metrics_null(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "prefix_sweep.csv"
    rows = [
        _prefix_row(ratio=0.5, cache_state="cold", ttft_ms=260.0),
        _prefix_row(ratio=0.5, cache_state="warm", ttft_ms=None),
    ]
    (raw_dir / "prefix.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    compare_prefix_sweep_results(input_path=raw_dir, output_path=output)

    row = pd.read_csv(output).iloc[0]
    assert pd.isna(row["warm_ttft_ms"])
    assert pd.isna(row["cache_gain_ms"])
    assert pd.isna(row["miss_penalty_per_1k_tokens"])
    assert row["interpretation"] == "insufficient_prefix_sweep_signal"


def _prefix_row(*, ratio: float, cache_state: str, ttft_ms: float | None) -> dict:
    shared_prefix_tokens = int(1000 * ratio)
    return {
        "run_id": "run",
        "experiment_id": f"prefix_sweep_{cache_state}",
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": "cache_on",
        "workload": "partial_prefix_reuse",
        "task_id": f"partial_prefix_{int(ratio * 100):03d}_{cache_state}",
        "concurrency": 1,
        "input_tokens": 1000,
        "output_tokens": 32,
        "target_input_tokens": 1000,
        "target_output_tokens": 32,
        "shared_prefix_tokens": shared_prefix_tokens,
        "cache_state": cache_state,
        "ttft_ms": ttft_ms,
        "tpot_ms": 10.0,
        "itl_ms": 10.0,
        "e2e_latency_ms": None if ttft_ms is None else ttft_ms + 100.0,
        "success": ttft_ms is not None,
        "missing_metrics": [] if ttft_ms is not None else ["ttft_ms"],
        "metadata": {
            "config_metadata": {
                "cache_experiment": True,
                "workload_profile": "shared_prefix",
            },
            "workload_metadata": {
                "shared_prefix_ratio": ratio,
            },
        },
    }
