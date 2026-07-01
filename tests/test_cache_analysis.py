import json

import pandas as pd

from kvoptbench.analysis.cache import (
    cache_miss_penalty_ms,
    compare_cache_signal,
    miss_penalty_per_1k_tokens,
    prefix_cache_speedup,
    summarize_cold_warm_ttft,
)
from kvoptbench.analysis.cache_compare import compare_cache_results


def test_cache_metric_helpers_handle_missing_and_zero_values() -> None:
    assert cache_miss_penalty_ms(250.0, 100.0) == 150.0
    assert cache_miss_penalty_ms(None, 100.0) is None
    assert prefix_cache_speedup(250.0, 100.0) == 2.5
    assert prefix_cache_speedup(250.0, 0.0) is None
    assert miss_penalty_per_1k_tokens(150.0, 3000) == 50.0
    assert miss_penalty_per_1k_tokens(None, 3000) is None


def test_summarize_cold_warm_ttft_groups_cache_runs() -> None:
    frame = pd.DataFrame(
        [
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "cache_state": "cold",
                "ttft_ms": 300.0,
                "shared_prefix_tokens": 10000,
            },
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "cache_state": "warm",
                "ttft_ms": 100.0,
                "shared_prefix_tokens": 10000,
            },
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "random_prefix_control",
                "cache_state": "na",
                "ttft_ms": 280.0,
                "shared_prefix_tokens": 0,
            },
        ]
    )

    summary = summarize_cold_warm_ttft(frame)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["engine"] == "vllm"
    assert row["strategy"] == "cache_on"
    assert row["cold_ttft_ms_mean"] == 300.0
    assert row["warm_ttft_ms_mean"] == 100.0
    assert row["cache_miss_penalty_ms"] == 200.0
    assert row["miss_penalty_per_1k_tokens"] == 20.0


def test_compare_cache_signal_interprets_shared_vs_random_controls() -> None:
    frame = pd.DataFrame(
        [
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "cache_state": "cold",
                "ttft_ms": 300.0,
                "shared_prefix_tokens": 10000,
                "metadata": {
                    "config_metadata": {
                        "workload_profile": "shared_prefix",
                        "cache_pass": "cold",
                    }
                },
            },
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "shared_prefix_long_doc",
                "cache_state": "warm",
                "ttft_ms": 100.0,
                "shared_prefix_tokens": 10000,
                "metadata": {
                    "config_metadata": {
                        "workload_profile": "shared_prefix",
                        "cache_pass": "warm",
                    }
                },
            },
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "random_prefix_control",
                "cache_state": "cold",
                "ttft_ms": 290.0,
                "shared_prefix_tokens": 0,
                "metadata": {
                    "config_metadata": {
                        "workload_profile": "random_prefix",
                        "cache_pass": "cold",
                    }
                },
            },
            {
                "engine": "vllm",
                "strategy": "cache_on",
                "workload": "random_prefix_control",
                "cache_state": "warm",
                "ttft_ms": 275.0,
                "shared_prefix_tokens": 0,
                "metadata": {
                    "config_metadata": {
                        "workload_profile": "random_prefix",
                        "cache_pass": "warm",
                    }
                },
            },
        ]
    )

    comparison = compare_cache_signal(frame)

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["shared_cache_miss_penalty_ms"] == 200.0
    assert row["random_cache_miss_penalty_ms"] == 15.0
    assert row["interpretation"] == "credible_cache_reuse_signal"


def test_compare_cache_results_writes_control_adjusted_csv(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "cache_summary.csv"
    rows = [
        _cache_row(
            experiment_id="cache_exp_shared_cold",
            workload="shared_prefix_long_doc",
            cache_state="cold",
            ttft_ms=320.0,
            shared_prefix_tokens=12000,
            workload_profile="shared_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_shared_cold_2",
            workload="shared_prefix_long_doc",
            cache_state="cold",
            ttft_ms=340.0,
            shared_prefix_tokens=12000,
            workload_profile="shared_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_shared_warm",
            workload="shared_prefix_long_doc",
            cache_state="warm",
            ttft_ms=110.0,
            shared_prefix_tokens=12000,
            workload_profile="shared_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_shared_warm_2",
            workload="shared_prefix_long_doc",
            cache_state="warm",
            ttft_ms=130.0,
            shared_prefix_tokens=12000,
            workload_profile="shared_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_random_cold",
            workload="random_prefix_control",
            cache_state="cold",
            ttft_ms=295.0,
            shared_prefix_tokens=0,
            workload_profile="random_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_random_cold_2",
            workload="random_prefix_control",
            cache_state="cold",
            ttft_ms=297.0,
            shared_prefix_tokens=0,
            workload_profile="random_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_random_warm",
            workload="random_prefix_control",
            cache_state="warm",
            ttft_ms=280.0,
            shared_prefix_tokens=0,
            workload_profile="random_prefix",
        ),
        _cache_row(
            experiment_id="cache_exp_random_warm_2",
            workload="random_prefix_control",
            cache_state="warm",
            ttft_ms=282.0,
            shared_prefix_tokens=0,
            workload_profile="random_prefix",
        ),
    ]
    (raw_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    written = compare_cache_results(input_path=raw_dir, output_path=output)

    assert written == output
    frame = pd.read_csv(output)
    assert list(frame["engine"]) == ["vllm"]
    row = frame.iloc[0]
    assert row["strategy"] == "cache_on"
    assert row["shared_cold_ttft_ms"] == 330.0
    assert row["shared_warm_ttft_ms"] == 120.0
    assert row["random_cold_ttft_ms"] == 296.0
    assert row["random_warm_ttft_ms"] == 281.0
    assert row["shared_cold_ttft_ms_count"] == 2
    assert row["shared_cold_ttft_ms_stats_status"] == "ok"
    assert row["shared_cold_ttft_ms_ci95_low"] < row["shared_cold_ttft_ms"]
    assert row["shared_cold_ttft_ms_ci95_high"] > row["shared_cold_ttft_ms"]
    assert row["shared_cache_miss_penalty_ms"] == 210.0
    assert row["random_cache_miss_penalty_ms"] == 15.0
    assert row["control_adjusted_cache_gain_ms"] == 195.0
    assert row["shared_prefix_tokens"] == 12000
    assert row["miss_penalty_per_1k_tokens"] == 17.5
    assert row["interpretation"] == "credible_cache_reuse_signal"


def _cache_row(
    *,
    experiment_id: str,
    workload: str,
    cache_state: str,
    ttft_ms: float,
    shared_prefix_tokens: int,
    workload_profile: str,
) -> dict:
    return {
        "run_id": "run",
        "experiment_id": experiment_id,
        "provider": "mock",
        "engine": "vllm",
        "model_id": "mock-frontier-model",
        "strategy": "cache_on",
        "workload": workload,
        "task_id": experiment_id,
        "concurrency": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "target_input_tokens": 100,
        "target_output_tokens": 10,
        "shared_prefix_tokens": shared_prefix_tokens,
        "cache_state": cache_state,
        "cache_hit_rate": None,
        "cache_hit_proxy": None,
        "cache_miss_penalty_ms": None,
        "ttft_ms": ttft_ms,
        "tpot_ms": 10.0,
        "itl_ms": 10.0,
        "e2e_latency_ms": ttft_ms + 100.0,
        "success": True,
        "quality_score": 1.0,
        "quality_method": "contains_expected",
        "missing_metrics": ["cache_hit_rate", "cache_miss_penalty_ms"],
        "metadata": {
            "config_metadata": {
                "cache_experiment": True,
                "workload_profile": workload_profile,
            }
        },
    }

