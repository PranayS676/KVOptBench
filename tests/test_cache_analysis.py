import pandas as pd

from kvoptbench.analysis.cache import (
    cache_miss_penalty_ms,
    miss_penalty_per_1k_tokens,
    prefix_cache_speedup,
    summarize_cold_warm_ttft,
)


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

