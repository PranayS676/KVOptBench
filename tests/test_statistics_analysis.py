import json
from pathlib import Path

import pandas as pd

from kvoptbench.analysis.statistics import (
    aggregate_repeated_results,
    compare_repeated_results,
    flatten_metric_stats,
    load_results,
    mean_effect_size,
    mean_effect_size_from_stats,
    percent_delta,
    summarize_metric_values,
)


def test_aggregate_repeated_trials_groups_metrics_by_workload_and_strategy() -> None:
    frame = pd.DataFrame(
        [
            _result_row("shared_prefix", "baseline", 100.0, 10.0),
            _result_row("shared_prefix", "baseline", 110.0, 12.0),
            _result_row("shared_prefix", "baseline", 120.0, 14.0),
            _result_row("shared_prefix", "candidate", 80.0, 8.0),
            _result_row("random_prefix", "baseline", 200.0, 20.0),
        ]
    )

    summary = aggregate_repeated_results(
        frame,
        group_columns=["workload", "strategy"],
        metric_columns=["ttft_ms", "tpot_ms"],
    )

    assert list(summary[["workload", "strategy"]].itertuples(index=False, name=None)) == [
        ("random_prefix", "baseline"),
        ("shared_prefix", "baseline"),
        ("shared_prefix", "candidate"),
    ]

    shared_baseline = summary[
        (summary["workload"] == "shared_prefix") & (summary["strategy"] == "baseline")
    ].iloc[0]
    assert shared_baseline["ttft_ms_count"] == 3
    assert shared_baseline["ttft_ms_mean"] == 110.0
    assert shared_baseline["ttft_ms_p50"] == 110.0
    assert shared_baseline["ttft_ms_p95"] == 119.0
    assert shared_baseline["ttft_ms_min"] == 100.0
    assert shared_baseline["ttft_ms_max"] == 120.0
    assert shared_baseline["tpot_ms_mean"] == 12.0


def test_summarize_metric_values_keeps_nulls_and_insufficient_ci_as_none() -> None:
    summary = summarize_metric_values([None, "", float("nan"), "5.5"])

    assert summary == {
        "count": 1,
        "mean": 5.5,
        "p50": 5.5,
        "p95": 5.5,
        "std": None,
        "min": 5.5,
        "max": 5.5,
        "ci95_low": None,
        "ci95_high": None,
    }


def test_confidence_interval_uses_normal_approximation_for_repeated_trials() -> None:
    summary = summarize_metric_values([100.0, 110.0, 120.0])

    assert summary["std"] == 10.0
    assert summary["ci95_low"] == 98.684
    assert summary["ci95_high"] == 121.316


def test_flatten_metric_stats_adds_status_columns() -> None:
    stats = flatten_metric_stats("baseline_ttft_ms", [100.0, 110.0, 120.0])

    assert stats["baseline_ttft_ms_count"] == 3
    assert stats["baseline_ttft_ms_ci95_low"] == 98.684
    assert stats["baseline_ttft_ms_stats_status"] == "ok"
    assert flatten_metric_stats("metric", [1.0])["metric_stats_status"] == (
        "insufficient_repetitions"
    )


def test_mean_effect_size_requires_repeated_baseline_and_candidate_values() -> None:
    assert mean_effect_size([100.0, 110.0, 120.0], [80.0, 90.0, 100.0]) == -2.0
    assert mean_effect_size([100.0], [80.0, 90.0]) is None
    assert mean_effect_size_from_stats(
        baseline_mean=110.0,
        baseline_std=10.0,
        baseline_count=3,
        candidate_mean=90.0,
        candidate_std=10.0,
        candidate_count=3,
    ) == -2.0


def test_percent_delta_direction_candidate_relative_to_baseline() -> None:
    assert percent_delta(baseline=100.0, candidate=80.0) == -20.0
    assert percent_delta(baseline=100.0, candidate=125.0) == 25.0
    assert percent_delta(baseline=0.0, candidate=10.0) is None
    assert percent_delta(baseline=None, candidate=10.0) is None


def test_compare_repeated_results_adds_candidate_vs_baseline_delta() -> None:
    frame = pd.DataFrame(
        [
            _result_row("shared_prefix", "baseline", 100.0, 10.0),
            _result_row("shared_prefix", "baseline", 120.0, 12.0),
            _result_row("shared_prefix", "candidate", 80.0, 11.0),
            _result_row("shared_prefix", "candidate", 90.0, None),
            _result_row("random_prefix", "baseline", 200.0, 20.0),
        ]
    )

    comparison = compare_repeated_results(
        frame,
        group_columns=["workload"],
        metric_columns=["ttft_ms", "tpot_ms"],
        baseline_strategy="baseline",
        candidate_strategy="candidate",
    )

    assert list(comparison["workload"]) == ["shared_prefix"]
    row = comparison.iloc[0]
    assert row["baseline_strategy"] == "baseline"
    assert row["candidate_strategy"] == "candidate"
    assert row["ttft_ms_baseline_mean"] == 110.0
    assert row["ttft_ms_candidate_mean"] == 85.0
    assert row["ttft_ms_percent_delta"] == -22.727
    assert row["tpot_ms_baseline_count"] == 2
    assert row["tpot_ms_candidate_count"] == 1
    assert row["tpot_ms_candidate_ci95_low"] is None


def test_load_results_reads_csv_jsonl_file_and_jsonl_directory(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    pd.DataFrame([_result_row("shared_prefix", "baseline", 100.0, 10.0)]).to_csv(
        csv_path,
        index=False,
    )

    jsonl_dir = tmp_path / "raw"
    jsonl_dir.mkdir()
    jsonl_rows = [
        _result_row("shared_prefix", "baseline", 100.0, 10.0),
        _result_row("shared_prefix", "candidate", 80.0, None),
    ]
    (jsonl_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in jsonl_rows),
        encoding="utf-8",
    )

    assert len(load_results(csv_path)) == 1
    assert list(load_results(jsonl_dir)["strategy"]) == ["baseline", "candidate"]


def _result_row(
    workload: str,
    strategy: str,
    ttft_ms: float | None,
    tpot_ms: float | None,
) -> dict[str, object]:
    return {
        "experiment_id": "exp",
        "provider": "mock",
        "engine": "mock",
        "model_id": "model",
        "workload": workload,
        "strategy": strategy,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
    }
