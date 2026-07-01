"""Shared repeated-run columns for strategy comparison CSVs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from kvoptbench.analysis.statistics import (
    comparison_methodology_status,
    flatten_metric_stats,
    mean_effect_size_from_stats,
)


REPEATED_COMPARISON_SUFFIXES = ("count", "std", "ci95_low", "ci95_high", "stats_status")


def add_group_metric_stats(
    row: dict[str, Any],
    group: pd.DataFrame,
    metrics: list[str],
) -> None:
    """Add count/std/CI/status fields for raw metric columns in one aggregate row."""
    for metric in metrics:
        values = group[metric] if metric in group else []
        row.update(flatten_metric_stats(metric, values))


def repeated_comparison_columns(
    metric_specs: list[tuple[str, str, str]],
) -> list[str]:
    """Build output columns for baseline/candidate repeated-stat comparison fields."""
    columns: list[str] = []
    for baseline_prefix, candidate_prefix, effect_prefix in metric_specs:
        for suffix in REPEATED_COMPARISON_SUFFIXES:
            columns.append(f"{baseline_prefix}_{suffix}")
            columns.append(f"{candidate_prefix}_{suffix}")
        columns.append(f"{effect_prefix}_effect_size")
        columns.append(f"{effect_prefix}_methodology_status")
        columns.append(f"{effect_prefix}_methodology_caveats")
    return columns


def repeated_comparison_fields(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    source_metric: str,
    baseline_prefix: str,
    candidate_prefix: str,
    effect_prefix: str,
) -> dict[str, Any]:
    """Build repeated-stat comparison fields from two aggregated strategy rows."""
    fields: dict[str, Any] = {}
    for suffix in REPEATED_COMPARISON_SUFFIXES:
        fields[f"{baseline_prefix}_{suffix}"] = _get(baseline, source_metric, suffix)
        fields[f"{candidate_prefix}_{suffix}"] = _get(candidate, source_metric, suffix)
    fields[f"{effect_prefix}_effect_size"] = mean_effect_size_from_stats(
        baseline_mean=baseline.get(f"{source_metric}_mean"),
        baseline_std=baseline.get(f"{source_metric}_std"),
        baseline_count=baseline.get(f"{source_metric}_count"),
        candidate_mean=candidate.get(f"{source_metric}_mean"),
        candidate_std=candidate.get(f"{source_metric}_std"),
        candidate_count=candidate.get(f"{source_metric}_count"),
    )
    status, caveats = comparison_methodology_status(
        baseline_count=fields.get(f"{baseline_prefix}_count"),
        candidate_count=fields.get(f"{candidate_prefix}_count"),
        baseline_stats_status=fields.get(f"{baseline_prefix}_stats_status"),
        candidate_stats_status=fields.get(f"{candidate_prefix}_stats_status"),
    )
    fields[f"{effect_prefix}_methodology_status"] = status
    fields[f"{effect_prefix}_methodology_caveats"] = "; ".join(caveats)
    return fields


def _get(row: pd.Series, metric: str, suffix: str) -> Any:
    value = row.get(f"{metric}_{suffix}")
    if pd.isna(value):
        return None
    return value
