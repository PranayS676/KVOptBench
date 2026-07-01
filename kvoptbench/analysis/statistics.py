"""Statistical comparison helpers for repeated benchmark runs."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

NORMAL_95_Z = 1.96
ROUND_DIGITS = 3
STAT_SUFFIXES = (
    "count",
    "mean",
    "p50",
    "p95",
    "std",
    "min",
    "max",
    "ci95_low",
    "ci95_high",
)

ResultsInput = pd.DataFrame | str | Path

__all__ = [
    "NORMAL_95_Z",
    "STAT_SUFFIXES",
    "aggregate_repeated_results",
    "compare_aggregates",
    "compare_repeated_results",
    "load_results",
    "percent_delta",
    "summarize_metric_values",
]


def load_results(input_data: ResultsInput) -> pd.DataFrame:
    """Load benchmark results from a DataFrame, CSV file, JSONL file, or directory."""
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    input_path = Path(input_data)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_dir():
        files = sorted(
            [*input_path.glob("*.csv"), *input_path.glob("*.jsonl")],
            key=lambda path: path.name,
        )
        if not files:
            raise ValueError(f"No CSV or JSONL result files found under {input_path}")
        frames = [_read_result_file(file) for file in files]
        return pd.concat(frames, ignore_index=True, sort=False)

    return _read_result_file(input_path)


def summarize_metric_values(values: Iterable[Any]) -> dict[str, float | int | None]:
    """Summarize one metric while ignoring missing and non-numeric values."""
    numeric_values = _valid_numeric_values(values)
    count = len(numeric_values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean = statistics.fmean(numeric_values)
    series = pd.Series(numeric_values, dtype="float64")
    std = statistics.stdev(numeric_values) if count >= 2 else None
    ci95_low = None
    ci95_high = None
    if std is not None:
        margin = NORMAL_95_Z * (std / math.sqrt(count))
        ci95_low = _round(mean - margin)
        ci95_high = _round(mean + margin)

    return {
        "count": count,
        "mean": _round(mean),
        "p50": _round(series.quantile(0.50)),
        "p95": _round(series.quantile(0.95)),
        "std": _round(std) if std is not None else None,
        "min": _round(min(numeric_values)),
        "max": _round(max(numeric_values)),
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }


def aggregate_repeated_results(
    data: ResultsInput,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate repeated trials by configurable keys and metric columns."""
    frame = load_results(data)
    group_columns = list(group_columns)
    metric_columns = list(metric_columns)
    _require_columns(frame, group_columns, purpose="grouping")

    output_columns = _aggregate_columns(group_columns, metric_columns)
    if frame.empty:
        return _records_to_frame([], output_columns)

    records: list[dict[str, Any]] = []
    for keys, group in _iter_groups(frame, group_columns):
        record = _group_record(group_columns, keys)
        for metric in metric_columns:
            values = group[metric] if metric in group.columns else []
            summary = summarize_metric_values(values)
            for suffix in STAT_SUFFIXES:
                record[f"{metric}_{suffix}"] = summary[suffix]
        records.append(record)

    return _records_to_frame(records, output_columns)


def compare_aggregates(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    label_column: str = "label",
) -> pd.DataFrame:
    """Compare pre-aggregated baseline and candidate summaries."""
    group_columns = list(group_columns)
    metric_columns = list(metric_columns)
    baseline = baseline.copy()
    candidate = candidate.copy()
    _require_columns(baseline, group_columns, purpose="baseline grouping")
    _require_columns(candidate, group_columns, purpose="candidate grouping")

    output_columns = _comparison_columns(group_columns, metric_columns, label_column)
    if baseline.empty or candidate.empty:
        return _records_to_frame([], output_columns)

    if group_columns:
        merged = baseline.merge(
            candidate,
            on=group_columns,
            how="inner",
            suffixes=("_baseline", "_candidate"),
            sort=True,
        )
    else:
        merged = baseline.head(1).merge(
            candidate.head(1),
            how="cross",
            suffixes=("_baseline", "_candidate"),
            sort=True,
        )

    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        record: dict[str, Any] = {
            column: _none_if_missing(row[column]) for column in group_columns
        }
        record[f"baseline_{label_column}"] = baseline_label
        record[f"candidate_{label_column}"] = candidate_label

        for metric in metric_columns:
            baseline_mean: float | None = None
            candidate_mean: float | None = None
            for suffix in STAT_SUFFIXES:
                baseline_value = _none_if_missing(row.get(f"{metric}_{suffix}_baseline"))
                candidate_value = _none_if_missing(row.get(f"{metric}_{suffix}_candidate"))
                record[f"{metric}_baseline_{suffix}"] = baseline_value
                record[f"{metric}_candidate_{suffix}"] = candidate_value
                if suffix == "mean":
                    baseline_mean = _as_float_or_none(baseline_value)
                    candidate_mean = _as_float_or_none(candidate_value)
            record[f"{metric}_percent_delta"] = percent_delta(baseline_mean, candidate_mean)
        records.append(record)

    return _records_to_frame(records, output_columns)


def compare_repeated_results(
    data: ResultsInput,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    strategy_column: str = "strategy",
    baseline_strategy: str = "baseline",
    candidate_strategy: str = "candidate",
) -> pd.DataFrame:
    """Aggregate and compare repeated trials for baseline/candidate strategies."""
    frame = load_results(data)
    _require_columns(frame, [strategy_column], purpose="strategy comparison")

    baseline = frame[frame[strategy_column] == baseline_strategy]
    candidate = frame[frame[strategy_column] == candidate_strategy]
    baseline_summary = aggregate_repeated_results(
        baseline,
        group_columns=group_columns,
        metric_columns=metric_columns,
    )
    candidate_summary = aggregate_repeated_results(
        candidate,
        group_columns=group_columns,
        metric_columns=metric_columns,
    )
    return compare_aggregates(
        baseline_summary,
        candidate_summary,
        group_columns=group_columns,
        metric_columns=metric_columns,
        baseline_label=baseline_strategy,
        candidate_label=candidate_strategy,
        label_column=strategy_column,
    )


def percent_delta(
    baseline: float | int | None,
    candidate: float | int | None,
) -> float | None:
    """Return candidate percent change relative to baseline."""
    baseline_value = _as_float_or_none(baseline)
    candidate_value = _as_float_or_none(candidate)
    if baseline_value is None or candidate_value is None or baseline_value == 0:
        return None
    return _round(((candidate_value - baseline_value) / baseline_value) * 100.0)


def _read_result_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"No JSONL result rows found in {path}")
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported result file type: {path.suffix}")


def _valid_numeric_values(values: Iterable[Any]) -> list[float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return [float(value) for value in series.dropna().tolist() if math.isfinite(float(value))]


def _iter_groups(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
    if not group_columns:
        yield (), frame
        return

    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        yield keys, group


def _group_record(group_columns: Sequence[str], keys: Sequence[Any]) -> dict[str, Any]:
    return {
        column: _none_if_missing(value)
        for column, value in zip(group_columns, keys, strict=True)
    }


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, purpose: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {purpose} column(s): {', '.join(missing)}")


def _aggregate_columns(
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> list[str]:
    columns = list(group_columns)
    for metric in metric_columns:
        columns.extend(f"{metric}_{suffix}" for suffix in STAT_SUFFIXES)
    return columns


def _comparison_columns(
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    label_column: str,
) -> list[str]:
    columns = [
        *group_columns,
        f"baseline_{label_column}",
        f"candidate_{label_column}",
    ]
    for metric in metric_columns:
        for suffix in STAT_SUFFIXES:
            columns.append(f"{metric}_baseline_{suffix}")
            columns.append(f"{metric}_candidate_{suffix}")
        columns.append(f"{metric}_percent_delta")
    return columns


def _records_to_frame(records: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=list(columns)).astype(object)
    return frame.where(pd.notna(frame), None)


def _none_if_missing(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _as_float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)
