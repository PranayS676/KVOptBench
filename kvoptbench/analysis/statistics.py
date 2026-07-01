"""Statistical comparison helpers for repeated benchmark runs."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
    "MethodologyPolicy",
    "MethodologyPolicyResult",
    "NORMAL_95_Z",
    "STAT_SUFFIXES",
    "aggregate_repeated_results",
    "apply_methodology_policy",
    "bootstrap_mean_ci",
    "compare_aggregates",
    "compare_repeated_results",
    "comparison_methodology_status",
    "flatten_metric_stats",
    "load_results",
    "mean_effect_size",
    "mean_effect_size_from_stats",
    "percent_delta",
    "summarize_metric_values",
]


@dataclass(frozen=True, slots=True)
class MethodologyPolicy:
    """Configurable methodology rules applied before aggregation."""

    warmup_column: str | None = None
    outlier_policy: str = "none"
    outlier_metric_columns: Sequence[str] = ()
    min_samples: int = 2
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 0


@dataclass(frozen=True, slots=True)
class MethodologyPolicyResult:
    """Filtered data and caveats produced by methodology policy enforcement."""

    frame: pd.DataFrame
    removed_warmup_rows: int
    removed_outlier_rows: int
    caveats: list[str]


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


def apply_methodology_policy(
    frame: pd.DataFrame,
    policy: MethodologyPolicy | None = None,
) -> MethodologyPolicyResult:
    """Apply warmup exclusion and optional outlier filtering to a result frame."""
    policy = policy or MethodologyPolicy()
    filtered = frame.copy()
    caveats: list[str] = []
    removed_warmup = 0
    removed_outliers = 0

    if policy.warmup_column and policy.warmup_column in filtered.columns:
        warmup_mask = filtered[policy.warmup_column].fillna(False).astype(bool)
        removed_warmup = int(warmup_mask.sum())
        filtered = filtered.loc[~warmup_mask].copy()
        if removed_warmup:
            caveats.append(f"warmup rows excluded: {removed_warmup}")

    if policy.outlier_policy == "iqr" and policy.outlier_metric_columns:
        keep_mask = pd.Series(True, index=filtered.index)
        for metric in policy.outlier_metric_columns:
            if metric not in filtered.columns:
                continue
            values = pd.to_numeric(filtered[metric], errors="coerce")
            valid = values.dropna()
            if len(valid) < 4:
                continue
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            keep_mask &= values.isna() | values.between(lower, upper)
        removed_outliers = int((~keep_mask).sum())
        filtered = filtered.loc[keep_mask].copy()
        if removed_outliers:
            caveats.append(f"outlier rows excluded by iqr policy: {removed_outliers}")
    elif policy.outlier_policy not in {"none", "iqr"}:
        raise ValueError(f"Unsupported outlier policy: {policy.outlier_policy}")

    return MethodologyPolicyResult(
        frame=filtered.reset_index(drop=True),
        removed_warmup_rows=removed_warmup,
        removed_outlier_rows=removed_outliers,
        caveats=caveats,
    )


def bootstrap_mean_ci(
    values: Iterable[Any],
    *,
    iterations: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Return a deterministic bootstrap confidence interval for the mean."""
    numeric_values = _valid_numeric_values(values)
    if len(numeric_values) < 2:
        return None, None
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    sample_size = len(numeric_values)
    means = []
    for _ in range(iterations):
        sample = [numeric_values[rng.randrange(sample_size)] for _ in range(sample_size)]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2
    low_index = max(0, min(len(means) - 1, int(alpha * len(means))))
    high_index = max(0, min(len(means) - 1, int((1.0 - alpha) * len(means)) - 1))
    return _round(means[low_index]), _round(means[high_index])


def summarize_metric_values(
    values: Iterable[Any],
    *,
    ci_method: str = "normal",
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, float | int | None]:
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
    if std is not None and ci_method == "normal":
        margin = NORMAL_95_Z * (std / math.sqrt(count))
        ci95_low = _round(mean - margin)
        ci95_high = _round(mean + margin)
    elif ci_method == "bootstrap":
        ci95_low, ci95_high = bootstrap_mean_ci(
            numeric_values,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
    elif ci_method != "normal":
        raise ValueError(f"Unsupported ci_method: {ci_method}")

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


def flatten_metric_stats(
    prefix: str,
    values: Iterable[Any],
    *,
    ci_method: str = "normal",
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, float | int | str | None]:
    """Return stable repeated-run columns for one metric prefix."""
    stats = summarize_metric_values(
        values,
        ci_method=ci_method,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    count = int(stats["count"] or 0)
    return {
        f"{prefix}_count": count,
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_p50": stats["p50"],
        f"{prefix}_p95": stats["p95"],
        f"{prefix}_std": stats["std"],
        f"{prefix}_ci95_low": stats["ci95_low"],
        f"{prefix}_ci95_high": stats["ci95_high"],
        f"{prefix}_stats_status": "ok"
        if count >= 2
        else ("missing_metric" if count == 0 else "insufficient_repetitions"),
    }


def mean_effect_size(
    baseline_values: Iterable[Any],
    candidate_values: Iterable[Any],
) -> float | None:
    """Return Cohen's d effect size from repeated baseline/candidate samples."""
    baseline = _valid_numeric_values(baseline_values)
    candidate = _valid_numeric_values(candidate_values)
    if len(baseline) < 2 or len(candidate) < 2:
        return None
    baseline_std = statistics.stdev(baseline)
    candidate_std = statistics.stdev(candidate)
    pooled_variance = (
        ((len(baseline) - 1) * baseline_std**2)
        + ((len(candidate) - 1) * candidate_std**2)
    ) / (len(baseline) + len(candidate) - 2)
    if pooled_variance <= 0:
        return None
    pooled_std = math.sqrt(pooled_variance)
    if pooled_std == 0:
        return None
    return _round((statistics.fmean(candidate) - statistics.fmean(baseline)) / pooled_std)


def mean_effect_size_from_stats(
    *,
    baseline_mean: float | int | None,
    baseline_std: float | int | None,
    baseline_count: float | int | None,
    candidate_mean: float | int | None,
    candidate_std: float | int | None,
    candidate_count: float | int | None,
) -> float | None:
    """Return Cohen's d from pre-aggregated mean/std/count fields."""
    b_mean = _as_float_or_none(baseline_mean)
    c_mean = _as_float_or_none(candidate_mean)
    b_std = _as_float_or_none(baseline_std)
    c_std = _as_float_or_none(candidate_std)
    b_count = _as_float_or_none(baseline_count)
    c_count = _as_float_or_none(candidate_count)
    if (
        b_mean is None
        or c_mean is None
        or b_std is None
        or c_std is None
        or b_count is None
        or c_count is None
        or b_count < 2
        or c_count < 2
    ):
        return None
    pooled_variance = (((b_count - 1) * b_std**2) + ((c_count - 1) * c_std**2)) / (
        b_count + c_count - 2
    )
    if pooled_variance <= 0:
        return None
    return _round((c_mean - b_mean) / math.sqrt(pooled_variance))


def aggregate_repeated_results(
    data: ResultsInput,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    methodology_policy: MethodologyPolicy | None = None,
    ci_method: str = "normal",
) -> pd.DataFrame:
    """Aggregate repeated trials by configurable keys and metric columns."""
    frame = load_results(data)
    policy_result = apply_methodology_policy(frame, methodology_policy)
    frame = policy_result.frame
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
            summary = summarize_metric_values(values, ci_method=ci_method)
            for suffix in STAT_SUFFIXES:
                record[f"{metric}_{suffix}"] = summary[suffix]
        record["methodology_status"] = _aggregate_methodology_status(
            record,
            metric_columns,
            methodology_policy or MethodologyPolicy(),
        )
        record["methodology_caveats"] = "; ".join(policy_result.caveats)
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
    min_samples: int = 2,
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
            record[f"{metric}_effect_size"] = mean_effect_size_from_stats(
                baseline_mean=baseline_mean,
                baseline_std=record.get(f"{metric}_baseline_std"),
                baseline_count=record.get(f"{metric}_baseline_count"),
                candidate_mean=candidate_mean,
                candidate_std=record.get(f"{metric}_candidate_std"),
                candidate_count=record.get(f"{metric}_candidate_count"),
            )
            status, caveats = comparison_methodology_status(
                baseline_count=record.get(f"{metric}_baseline_count"),
                candidate_count=record.get(f"{metric}_candidate_count"),
                baseline_stats_status=record.get(f"{metric}_baseline_stats_status"),
                candidate_stats_status=record.get(f"{metric}_candidate_stats_status"),
                min_samples=min_samples,
            )
            record[f"{metric}_comparison_status"] = status
            record[f"{metric}_comparison_caveats"] = "; ".join(caveats)
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
    min_samples: int = 2,
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
        min_samples=min_samples,
    )


def comparison_methodology_status(
    *,
    baseline_count: Any,
    candidate_count: Any,
    baseline_stats_status: Any = None,
    candidate_stats_status: Any = None,
    min_samples: int = 2,
) -> tuple[str, list[str]]:
    """Return an evidence status and caveats for one baseline/candidate comparison."""
    baseline_n = int(_as_float_or_none(baseline_count) or 0)
    candidate_n = int(_as_float_or_none(candidate_count) or 0)
    caveats: list[str] = []
    if baseline_n < min_samples or candidate_n < min_samples:
        caveats.append(
            "insufficient samples: "
            f"baseline n={baseline_n}, candidate n={candidate_n}, minimum={min_samples}"
        )
    for label, status in [
        ("baseline", baseline_stats_status),
        ("candidate", candidate_stats_status),
    ]:
        status_value = _none_if_missing(status)
        if status_value not in (None, "ok") and str(status_value).strip():
            caveats.append(f"{label} stats status: {status_value}")
    return ("exploratory" if caveats else "ok"), caveats


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
    columns.extend(["methodology_status", "methodology_caveats"])
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
        columns.append(f"{metric}_effect_size")
        columns.append(f"{metric}_comparison_status")
        columns.append(f"{metric}_comparison_caveats")
    return columns


def _aggregate_methodology_status(
    record: dict[str, Any],
    metric_columns: Sequence[str],
    policy: MethodologyPolicy,
) -> str:
    for metric in metric_columns:
        count = int(_as_float_or_none(record.get(f"{metric}_count")) or 0)
        status = record.get(f"{metric}_stats_status")
        if count < policy.min_samples or status not in (None, "ok"):
            return "exploratory"
    return "ok"


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
