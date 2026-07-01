"""Transparent strategy advisor for KVOptBench comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from kvoptbench.schemas import utc_now_iso
from kvoptbench.strategy.report import render_strategy_advisor_markdown

Decision = Literal["recommend", "consider", "do_not_recommend", "inconclusive", "needs_more_data"]
Confidence = Literal["high", "medium", "low"]

MEANINGFUL_CACHE_GAIN_MS = 25.0
MEANINGFUL_THROUGHPUT_DELTA_PCT = 5.0
MEANINGFUL_LATENCY_DELTA_PCT = -5.0
MEANINGFUL_MEMORY_DELTA_PCT = -5.0
QUALITY_REGRESSION_THRESHOLD = -0.05
TINY_SAMPLE_SUPPORT = 4


class EvidenceItem(BaseModel):
    """One observed metric or result label used by a recommendation."""

    message: str
    metric: str | None = None
    value: float | str | None = None


class StrategyRecommendation(BaseModel):
    """Advisor output for one strategy."""

    strategy: str
    decision: Decision
    confidence: Confidence
    confidence_score: float = 0.0
    confidence_reasons: list[str] = Field(default_factory=list)
    rank: int | None = None
    score: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    next_experiments: list[str] = Field(default_factory=list)
    next_experiment_priority: list[str] = Field(default_factory=list)
    quality_guardrail: str | None = None
    source: str


class StrategyAdvisorReport(BaseModel):
    """Full strategy advisor report."""

    generated_at: str = Field(default_factory=utc_now_iso)
    overall_recommendation: str
    recommendations: list[StrategyRecommendation]


class StrategyAdvisorInputs(BaseModel):
    """Input paths consumed by the advisor."""

    summary_path: Path
    cache_input_path: Path | None = None
    prefix_sweep_input_path: Path | None = None
    prefill_decode_input_path: Path | None = None
    long_context_input_path: Path | None = None
    kv_quant_input_path: Path | None = None
    kv_offload_input_path: Path | None = None
    spec_decoding_input_path: Path | None = None
    disagg_input_path: Path | None = None


def build_strategy_advisor_report(
    *,
    summary_path: str | Path,
    cache_input_path: str | Path | None = None,
    prefix_sweep_input_path: str | Path | None = None,
    prefill_decode_input_path: str | Path | None = None,
    long_context_input_path: str | Path | None = None,
    kv_quant_input_path: str | Path | None = None,
    kv_offload_input_path: str | Path | None = None,
    spec_decoding_input_path: str | Path | None = None,
    disagg_input_path: str | Path | None = None,
) -> StrategyAdvisorReport:
    """Build a deterministic recommendation report from available CSV outputs."""
    inputs = StrategyAdvisorInputs(
        summary_path=Path(summary_path),
        cache_input_path=_optional_path(cache_input_path),
        prefix_sweep_input_path=_optional_path(prefix_sweep_input_path),
        prefill_decode_input_path=_optional_path(prefill_decode_input_path),
        long_context_input_path=_optional_path(long_context_input_path),
        kv_quant_input_path=_optional_path(kv_quant_input_path),
        kv_offload_input_path=_optional_path(kv_offload_input_path),
        spec_decoding_input_path=_optional_path(spec_decoding_input_path),
        disagg_input_path=_optional_path(disagg_input_path),
    )
    summary = _read_required_csv(inputs.summary_path)
    frames = {
        "cache": _read_optional_csv(inputs.cache_input_path),
        "prefix_sweep": _read_optional_csv(inputs.prefix_sweep_input_path),
        "long_context": _read_optional_csv(inputs.long_context_input_path),
        "kv_quant": _read_optional_csv(inputs.kv_quant_input_path),
        "kv_offload": _read_optional_csv(inputs.kv_offload_input_path),
        "spec_decoding": _read_optional_csv(inputs.spec_decoding_input_path),
        "disagg": _read_optional_csv(inputs.disagg_input_path),
    }

    recommendations = [
        _apply_confidence_model(
            evaluate_prefix_caching(frames["cache"], frames["prefix_sweep"]),
            evidence_frames=[frames["cache"], frames["prefix_sweep"]],
            summary_frame=summary,
        ),
        _apply_confidence_model(
            evaluate_kv_quantization(frames["kv_quant"]),
            evidence_frames=[frames["kv_quant"]],
            summary_frame=summary,
        ),
        _apply_confidence_model(
            evaluate_kv_offload(frames["kv_offload"]),
            evidence_frames=[frames["kv_offload"]],
            summary_frame=summary,
        ),
        _apply_confidence_model(
            evaluate_speculative_decoding(frames["spec_decoding"]),
            evidence_frames=[frames["spec_decoding"]],
            summary_frame=summary,
        ),
        _apply_confidence_model(
            evaluate_disaggregation(frames["disagg"]),
            evidence_frames=[frames["disagg"]],
            summary_frame=summary,
        ),
    ]
    recommendations.extend(
        _apply_confidence_model(
            item,
            evidence_frames=[frames["long_context"]],
            summary_frame=summary,
        )
        for item in evaluate_long_context_next_steps(frames["long_context"], summary)
    )
    ranked = _rank_recommendations(recommendations)
    return StrategyAdvisorReport(
        overall_recommendation=_overall_recommendation(ranked),
        recommendations=ranked,
    )


def write_strategy_advisor_outputs(
    *,
    report: StrategyAdvisorReport,
    json_output_path: str | Path | None = None,
    markdown_output_path: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Write JSON and/or Markdown advisor outputs."""
    json_path = Path(json_output_path) if json_output_path is not None else None
    markdown_path = Path(markdown_output_path) if markdown_output_path is not None else None
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_strategy_advisor_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def evaluate_prefix_caching(
    cache_frame: pd.DataFrame | None,
    prefix_sweep_frame: pd.DataFrame | None,
) -> StrategyRecommendation:
    """Recommend prefix caching when cache ablations show a meaningful warm-prefix gain."""
    if _is_missing_frame(cache_frame):
        return _needs_more_data(
            "prefix_caching",
            "cache comparison CSV",
            "Run cache-plan, cache-run, cache-compare, then rerun the advisor.",
        )

    gain_row = _max_numeric_row(cache_frame, "control_adjusted_cache_gain_ms")
    if gain_row is None:
        return StrategyRecommendation(
            strategy="prefix_caching",
            decision="inconclusive",
            confidence="low",
            score=0.0,
            source="cache comparison CSV",
            caveats=["Cache comparison is missing control-adjusted cache gain."],
            next_experiments=[
                "Run shared-prefix and random-prefix controls so cache gain is isolated."
            ],
        )

    gain = _to_float(gain_row.get("control_adjusted_cache_gain_ms"))
    interpretation = _clean_string(gain_row.get("interpretation"))
    evidence = [
        EvidenceItem(
            message=f"Observed control-adjusted cache gain of {_fmt_metric(gain)} ms.",
            metric="control_adjusted_cache_gain_ms",
            value=gain,
        )
    ]
    caveats: list[str] = []
    next_experiments: list[str] = []
    prefix_bonus = 0.0

    meaningful_prefix_row = _first_prefix_sweep_hit(prefix_sweep_frame)
    if meaningful_prefix_row is not None:
        ratio = _to_float(meaningful_prefix_row.get("shared_prefix_ratio"))
        evidence.append(
            EvidenceItem(
                message=(
                    "Meaningful cache gain appeared at shared-prefix ratio "
                    f"{_fmt_metric(ratio)}."
                ),
                metric="shared_prefix_ratio",
                value=ratio,
            )
        )
        prefix_bonus = 0.5
    elif prefix_sweep_frame is None:
        caveats.append("No prefix-overlap sweep CSV was provided.")
        next_experiments.append(
            "Run prefix-sweep-compare to find the minimum shared-prefix ratio that pays off."
        )

    if interpretation == "credible_cache_reuse_signal" or (
        gain is not None and gain >= MEANINGFUL_CACHE_GAIN_MS
    ):
        confidence: Confidence = "high" if meaningful_prefix_row is not None else "medium"
        return StrategyRecommendation(
            strategy="prefix_caching",
            decision="recommend",
            confidence=confidence,
            score=3.0 + ((gain or 0.0) / 100.0) + prefix_bonus,
            evidence=evidence,
            caveats=caveats,
            next_experiments=next_experiments,
            source="cache comparison CSV",
        )

    if gain is not None and gain < 0:
        return StrategyRecommendation(
            strategy="prefix_caching",
            decision="do_not_recommend",
            confidence="medium",
            score=-1.0,
            evidence=evidence,
            caveats=["Warm-prefix traffic did not outperform the random-prefix control."],
            next_experiments=[
                "Retest with higher shared-prefix overlap before enabling prefix caching broadly."
            ],
            source="cache comparison CSV",
        )

    return StrategyRecommendation(
        strategy="prefix_caching",
        decision="inconclusive",
        confidence="low",
        score=0.0,
        evidence=evidence,
        caveats=["Observed cache gain did not cross the meaningful-gain threshold."],
        next_experiments=[
            "Run a larger shared-prefix workload and include a prefix-overlap sweep."
        ],
        source="cache comparison CSV",
    )


def evaluate_kv_quantization(frame: pd.DataFrame | None) -> StrategyRecommendation:
    """Recommend KV quantization only when benefits outweigh quality and latency risk."""
    if _is_missing_frame(frame):
        return _needs_more_data(
            "kv_quantization",
            "KV quantization comparison CSV",
            "Run kv-quant-plan, kv-quant-run, kv-quant-compare, then rerun the advisor.",
        )

    row = _best_interpretation_row(frame, "quantization_interpretation")
    interpretation = _clean_string(row.get("quantization_interpretation"))
    evidence = _tradeoff_evidence(row)
    caveats = _missing_metric_caveats(row)

    if interpretation == "quantization_promising":
        confidence: Confidence = "medium" if caveats else "high"
        return StrategyRecommendation(
            strategy="kv_quantization",
            decision="recommend",
            confidence=confidence,
            score=2.5 + _positive_delta_score(row),
            evidence=evidence,
            caveats=caveats,
            next_experiments=[
                "Repeat KV quantization across larger context buckets and quality-sensitive tasks."
            ],
            source="KV quantization comparison CSV",
        )
    if interpretation in {"quality_regression", "latency_regression"}:
        return StrategyRecommendation(
            strategy="kv_quantization",
            decision="do_not_recommend",
            confidence="high",
            score=-2.0,
            evidence=evidence,
            caveats=[f"Comparison was classified as {interpretation}.", *caveats],
            next_experiments=[
                "Retest with a safer KV precision setting or a workload with lower quality risk."
            ],
            source="KV quantization comparison CSV",
        )

    return StrategyRecommendation(
        strategy="kv_quantization",
        decision="inconclusive",
        confidence="low",
        score=0.0,
        evidence=evidence,
        caveats=[f"Comparison was classified as {interpretation or 'unknown'}.", *caveats],
        next_experiments=[
            "Add memory telemetry and quality probes before choosing KV quantization."
        ],
        source="KV quantization comparison CSV",
    )


def evaluate_kv_offload(frame: pd.DataFrame | None) -> StrategyRecommendation:
    """Recommend KV offload only when memory relief or throughput gains are visible."""
    if _is_missing_frame(frame):
        return _needs_more_data(
            "kv_offload",
            "KV offload comparison CSV",
            "Run kv-offload-plan, kv-offload-run, kv-offload-compare, then rerun the advisor.",
        )

    row = _best_interpretation_row(frame, "offload_interpretation")
    interpretation = _clean_string(row.get("offload_interpretation"))
    evidence = _tradeoff_evidence(row)
    caveats = _missing_metric_caveats(row)

    if interpretation == "offload_promising":
        confidence: Confidence = "medium" if caveats else "high"
        return StrategyRecommendation(
            strategy="kv_offload",
            decision="consider",
            confidence=confidence,
            score=2.0 + _positive_delta_score(row),
            evidence=evidence,
            caveats=caveats,
            next_experiments=[
                "Stress long-context concurrency and capture host/device transfer telemetry."
            ],
            source="KV offload comparison CSV",
        )
    if interpretation == "memory_telemetry_missing":
        return StrategyRecommendation(
            strategy="kv_offload",
            decision="inconclusive",
            confidence="low",
            score=0.0,
            evidence=evidence,
            caveats=["KV offload memory telemetry is missing.", *caveats],
            next_experiments=["Rerun KV offload with GPU memory telemetry enabled."],
            source="KV offload comparison CSV",
        )
    if interpretation in {"quality_regression", "latency_regression"}:
        return StrategyRecommendation(
            strategy="kv_offload",
            decision="do_not_recommend",
            confidence="high",
            score=-2.0,
            evidence=evidence,
            caveats=[f"Comparison was classified as {interpretation}.", *caveats],
            next_experiments=[
                "Retest offload only after confirming memory pressure exceeds device capacity."
            ],
            source="KV offload comparison CSV",
        )

    return StrategyRecommendation(
        strategy="kv_offload",
        decision="inconclusive",
        confidence="low",
        score=0.0,
        evidence=evidence,
        caveats=[f"Comparison was classified as {interpretation or 'unknown'}.", *caveats],
        next_experiments=[
            "Run larger context buckets with GPU memory and transfer telemetry available."
        ],
        source="KV offload comparison CSV",
    )


def evaluate_speculative_decoding(frame: pd.DataFrame | None) -> StrategyRecommendation:
    """Recommend speculative decoding when decode-heavy runs improve without quality loss."""
    if _is_missing_frame(frame):
        return _needs_more_data(
            "speculative_decoding",
            "speculative decoding comparison CSV",
            "Run spec-decoding-plan, spec-decoding-run, spec-decoding-compare, then rerun the advisor.",
        )

    row = _best_interpretation_row(frame, "speculative_decoding_interpretation")
    interpretation = _clean_string(row.get("speculative_decoding_interpretation"))
    evidence = _tradeoff_evidence(row)
    caveats = _missing_metric_caveats(row)
    if "speculative_acceptance_rate" in _missing_metric_names(row):
        caveats.append("Speculative acceptance telemetry is missing.")

    if interpretation == "speculative_decoding_promising":
        confidence: Confidence = "medium" if caveats else "high"
        return StrategyRecommendation(
            strategy="speculative_decoding",
            decision="recommend",
            confidence=confidence,
            score=2.5 + _positive_delta_score(row),
            evidence=evidence,
            caveats=caveats,
            next_experiments=[
                "Retest with acceptance-rate telemetry and representative long-output prompts."
            ],
            source="speculative decoding comparison CSV",
        )
    if interpretation in {"quality_regression", "latency_regression"}:
        return StrategyRecommendation(
            strategy="speculative_decoding",
            decision="do_not_recommend",
            confidence="high",
            score=-2.0,
            evidence=evidence,
            caveats=[f"Comparison was classified as {interpretation}.", *caveats],
            next_experiments=[
                "Try a different draft model or disable speculative decoding for this workload."
            ],
            source="speculative decoding comparison CSV",
        )

    return StrategyRecommendation(
        strategy="speculative_decoding",
        decision="inconclusive",
        confidence="low",
        score=0.0,
        evidence=evidence,
        caveats=[f"Comparison was classified as {interpretation or 'unknown'}.", *caveats],
        next_experiments=[
            "Run decode-heavy prompts with acceptance-rate telemetry before enabling this strategy."
        ],
        source="speculative decoding comparison CSV",
    )


def evaluate_disaggregation(frame: pd.DataFrame | None) -> StrategyRecommendation:
    """Recommend prefill/decode disaggregation only when decode latency is not harmed."""
    if _is_missing_frame(frame):
        return _needs_more_data(
            "prefill_decode_disaggregation",
            "prefill/decode disaggregation comparison CSV",
            "Run disagg-plan, disagg-run, disagg-compare, then rerun the advisor.",
        )

    row = _best_interpretation_row(frame, "disaggregation_interpretation")
    interpretation = _clean_string(row.get("disaggregation_interpretation"))
    evidence = _tradeoff_evidence(row)
    caveats = _missing_metric_caveats(row)

    if interpretation == "disaggregation_promising":
        confidence: Confidence = "medium" if caveats else "high"
        return StrategyRecommendation(
            strategy="prefill_decode_disaggregation",
            decision="consider",
            confidence=confidence,
            score=2.0 + _positive_delta_score(row),
            evidence=evidence,
            caveats=caveats,
            next_experiments=[
                "Retest under mixed prefill/decode traffic with production-like routing."
            ],
            source="prefill/decode disaggregation comparison CSV",
        )
    if interpretation == "decode_regression":
        return StrategyRecommendation(
            strategy="prefill_decode_disaggregation",
            decision="do_not_recommend",
            confidence="high",
            score=-3.0,
            evidence=[
                EvidenceItem(
                    message="Disaggregated decode latency regressed.",
                    metric="disaggregation_interpretation",
                    value=interpretation,
                ),
                *evidence,
            ],
            caveats=caveats,
            next_experiments=[
                "Inspect decode placement, routing, and KV transfer path before retrying."
            ],
            source="prefill/decode disaggregation comparison CSV",
        )
    if interpretation in {"quality_regression", "latency_regression"}:
        return StrategyRecommendation(
            strategy="prefill_decode_disaggregation",
            decision="do_not_recommend",
            confidence="high",
            score=-2.0,
            evidence=evidence,
            caveats=[f"Comparison was classified as {interpretation}.", *caveats],
            next_experiments=[
                "Retest only after isolating whether prefill, decode, or KV transport regressed."
            ],
            source="prefill/decode disaggregation comparison CSV",
        )

    return StrategyRecommendation(
        strategy="prefill_decode_disaggregation",
        decision="inconclusive",
        confidence="low",
        score=0.0,
        evidence=evidence,
        caveats=[f"Comparison was classified as {interpretation or 'unknown'}.", *caveats],
        next_experiments=[
            "Run the prefill/decode grid with TTFT, TPOT, ITL, and throughput telemetry."
        ],
        source="prefill/decode disaggregation comparison CSV",
    )


def evaluate_long_context_next_steps(
    long_context_frame: pd.DataFrame | None,
    summary_frame: pd.DataFrame,
) -> list[StrategyRecommendation]:
    """Add a memory-pressure investigation item when long-context data is available."""
    if _is_missing_frame(long_context_frame):
        return []

    labels = sorted(
        {
            _clean_string(value)
            for value in long_context_frame.get(
                "pressure_classification", pd.Series(dtype=str)
            )
            if _clean_string(value)
        }
    )
    if not labels:
        return []

    requests = int(summary_frame["requests"].sum()) if "requests" in summary_frame else 0
    return [
        StrategyRecommendation(
            strategy="long_context_memory_pressure",
            decision="inconclusive",
            confidence="low",
            score=0.1,
            evidence=[
                EvidenceItem(
                    message="Observed long-context pressure classifications: "
                    + ", ".join(labels)
                    + ".",
                    metric="pressure_classification",
                    value=";".join(labels),
                )
            ],
            caveats=["This is a workload-shape signal, not a standalone serving strategy."],
            next_experiments=[
                f"Use the {requests} summarized requests to choose larger context sweeps.",
                "Compare KV quantization and KV offload on the stressed context buckets.",
            ],
            source="long-context comparison CSV",
        )
    ]


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value) if value is not None else None


def _read_required_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Summary CSV is empty: {path}")
    return frame


def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return pd.read_csv(path)


def _needs_more_data(
    strategy: str,
    source: str,
    next_experiment: str,
) -> StrategyRecommendation:
    return StrategyRecommendation(
        strategy=strategy,
        decision="needs_more_data",
        confidence="low",
        score=-1.0,
        source=source,
        caveats=[f"No {source} was provided."],
        next_experiments=[next_experiment],
    )


def _apply_confidence_model(
    recommendation: StrategyRecommendation,
    *,
    evidence_frames: list[pd.DataFrame | None],
    summary_frame: pd.DataFrame,
) -> StrategyRecommendation:
    """Add transparent confidence scoring without changing legacy fields."""
    evidence_rows = _frame_rows(evidence_frames)
    summary_rows = _frame_rows([summary_frame])
    context_rows = [*evidence_rows, *summary_rows]

    score = _base_confidence_score(recommendation)
    reasons: list[str] = []
    priority: list[str] = []

    metric_evidence_count = sum(1 for item in recommendation.evidence if item.metric)
    if metric_evidence_count:
        reasons.append(
            f"Evidence completeness: {metric_evidence_count} metric-backed evidence item(s)."
        )
    else:
        reasons.append(
            "Evidence completeness is limited: no metric-backed evidence items were available."
        )
        score -= 0.20
        _append_unique(
            priority,
            "Add comparable latency, throughput, quality, and memory deltas before relying on this recommendation.",
        )

    missing_metrics = _collect_missing_metrics(evidence_rows)
    if not evidence_rows:
        missing_metrics = missing_metrics or _collect_missing_metrics(summary_rows)
    if missing_metrics:
        score -= min(0.08 + (0.04 * len(missing_metrics)), 0.30)
        reasons.append(
            "missing telemetry lowers confidence: " + ", ".join(missing_metrics) + "."
        )
        for next_experiment in _missing_metric_next_experiments(missing_metrics):
            _append_unique(priority, next_experiment)
    else:
        reasons.append("Missing telemetry: no missing metrics were reported.")

    sample_count = _sample_support_count(evidence_rows)
    if sample_count is None:
        sample_count = _sample_support_count(summary_rows)
    if sample_count is None:
        reasons.append("Sample support: no request/sample count column was available.")
    elif sample_count < TINY_SAMPLE_SUPPORT:
        score -= 0.25
        reasons.append(
            f"Tiny sample support: only {sample_count} requests/trials were reported."
        )
        _append_unique(
            priority,
            "Repeat trials until each compared condition has at least 4 requests before relying on the recommendation.",
        )
    else:
        reasons.append(f"Sample support: {sample_count} requests/trials were reported.")

    score = _apply_quality_guardrail(recommendation, evidence_rows, reasons, priority, score)

    if _has_mock_source(context_rows):
        score -= 0.05
        _append_unique(
            recommendation.caveats,
            "mock source data validates benchmark wiring only; it is not real engine validation.",
        )
        reasons.append(
            "Source type: mock source; do not treat this as real engine validation."
        )
        _append_unique(
            priority,
            "If making real-engine performance claims, repeat on a real endpoint; otherwise treat this as pipeline validation.",
        )
    elif _has_source_signal(context_rows):
        reasons.append("Source type: no mock source fields were detected.")

    if recommendation.strategy == "prefix_caching" and any(
        "No prefix-overlap sweep CSV" in caveat for caveat in recommendation.caveats
    ):
        score -= 0.15
        reasons.append("Prefix sweep evidence is missing, so the cache threshold is unknown.")
        _append_unique(
            priority,
            "Run prefix-sweep-compare to find the minimum shared-prefix ratio that pays off.",
        )

    recommendation.confidence_score = _clamp_score(score)
    recommendation.confidence = _confidence_from_score(recommendation.confidence_score)
    recommendation.confidence_reasons = reasons
    recommendation.next_experiment_priority = priority
    return recommendation


def _base_confidence_score(recommendation: StrategyRecommendation) -> float:
    if recommendation.decision == "needs_more_data":
        return 0.20
    return {"high": 0.95, "medium": 0.70, "low": 0.40}[recommendation.confidence]


def _frame_rows(frames: list[pd.DataFrame | None]) -> list[pd.Series]:
    rows: list[pd.Series] = []
    for frame in frames:
        if _is_missing_frame(frame):
            continue
        rows.extend(row for _, row in frame.iterrows())
    return rows


def _collect_missing_metrics(rows: list[pd.Series]) -> list[str]:
    missing: set[str] = set()
    for row in rows:
        missing.update(_missing_metric_names(row))
    return sorted(missing)


def _sample_support_count(rows: list[pd.Series]) -> int | None:
    columns = (
        "requests",
        "request_count",
        "sample_count",
        "samples",
        "trial_count",
        "repetition_count",
        "repeat_count",
        "count",
    )
    counts: list[int] = []
    for row in rows:
        for column in columns:
            if column not in row.index:
                continue
            value = _to_float(row.get(column))
            if value is not None and value > 0:
                counts.append(int(value))
    if not counts:
        return None
    return min(counts)


def _apply_quality_guardrail(
    recommendation: StrategyRecommendation,
    rows: list[pd.Series],
    reasons: list[str],
    priority: list[str],
    score: float,
) -> float:
    if recommendation.strategy not in {
        "kv_quantization",
        "kv_offload",
        "speculative_decoding",
        "prefill_decode_disaggregation",
    }:
        return score
    if not rows:
        return score

    quality_delta = _min_numeric_row_value(rows, "quality_delta")
    if quality_delta is None:
        recommendation.quality_guardrail = "unknown"
        if recommendation.decision in {"recommend", "consider"}:
            score -= 0.20
            reasons.append(
                "Quality guardrail is unknown because quality_delta telemetry is missing."
            )
            _append_unique(
                priority,
                "Add quality-score telemetry or task-specific evaluators before selecting this strategy.",
            )
        return score

    if quality_delta < QUALITY_REGRESSION_THRESHOLD:
        recommendation.quality_guardrail = "failed"
        reasons.append(
            "Failed quality guardrail: quality delta "
            f"{_fmt_metric(quality_delta)} is below {_fmt_metric(QUALITY_REGRESSION_THRESHOLD)}."
        )
        _append_unique(
            recommendation.caveats,
            "Quality guardrail failed; observed quality regression is beyond the allowed threshold.",
        )
        _append_unique(
            priority,
            "Retest with quality-sensitive tasks or a safer strategy setting before considering rollout.",
        )
        if recommendation.decision in {"recommend", "consider"}:
            recommendation.decision = "do_not_recommend"
            recommendation.score = min(recommendation.score, -2.0)
        return score

    recommendation.quality_guardrail = "passed"
    reasons.append(
        "Quality guardrail passed: quality delta "
        f"{_fmt_metric(quality_delta)} stayed above {_fmt_metric(QUALITY_REGRESSION_THRESHOLD)}."
    )
    return score


def _min_numeric_row_value(rows: list[pd.Series], column: str) -> float | None:
    values = [_to_float(row.get(column)) for row in rows if column in row.index]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return min(numeric)


def _missing_metric_next_experiments(missing_metrics: list[str]) -> list[str]:
    experiments: list[str] = []
    lower_missing = {metric.lower() for metric in missing_metrics}
    if any("gpu_memory" in metric or "memory" in metric for metric in lower_missing):
        experiments.append("Capture GPU memory telemetry and rerun the comparison.")
    if "speculative_acceptance_rate" in lower_missing:
        experiments.append(
            "Add speculative acceptance-rate telemetry before promoting speculative decoding."
        )
    if any("cache_hit" in metric or "cache_miss" in metric for metric in lower_missing):
        experiments.append(
            "Capture cache-hit or cache-proxy telemetry for the cache comparison."
        )
    if any("quality" in metric for metric in lower_missing):
        experiments.append(
            "Add quality-score telemetry or task-specific evaluators before selecting this strategy."
        )
    if any(
        metric in lower_missing
        for metric in {"ttft_ms", "tpot_ms", "itl_ms", "e2e_latency_ms"}
    ):
        experiments.append(
            "Capture TTFT, TPOT, ITL, and end-to-end latency telemetry for the comparison."
        )
    if not experiments:
        experiments.append("Capture the missing telemetry and rerun the comparison.")
    return experiments


def _has_mock_source(rows: list[pd.Series]) -> bool:
    source_columns = ("source_type", "provider", "endpoint_type", "engine", "model_id")
    for row in rows:
        for column in source_columns:
            if column not in row:
                continue
            value = _clean_string(row.get(column)).lower()
            if "mock" in value:
                return True
    return False


def _has_source_signal(rows: list[pd.Series]) -> bool:
    source_columns = ("source_type", "provider", "endpoint_type", "engine", "model_id")
    for row in rows:
        if any(column in row and _clean_string(row.get(column)) for column in source_columns):
            return True
    return False


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)


def _confidence_from_score(score: float) -> Confidence:
    if score >= 0.80:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _rank_recommendations(
    recommendations: list[StrategyRecommendation],
) -> list[StrategyRecommendation]:
    decision_rank = {
        "recommend": 0,
        "consider": 1,
        "do_not_recommend": 2,
        "inconclusive": 3,
        "needs_more_data": 4,
    }
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(
        recommendations,
        key=lambda item: (
            decision_rank[item.decision],
            confidence_rank[item.confidence],
            -item.score,
            item.strategy,
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item.rank = rank
    return ranked


def _overall_recommendation(recommendations: list[StrategyRecommendation]) -> str:
    for item in recommendations:
        if item.decision in {"recommend", "consider"}:
            return item.strategy
    return "needs_more_data"


def _is_missing_frame(frame: pd.DataFrame | None) -> bool:
    return frame is None or frame.empty


def _max_numeric_row(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.dropna().empty:
        return None
    return frame.loc[numeric.idxmax()]


def _first_prefix_sweep_hit(frame: pd.DataFrame | None) -> pd.Series | None:
    if _is_missing_frame(frame) or "interpretation" not in frame:
        return None
    matches = frame[frame["interpretation"] == "meaningful_prefix_cache_gain"]
    if matches.empty:
        return None
    if "shared_prefix_ratio" not in matches:
        return matches.iloc[0]
    sortable = matches.copy()
    sortable["shared_prefix_ratio"] = pd.to_numeric(
        sortable["shared_prefix_ratio"], errors="coerce"
    )
    return sortable.sort_values("shared_prefix_ratio", na_position="last").iloc[0]


def _best_interpretation_row(frame: pd.DataFrame, interpretation_column: str) -> pd.Series:
    if interpretation_column not in frame:
        return frame.iloc[0]
    priorities = {
        "quantization_promising": 0,
        "offload_promising": 0,
        "speculative_decoding_promising": 0,
        "disaggregation_promising": 0,
        "decode_regression": 1,
        "quality_regression": 1,
        "latency_regression": 1,
        "memory_telemetry_missing": 2,
        "no_observed_benefit": 3,
    }
    scored = frame.copy()
    scored["_advisor_priority"] = scored[interpretation_column].map(
        lambda value: priorities.get(_clean_string(value), 4)
    )
    return scored.sort_values("_advisor_priority").iloc[0]


def _tradeoff_evidence(row: pd.Series) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for column, label, lower_is_better in [
        ("ttft_delta_pct", "TTFT", True),
        ("tpot_delta_pct", "TPOT", True),
        ("itl_delta_pct", "ITL", True),
        ("e2e_delta_pct", "End-to-end latency", True),
        ("throughput_delta_pct", "Output throughput", False),
        ("memory_delta_pct", "GPU memory", True),
    ]:
        value = _to_float(row.get(column))
        if value is None:
            continue
        improved = value <= 0 if lower_is_better else value >= 0
        direction = "improved" if improved else "regressed"
        if not lower_is_better and direction == "improved":
            message = f"{label} improved by {_fmt_metric(abs(value))}%."
        elif not lower_is_better:
            message = f"{label} regressed by {_fmt_metric(abs(value))}%."
        else:
            message = f"{label} {direction} by {_fmt_metric(abs(value))}%."
        evidence.append(EvidenceItem(message=message, metric=column, value=value))

    quality_delta = _to_float(row.get("quality_delta"))
    if quality_delta is not None:
        direction = "improved" if quality_delta >= 0 else "regressed"
        evidence.append(
            EvidenceItem(
                message=f"Quality {direction} by {_fmt_metric(abs(quality_delta))}.",
                metric="quality_delta",
                value=quality_delta,
            )
        )
    if not evidence:
        evidence.append(
            EvidenceItem(
                message="No comparable latency, throughput, quality, or memory deltas were available."
            )
        )
    return evidence


def _missing_metric_names(row: pd.Series) -> set[str]:
    raw = row.get("missing_metrics")
    if raw is None or pd.isna(raw):
        return set()
    if isinstance(raw, list):
        return {str(item) for item in raw if str(item).strip()}
    return {item.strip() for item in str(raw).split(";") if item.strip()}


def _missing_metric_caveats(row: pd.Series) -> list[str]:
    missing = sorted(_missing_metric_names(row))
    if not missing:
        return []
    return ["Missing metrics: " + ", ".join(missing) + "."]


def _positive_delta_score(row: pd.Series) -> float:
    score = 0.0
    throughput_delta = _to_float(row.get("throughput_delta_pct"))
    latency_delta = _to_float(row.get("e2e_delta_pct"))
    memory_delta = _to_float(row.get("memory_delta_pct"))
    if throughput_delta is not None and throughput_delta >= MEANINGFUL_THROUGHPUT_DELTA_PCT:
        score += min(throughput_delta / 25.0, 2.0)
    if latency_delta is not None and latency_delta <= MEANINGFUL_LATENCY_DELTA_PCT:
        score += min(abs(latency_delta) / 25.0, 2.0)
    if memory_delta is not None and memory_delta <= MEANINGFUL_MEMORY_DELTA_PCT:
        score += min(abs(memory_delta) / 25.0, 2.0)
    return score


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
