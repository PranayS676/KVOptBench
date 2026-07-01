"""Markdown rendering for strategy advisor reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kvoptbench.strategy.advisor import StrategyAdvisorReport, StrategyRecommendation


def render_strategy_advisor_markdown(report: "StrategyAdvisorReport") -> str:
    """Render a human-readable strategy advisor report."""
    lines = [
        "# Strategy Advisor",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Overall recommendation: `{report.overall_recommendation}`",
        "",
    ]

    for decision, heading in [
        ("recommend", "Recommended"),
        ("consider", "Consider"),
        ("do_not_recommend", "Do Not Recommend"),
        ("inconclusive", "Inconclusive"),
        ("needs_more_data", "Needs More Data"),
    ]:
        _append_group(lines, report.recommendations, decision, heading)

    return "\n".join(lines) + "\n"


def _append_group(
    lines: list[str],
    recommendations: list["StrategyRecommendation"],
    decision: str,
    heading: str,
) -> None:
    matches = [item for item in recommendations if item.decision == decision]
    if not matches:
        return
    lines.extend([f"## {heading}", ""])
    for item in matches:
        lines.append(
            f"{item.rank}. `{item.strategy}` - `{item.decision}` "
            f"(confidence: `{item.confidence}`, confidence score: `{item.confidence_score:.2f}`)"
        )
        if item.quality_guardrail:
            lines.append(f"   Quality guardrail: `{item.quality_guardrail}`")
        if item.quality_gate_status:
            lines.append(f"   Workload quality gate: `{item.quality_gate_status}`")
        if item.workload_profile:
            lines.append(f"   Workload profile: `{item.workload_profile}`")
        if item.workload_threshold:
            lines.append(f"   Threshold posture: {item.workload_threshold.threshold_posture}")
        if item.missing_required_metrics:
            lines.append(
                "   Missing required metrics: "
                + ", ".join(f"`{metric}`" for metric in item.missing_required_metrics)
            )
        if item.missing_recommended_metrics:
            lines.append(
                "   Missing recommended metrics: "
                + ", ".join(f"`{metric}`" for metric in item.missing_recommended_metrics)
            )
        if item.confidence_reasons:
            lines.append("   Confidence rationale:")
            for reason in item.confidence_reasons:
                lines.append(f"   - {reason}")
        if item.evidence:
            lines.append("   Evidence:")
            for evidence in item.evidence:
                lines.append(f"   - {evidence.message}")
        if item.caveats:
            lines.append("   Caveats:")
            for caveat in item.caveats:
                lines.append(f"   - {caveat}")
        if item.next_experiments:
            lines.append("   Next experiments:")
            for next_experiment in item.next_experiments:
                lines.append(f"   - {next_experiment}")
        if item.next_experiment_priority:
            lines.append("   Next experiment priority:")
            for next_experiment in item.next_experiment_priority:
                lines.append(f"   - {next_experiment}")
        if item.next_experiment_plans:
            lines.append("   Next experiment command plans:")
            for plan in item.next_experiment_plans:
                lines.append(f"   - Reason: `{plan.reason}`")
                lines.append(f"     Command: `{plan.command}`")
                if plan.options.get("required_metrics"):
                    lines.append(
                        "     Required metrics: "
                        + ", ".join(f"`{metric}`" for metric in plan.options["required_metrics"])
                    )
                if plan.options.get("required_quality_evaluators"):
                    lines.append(
                        "     Required evaluators: "
                        + ", ".join(
                            f"`{evaluator}`"
                            for evaluator in plan.options["required_quality_evaluators"]
                        )
                    )
        lines.append("")
