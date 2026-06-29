"""Evaluator dispatch."""

from __future__ import annotations

from kvoptbench.evals.exact_match import contains_expected_answer, exact_match
from kvoptbench.evals.json_validity import evaluate_json_validity
from kvoptbench.evals.llm_judge import evaluate_llm_judge
from kvoptbench.evals.needle import evaluate_needle
from kvoptbench.evals.rag import evaluate_rag
from kvoptbench.evals.tool_calling import evaluate_tool_calling
from kvoptbench.schemas import QualityResult, WorkloadItem


def evaluate_output(output: str, item: WorkloadItem) -> QualityResult:
    if item.eval_type == "json_validity":
        return evaluate_json_validity(output)
    if item.eval_type == "exact_match":
        return exact_match(output, item.expected_answer)
    if item.eval_type == "needle":
        return evaluate_needle(output, item)
    if item.eval_type == "tool_calling_placeholder":
        return evaluate_tool_calling(output, item)
    if item.eval_type == "rag_placeholder":
        return evaluate_rag(output, item)
    if item.eval_type == "llm_judge_placeholder":
        return evaluate_llm_judge(output, item)
    return contains_expected_answer(output, item.expected_answer)

