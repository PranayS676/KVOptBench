"""Evaluator dispatch."""

from __future__ import annotations

from kvoptbench.evals.answer_scoring import evaluate_answer
from kvoptbench.evals.exact_match import contains_expected_answer, exact_match
from kvoptbench.evals.json_validity import evaluate_json_validity
from kvoptbench.evals.llm_judge import evaluate_llm_judge
from kvoptbench.evals.needle import evaluate_needle
from kvoptbench.evals.rag import evaluate_rag
from kvoptbench.evals.tool_calling import evaluate_tool_calling
from kvoptbench.schemas import QualityResult, ToolCallRecord, WorkloadItem


def evaluate_output(
    output: str, item: WorkloadItem, tool_calls: list[ToolCallRecord] | None = None
) -> QualityResult:
    if item.eval_type in {"json_validity", "json_schema"}:
        return evaluate_json_validity(output, item.expected_schema)
    if item.eval_type == "qasper_answer":
        return evaluate_answer(output, item, method="qasper_answer")
    if item.eval_type in {"longbench_answer", "longbench_qa"}:
        return evaluate_answer(output, item, method="longbench_answer")
    if item.eval_type == "exact_match":
        return exact_match(output, item.expected_answer)
    if item.eval_type == "needle":
        return evaluate_needle(output, item)
    if item.eval_type in {"tool_calling", "tool_calling_placeholder", "bfcl_tool_call"}:
        return evaluate_tool_calling(output, item, tool_calls=tool_calls)
    if item.eval_type in {"rag", "rag_placeholder", "rag_source_match"}:
        return evaluate_rag(output, item)
    if item.eval_type == "llm_judge_placeholder":
        return evaluate_llm_judge(output, item)
    return contains_expected_answer(output, item.expected_answer)

