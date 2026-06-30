from kvoptbench.evals.exact_match import contains_expected_answer, exact_match
from kvoptbench.evals.json_validity import evaluate_json_validity
from kvoptbench.evals.needle import evaluate_needle
from kvoptbench.evals.tool_calling import evaluate_tool_calling
from kvoptbench.schemas import ToolCallRecord, WorkloadItem


def test_json_validity_evaluator_scores_valid_json() -> None:
    result = evaluate_json_validity('{"answer": "ok"}')

    assert result.quality_score == 1.0
    assert result.passed is True
    assert result.quality_method == "json_validity"


def test_json_validity_evaluator_scores_invalid_json() -> None:
    result = evaluate_json_validity("{not-json")

    assert result.quality_score == 0.0
    assert result.passed is False
    assert "error" in result.details


def test_exact_and_contains_evaluators() -> None:
    assert exact_match("Answer", " answer ").quality_score == 1.0
    assert contains_expected_answer("The needle is alpha-123.", "alpha-123").quality_score == 1.0
    assert contains_expected_answer("No match.", "alpha-123").quality_score == 0.0


def test_needle_evaluator_uses_workload_expected_answer() -> None:
    item = WorkloadItem(
        task_id="needle_001",
        workload="long_context_needle",
        category="long_context",
        prompt="Find the needle.",
        expected_answer="needle-code-42",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="needle",
    )

    result = evaluate_needle("The answer is needle-code-42.", item)

    assert result.quality_score == 1.0
    assert result.passed is True


def test_tool_calling_evaluator_scores_structured_tool_call() -> None:
    item = WorkloadItem(
        task_id="tool_001",
        workload="tool_calling",
        category="structured_output",
        prompt="Use the lookup_order tool.",
        expected_answer="lookup_order",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="tool_calling",
        metadata={"expected_arguments": {"order_id": "A123"}},
    )
    tool_calls = [
        ToolCallRecord(
            id="call_1",
            type="function",
            name="lookup_order",
            arguments={"order_id": "A123"},
            arguments_json='{"order_id": "A123"}',
        )
    ]

    result = evaluate_tool_calling("", item, tool_calls=tool_calls)

    assert result.quality_score == 1.0
    assert result.passed is True
    assert result.quality_method == "tool_calling"
    assert result.details["tool_call_count"] == 1


def test_tool_calling_evaluator_reports_wrong_tool_name() -> None:
    item = WorkloadItem(
        task_id="tool_002",
        workload="tool_calling",
        category="structured_output",
        prompt="Use the lookup_order tool.",
        expected_answer="lookup_order",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="tool_calling",
    )
    tool_calls = [
        ToolCallRecord(
            id="call_1",
            type="function",
            name="create_ticket",
            arguments={},
            arguments_json="{}",
        )
    ]

    result = evaluate_tool_calling("", item, tool_calls=tool_calls)

    assert result.quality_score == 0.0
    assert result.passed is False
    assert result.details["observed_tool"] == "create_ticket"


def test_tool_calling_evaluator_accepts_visible_json_fallback() -> None:
    item = WorkloadItem(
        task_id="tool_003",
        workload="tool_calling",
        category="structured_output",
        prompt="Return a JSON tool call.",
        expected_answer="lookup_order",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="tool_calling",
        metadata={"expected_arguments": {"order_id": "A123"}},
    )

    result = evaluate_tool_calling(
        '{"tool": "lookup_order", "arguments": {"order_id": "A123"}}',
        item,
    )

    assert result.quality_score == 1.0
    assert result.passed is True
    assert result.details["source"] == "visible_json"

