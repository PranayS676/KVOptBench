from kvoptbench.evals.dispatch import evaluate_output
from kvoptbench.evals.exact_match import contains_expected_answer, exact_match
from kvoptbench.evals.json_validity import evaluate_json_validity
from kvoptbench.evals.needle import evaluate_needle
from kvoptbench.evals.rag import evaluate_rag
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


def test_json_schema_evaluator_enforces_required_fields_and_extra_keys() -> None:
    schema = {
        "type": "object",
        "required": ["answer", "confidence"],
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    passed = evaluate_json_validity('{"answer": "ok", "confidence": 0.9}', schema)
    missing = evaluate_json_validity('{"answer": "ok"}', schema)
    extra = evaluate_json_validity('{"answer": "ok", "confidence": 0.9, "debug": true}', schema)

    assert passed.quality_method == "json_schema"
    assert passed.quality_score == 1.0
    assert missing.quality_score == 0.0
    assert missing.details["errors"] == ["missing required field: confidence"]
    assert extra.details["errors"] == ["unexpected field: debug"]


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


def test_qasper_answer_scoring_uses_expected_answer_aliases() -> None:
    item = WorkloadItem(
        task_id="qasper_001",
        workload="qasper_shared_prefix",
        category="prefix_cache",
        prompt="Answer using the paper.",
        expected_answer="KV cache blocks",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="qasper_answer",
        metadata={"expected_answers": ["attention cache blocks", "KV cache blocks"]},
    )

    result = evaluate_output("The paper says KV cache blocks are reused.", item)

    assert result.quality_method == "qasper_answer"
    assert result.quality_score == 1.0
    assert result.passed is True
    assert result.details["best_expected_answer"] == "KV cache blocks"


def test_longbench_answer_scoring_gives_partial_credit_with_token_f1() -> None:
    item = WorkloadItem(
        task_id="longbench_001",
        workload="long_context_qa",
        category="long_context",
        prompt="Answer using the document.",
        expected_answer="the cache miss forces a full prefill",
        target_input_tokens=4096,
        target_output_tokens=64,
        eval_type="longbench_answer",
    )

    result = evaluate_output("A cache miss triggers full prefill work.", item)

    assert result.quality_method == "longbench_answer"
    assert 0.0 < result.quality_score < 1.0
    assert result.details["best_token_f1"] > 0.0


def test_rag_evaluator_requires_expected_source_id_when_present() -> None:
    item = WorkloadItem(
        task_id="rag_001",
        workload="rag",
        category="rag",
        prompt="Answer with citation.",
        expected_answer="prefix caching reduced TTFT",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="rag",
        metadata={"expected_source_ids": ["doc-cache-17"]},
    )

    passed = evaluate_rag("prefix caching reduced TTFT [doc-cache-17]", item)
    missing_source = evaluate_rag("prefix caching reduced TTFT [doc-other]", item)

    assert passed.quality_method == "rag_source_match"
    assert passed.quality_score == 1.0
    assert passed.details["source_recall"] == 1.0
    assert missing_source.quality_score == 0.5
    assert missing_source.passed is False
    assert missing_source.details["missing_source_ids"] == ["doc-cache-17"]


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


def test_bfcl_tool_calling_evaluator_checks_function_and_required_fields() -> None:
    item = WorkloadItem(
        task_id="bfcl_001",
        workload="tool_calling",
        category="structured_output",
        prompt="Call get_weather.",
        target_input_tokens=100,
        target_output_tokens=20,
        eval_type="bfcl_tool_call",
        metadata={
            "expected_function_name": "get_weather",
            "required_arguments": ["location", "unit"],
        },
    )

    result = evaluate_tool_calling(
        '{"function": {"name": "get_weather", "arguments": {"location": "Austin"}}}',
        item,
    )

    assert result.quality_method == "bfcl_tool_call"
    assert result.quality_score == 0.0
    assert result.passed is False
    assert result.details["observed_tool"] == "get_weather"
    assert result.details["missing_required_arguments"] == ["unit"]

