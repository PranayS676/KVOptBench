from kvoptbench.evals.exact_match import contains_expected_answer, exact_match
from kvoptbench.evals.json_validity import evaluate_json_validity
from kvoptbench.evals.needle import evaluate_needle
from kvoptbench.schemas import WorkloadItem


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

