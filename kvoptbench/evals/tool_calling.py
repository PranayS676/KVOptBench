"""Tool-call quality evaluator."""

from __future__ import annotations

import json
from typing import Any

from kvoptbench.schemas import QualityResult, ToolCallRecord, WorkloadItem


def evaluate_tool_calling(
    output: str,
    item: WorkloadItem,
    tool_calls: list[ToolCallRecord] | None = None,
) -> QualityResult:
    """Evaluate structured OpenAI tool calls or JSON fallback tool-call text."""
    expected_tool = _expected_tool_name(item)
    expected_arguments = item.metadata.get("expected_arguments")
    required_arguments = _required_arguments(item)
    quality_method = "bfcl_tool_call" if item.eval_type == "bfcl_tool_call" else "tool_calling"
    observed = _observed_call(output, tool_calls or [])
    if observed is None:
        return QualityResult(
            quality_score=0.0,
            quality_method=quality_method,
            passed=False,
            details={
                "expected_tool": expected_tool,
                "tool_call_count": len(tool_calls or []),
                "error": "no_tool_call",
            },
        )

    name = observed["name"]
    arguments = observed["arguments"]
    parse_error = observed.get("arguments_parse_error")
    name_passed = expected_tool is None or name == expected_tool
    args_passed = _arguments_match(arguments, expected_arguments)
    missing_required = _missing_required_arguments(arguments, required_arguments)
    required_passed = not missing_required
    passed = bool(name_passed and args_passed and required_passed and not parse_error)
    return QualityResult(
        quality_score=1.0 if passed else 0.0,
        quality_method=quality_method,
        passed=passed,
        details={
            "source": observed["source"],
            "expected_tool": expected_tool,
            "observed_tool": name,
            "expected_arguments": expected_arguments,
            "observed_arguments": arguments,
            "required_arguments": required_arguments,
            "missing_required_arguments": missing_required,
            "arguments_parse_error": parse_error,
            "tool_call_count": len(tool_calls or []),
            "name_passed": name_passed,
            "arguments_passed": args_passed,
            "required_arguments_passed": required_passed,
        },
    )


def _expected_tool_name(item: WorkloadItem) -> str | None:
    value = (
        item.metadata.get("expected_function_name")
        or item.metadata.get("expected_tool")
        or item.metadata.get("expected_tool_name")
        or item.expected_answer
    )
    if value in (None, ""):
        return None
    return str(value)


def _observed_call(
    output: str, tool_calls: list[ToolCallRecord]
) -> dict[str, Any] | None:
    if tool_calls:
        first = tool_calls[0]
        arguments = first.arguments
        parse_error = first.arguments_parse_error
        if arguments is None and first.arguments_json:
            try:
                arguments = json.loads(first.arguments_json)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
        return {
            "source": "tool_calls",
            "name": first.name,
            "arguments": arguments,
            "arguments_parse_error": parse_error,
        }

    if not output.strip():
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return {
            "source": "visible_json",
            "name": None,
            "arguments": None,
            "arguments_parse_error": str(exc),
        }
    if not isinstance(parsed, dict):
        return {
            "source": "visible_json",
            "name": None,
            "arguments": None,
            "arguments_parse_error": "tool call JSON must be an object",
        }
    if isinstance(parsed.get("tool_calls"), list) and parsed["tool_calls"]:
        parsed = parsed["tool_calls"][0]
    function = parsed.get("function")
    if isinstance(function, dict):
        arguments = function.get("arguments")
        parse_error = None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
        return {
            "source": "visible_json",
            "name": function.get("name"),
            "arguments": arguments,
            "arguments_parse_error": parse_error,
        }
    return {
        "source": "visible_json",
        "name": parsed.get("tool") or parsed.get("name") or parsed.get("function"),
        "arguments": parsed.get("arguments"),
        "arguments_parse_error": None,
    }


def _arguments_match(arguments: Any, expected_arguments: Any) -> bool:
    if expected_arguments in (None, {}):
        return True
    if not isinstance(expected_arguments, dict):
        return arguments == expected_arguments
    if not isinstance(arguments, dict):
        return False
    for key, expected_value in expected_arguments.items():
        if arguments.get(key) != expected_value:
            return False
    return True


def _required_arguments(item: WorkloadItem) -> list[str]:
    raw = (
        item.metadata.get("required_arguments")
        or item.metadata.get("required_argument_names")
        or item.metadata.get("required_fields")
    )
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(value) for value in raw if str(value).strip()]
    return []


def _missing_required_arguments(arguments: Any, required_arguments: list[str]) -> list[str]:
    if not required_arguments:
        return []
    if not isinstance(arguments, dict):
        return required_arguments
    return [key for key in required_arguments if key not in arguments]
