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
    observed = _observed_call(output, tool_calls or [])
    if observed is None:
        return QualityResult(
            quality_score=0.0,
            quality_method="tool_calling",
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
    passed = bool(name_passed and args_passed and not parse_error)
    return QualityResult(
        quality_score=1.0 if passed else 0.0,
        quality_method="tool_calling",
        passed=passed,
        details={
            "source": observed["source"],
            "expected_tool": expected_tool,
            "observed_tool": name,
            "expected_arguments": expected_arguments,
            "observed_arguments": arguments,
            "arguments_parse_error": parse_error,
            "tool_call_count": len(tool_calls or []),
            "name_passed": name_passed,
            "arguments_passed": args_passed,
        },
    )


def _expected_tool_name(item: WorkloadItem) -> str | None:
    value = item.metadata.get("expected_tool") or item.expected_answer
    if value in (None, ""):
        return None
    return str(value)


def _observed_call(
    output: str, tool_calls: list[ToolCallRecord]
) -> dict[str, Any] | None:
    if tool_calls:
        first = tool_calls[0]
        return {
            "source": "tool_calls",
            "name": first.name,
            "arguments": first.arguments,
            "arguments_parse_error": first.arguments_parse_error,
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
