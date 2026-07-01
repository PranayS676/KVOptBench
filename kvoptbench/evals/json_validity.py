"""JSON validity evaluator."""

from __future__ import annotations

import json
from typing import Any

from kvoptbench.schemas import QualityResult


def evaluate_json_validity(output: str, expected_schema: dict[str, Any] | None = None) -> QualityResult:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return QualityResult(
            quality_score=0.0,
            quality_method="json_schema" if expected_schema else "json_validity",
            passed=False,
            details={"error": str(exc)},
        )
    if expected_schema:
        errors = _validate_json_schema(parsed, expected_schema)
        return QualityResult(
            quality_score=0.0 if errors else 1.0,
            quality_method="json_schema",
            passed=not errors,
            details={"errors": errors},
        )
    return QualityResult(
        quality_score=1.0,
        quality_method="json_validity",
        passed=True,
        details={},
    )


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, str(expected_type)):
        errors.append(f"{path or '<root>'}: expected {expected_type}")
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path or '<root>'}: value is not in enum")

    if expected_type == "object" or isinstance(value, dict):
        if not isinstance(value, dict):
            return errors
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for field in required:
            if field not in value:
                errors.append(f"missing required field: {_join_path(path, str(field))}")
        if schema.get("additionalProperties") is False:
            for field in sorted(value):
                if field not in properties:
                    errors.append(f"unexpected field: {_join_path(path, str(field))}")
        for field, field_schema in properties.items():
            if field in value and isinstance(field_schema, dict):
                errors.extend(
                    _validate_json_schema(value[field], field_schema, _join_path(path, field))
                )
    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_json_schema(item, item_schema, f"{path}[{index}]"))
    return errors


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _join_path(prefix: str, field: str) -> str:
    return f"{prefix}.{field}" if prefix else field

