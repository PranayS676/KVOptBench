"""Stable artifact contracts and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from kvoptbench.datasets.hashing import sha256_file
from kvoptbench.datasets.manifest import DatasetManifest
from kvoptbench.packaging.result_package import ResultPackageManifest
from kvoptbench.schemas import RequestResult, SUPPORTED_SCHEMA_VERSIONS
from kvoptbench.strategy.advisor import StrategyAdvisorReport
from kvoptbench.telemetry.runtime import TelemetryRunSummary

CURRENT_SCHEMA_VERSION = "1"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


class SchemaContract(BaseModel):
    """One exported artifact contract."""

    name: str
    file_name: str
    schema_version: str
    json_schema: dict[str, Any]


class SchemaBundle(BaseModel):
    """Collection of exported artifact contracts."""

    schema_version: str
    contracts: list[SchemaContract]


class ArtifactValidationReport(BaseModel):
    """Validation result for one artifact family."""

    ok: bool
    artifact_type: str
    schema_version: str = CURRENT_SCHEMA_VERSION
    checked_files: int = 0
    row_count: int = 0
    errors: list[dict[str, Any]] = []


CONTRACT_MODELS: dict[str, tuple[str, type[BaseModel]]] = {
    "request_result": ("request_result.schema.json", RequestResult),
    "telemetry_run_summary": ("telemetry_run_summary.schema.json", TelemetryRunSummary),
    "strategy_advisor": ("strategy_advisor.schema.json", StrategyAdvisorReport),
    "result_package_manifest": ("result_package_manifest.schema.json", ResultPackageManifest),
    "dataset_manifest": ("dataset_manifest.schema.json", DatasetManifest),
}


def build_schema_bundle() -> SchemaBundle:
    """Build JSON Schema documents from the authoritative Pydantic models."""
    contracts = []
    for name, (file_name, model) in CONTRACT_MODELS.items():
        schema = model.model_json_schema()
        schema["$schema"] = JSON_SCHEMA_DRAFT
        schema["x-kvoptbench-contract"] = name
        schema["x-kvoptbench-schema-version"] = CURRENT_SCHEMA_VERSION
        contracts.append(
            SchemaContract(
                name=name,
                file_name=file_name,
                schema_version=CURRENT_SCHEMA_VERSION,
                json_schema=_sort_json(schema),
            )
        )
    return SchemaBundle(schema_version=CURRENT_SCHEMA_VERSION, contracts=contracts)


def write_schema_files(output_dir: str | Path) -> dict[str, Path]:
    """Write all registered JSON Schema contracts to an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for contract in build_schema_bundle().contracts:
        path = output_path / contract.file_name
        path.write_text(_canonical_json(contract.json_schema), encoding="utf-8")
        written[contract.name] = path
    return written


def check_schema_files(output_dir: str | Path) -> list[str]:
    """Return schema snapshot mismatch messages for a committed schema directory."""
    output_path = Path(output_dir)
    mismatches: list[str] = []
    for contract in build_schema_bundle().contracts:
        path = output_path / contract.file_name
        if not path.exists():
            mismatches.append(f"Missing schema file: {path}")
            continue
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != contract.json_schema:
            mismatches.append(f"Schema file is stale: {path}")
    return mismatches


def validate_result_rows(input_path: str | Path) -> ArtifactValidationReport:
    """Validate request-level JSONL rows using the current result-row contract."""
    errors: list[dict[str, Any]] = []
    row_count = 0
    checked_files = 0
    for file_path in _jsonl_files(Path(input_path)):
        checked_files += 1
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row_count += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(_error(file_path, line_no, f"Invalid JSON: {exc}"))
                    continue
                version_error = _schema_version_error(payload)
                if version_error is not None:
                    errors.append(_error(file_path, line_no, version_error))
                    continue
                try:
                    RequestResult.model_validate(payload)
                except ValidationError as exc:
                    errors.append(_error(file_path, line_no, _validation_message(exc)))
    if checked_files == 0:
        errors.append({"file": str(input_path), "line": None, "message": "No JSONL files found."})
    return ArtifactValidationReport(
        ok=not errors,
        artifact_type="request_results",
        checked_files=checked_files,
        row_count=row_count,
        errors=errors,
    )


def validate_result_package(package_path: str | Path) -> ArtifactValidationReport:
    """Validate a result package manifest and package-relative artifact hashes."""
    root = Path(package_path)
    errors: list[dict[str, Any]] = []
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        return ArtifactValidationReport(
            ok=False,
            artifact_type="result_package",
            errors=[
                {
                    "file": manifest_path.as_posix(),
                    "line": None,
                    "message": "Missing result package run_manifest.json.",
                }
            ],
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ArtifactValidationReport(
            ok=False,
            artifact_type="result_package",
            errors=[_error(manifest_path, None, f"Invalid JSON: {exc}")],
        )
    version_error = _schema_version_error(payload)
    if version_error is not None:
        errors.append(_error(manifest_path, None, version_error))
    try:
        manifest = ResultPackageManifest.model_validate(payload)
    except ValidationError as exc:
        errors.append(_error(manifest_path, None, _validation_message(exc)))
        return ArtifactValidationReport(
            ok=False,
            artifact_type="result_package",
            checked_files=1,
            errors=errors,
        )

    checked_files = 1
    for artifact in manifest.artifacts:
        artifact_path = root / artifact.path
        checked_files += 1
        if not artifact_path.exists():
            errors.append(
                _error(
                    artifact_path,
                    None,
                    f"Package artifact is missing: {artifact.path}",
                )
            )
            continue
        actual_hash = sha256_file(artifact_path)
        if actual_hash != artifact.sha256:
            errors.append(
                _error(
                    artifact_path,
                    None,
                    f"Artifact sha256 mismatch for {artifact.path}.",
                )
            )
    return ArtifactValidationReport(
        ok=not errors,
        artifact_type="result_package",
        checked_files=checked_files,
        row_count=int(manifest.summary.get("requests") or 0),
        errors=errors,
    )


def _jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.jsonl")) if input_path.exists() else []


def _schema_version_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "Artifact payload must be a JSON object."
    version = str(payload.get("schema_version", CURRENT_SCHEMA_VERSION))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        valid = ", ".join(SUPPORTED_SCHEMA_VERSIONS)
        return f"Unsupported schema_version '{version}'. Supported versions: {valid}."
    return None


def _error(file_path: Path, line: int | None, message: str) -> dict[str, Any]:
    return {"file": file_path.as_posix(), "line": line, "message": message}


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
    return f"{loc}: {first.get('msg', 'validation failed')}"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(_sort_json(payload), indent=2, sort_keys=True) + "\n"


def _sort_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_json(item) for item in value]
    return value
