"""Preflight checks for KVOptBench experiment configs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from kvoptbench.client.openai_compat import OpenAICompatClient
from kvoptbench.config import ConfigError, validate_config
from kvoptbench.datasets.hashing import sha256_file
from kvoptbench.runner.environment import capture_environment
from kvoptbench.runner.experiment import load_workload
from kvoptbench.schemas import ExperimentConfig

CheckStatus = Literal["ok", "warn", "fail", "skipped"]


class DoctorCheck(BaseModel):
    """One preflight check result."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class DoctorReport(BaseModel):
    """Full doctor report for a config."""

    ok: bool
    config_path: Path
    checks: list[DoctorCheck]


def run_doctor(
    config_path: str | Path,
    *,
    check_endpoint: bool = True,
    repo_dir: str | Path | None = None,
) -> DoctorReport:
    """Validate config, workload, dataset manifest, endpoint, telemetry, and environment."""
    path = Path(config_path)
    checks: list[DoctorCheck] = []
    config: ExperimentConfig | None = None

    try:
        config = validate_config(path)
    except ConfigError as exc:
        checks.append(
            DoctorCheck(name="config", status="fail", message=str(exc), details={})
        )
        _append_skipped_after_config_failure(checks)
        return _report(path, checks)

    checks.append(
        DoctorCheck(
            name="config",
            status="ok",
            message=f"Loaded experiment config '{config.experiment_id}'.",
            details={"experiment_id": config.experiment_id},
        )
    )
    checks.append(_check_workload(config, path))
    checks.append(_check_dataset_manifest(config, path))
    checks.append(_check_telemetry(config))
    checks.append(_check_endpoint(config, check_endpoint=check_endpoint))
    checks.append(_check_environment(config, path, repo_dir=repo_dir))
    return _report(path, checks)


def _append_skipped_after_config_failure(checks: list[DoctorCheck]) -> None:
    for name in ["workload", "dataset_manifest", "telemetry", "endpoint", "environment"]:
        checks.append(
            DoctorCheck(
                name=name,
                status="skipped",
                message="Skipped because config validation failed.",
            )
        )


def _check_workload(config: ExperimentConfig, config_path: Path) -> DoctorCheck:
    workload_path = _resolve_path(config.workload_file, config_path)
    if not workload_path.exists():
        return DoctorCheck(
            name="workload",
            status="fail",
            message=f"Workload file does not exist: {workload_path}",
            details={"path": workload_path.as_posix()},
        )
    try:
        rows = load_workload(workload_path)
    except ValueError as exc:
        return DoctorCheck(
            name="workload",
            status="fail",
            message=str(exc),
            details={"path": workload_path.as_posix()},
        )
    return DoctorCheck(
        name="workload",
        status="ok",
        message=f"Loaded {len(rows)} workload rows.",
        details={"path": workload_path.as_posix(), "row_count": len(rows)},
    )


def _check_dataset_manifest(config: ExperimentConfig, config_path: Path) -> DoctorCheck:
    raw_path = config.metadata.get("dataset_manifest") or config.metadata.get(
        "dataset_manifest_path"
    )
    if not raw_path:
        return DoctorCheck(
            name="dataset_manifest",
            status="warn",
            message="No dataset manifest path was configured in metadata.",
        )
    manifest_path = _resolve_path(Path(str(raw_path)), config_path)
    if not manifest_path.exists():
        return DoctorCheck(
            name="dataset_manifest",
            status="fail",
            message=f"Dataset manifest does not exist: {manifest_path}",
            details={"path": manifest_path.as_posix()},
        )
    return DoctorCheck(
        name="dataset_manifest",
        status="ok",
        message="Dataset manifest exists.",
        details={"path": manifest_path.as_posix()},
    )


def _check_telemetry(config: ExperimentConfig) -> DoctorCheck:
    telemetry = config.telemetry
    if telemetry is None or not telemetry.enabled:
        return DoctorCheck(
            name="telemetry",
            status="ok",
            message="Telemetry is disabled; missing backend metrics will remain null.",
        )
    details: dict[str, Any] = {
        "prometheus_sources": len(telemetry.prometheus),
        "lmcache_sources": len(telemetry.lmcache),
        "gpu_enabled": telemetry.gpu.enabled if telemetry.gpu is not None else False,
    }
    if telemetry.output_dir is None:
        return DoctorCheck(
            name="telemetry",
            status="warn",
            message="Telemetry is enabled without an explicit output_dir.",
            details=details,
        )
    return DoctorCheck(
        name="telemetry",
        status="ok",
        message="Telemetry config is valid.",
        details={**details, "output_dir": telemetry.output_dir.as_posix()},
    )


def _check_endpoint(config: ExperimentConfig, *, check_endpoint: bool) -> DoctorCheck:
    if not check_endpoint:
        return DoctorCheck(
            name="endpoint",
            status="skipped",
            message="Endpoint healthcheck skipped by user request.",
        )
    try:
        health = asyncio.run(OpenAICompatClient(config).healthcheck())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            health = loop.run_until_complete(OpenAICompatClient(config).healthcheck())
        finally:
            loop.close()
    if not health.ok:
        return DoctorCheck(
            name="endpoint",
            status="fail",
            message=health.error_message or "Endpoint healthcheck failed.",
            details=health.model_dump(mode="json"),
        )
    return DoctorCheck(
        name="endpoint",
        status="ok",
        message="Endpoint healthcheck succeeded.",
        details=health.model_dump(mode="json"),
    )


def _check_environment(
    config: ExperimentConfig,
    config_path: Path,
    *,
    repo_dir: str | Path | None,
) -> DoctorCheck:
    metadata = {
        "engine_version": config.engine_version,
        "model_revision": config.model_revision,
        "cuda_version": config.cuda_version,
        "gpu_type": config.gpu_type,
        "gpu_count": config.gpu_count,
        "backend_launch_command": config.backend_launch_command,
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
    }
    workload_path = _resolve_path(config.workload_file, config_path)
    if workload_path.exists():
        metadata["workload_sha256"] = sha256_file(workload_path)
    environment = capture_environment(repo_dir or Path.cwd(), metadata=metadata)
    return DoctorCheck(
        name="environment",
        status="ok",
        message="Captured local reproducibility metadata.",
        details=environment.model_dump(mode="json"),
    )


def _resolve_path(path: Path, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    config_relative = config_path.parent / candidate
    if config_relative.exists():
        return config_relative
    return candidate


def _report(config_path: Path, checks: list[DoctorCheck]) -> DoctorReport:
    return DoctorReport(
        ok=not any(check.status == "fail" for check in checks),
        config_path=config_path,
        checks=checks,
    )
