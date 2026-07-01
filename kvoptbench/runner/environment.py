"""Environment capture helpers for reproducible benchmark results."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from kvoptbench import __version__
from kvoptbench.schemas import RunEnvironmentSnapshot


TRACKED_PACKAGES = ("kvoptbench", "pydantic", "pandas", "httpx", "fastapi", "uvicorn")
BACKEND_ENVIRONMENT_FIELDS = (
    "engine_version",
    "model_revision",
    "cuda_version",
    "gpu_type",
    "gpu_count",
    "backend_launch_command",
)
_MISSING_BACKEND_ENVIRONMENT_REASONS = {
    "engine_version": "Backend engine version was not provided or exposed.",
    "model_revision": "Model revision was not provided or exposed.",
    "cuda_version": "CUDA version was not provided or exposed.",
    "gpu_type": "GPU type was not provided or exposed.",
    "gpu_count": "GPU count was not provided or exposed.",
    "backend_launch_command": "Backend launch command was not provided.",
}


def capture_environment(
    repo_dir: str | Path | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> RunEnvironmentSnapshot:
    """Capture reproducibility metadata without storing local paths or secrets."""
    repo_path = Path(repo_dir) if repo_dir is not None else Path.cwd()
    metadata = metadata or {}
    return RunEnvironmentSnapshot(
        python_version=platform.python_version(),
        platform=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        kvoptbench_version=__version__,
        git_commit=_git_output(repo_path, "rev-parse", "HEAD"),
        git_branch=_git_output(repo_path, "branch", "--show-current"),
        git_dirty=_git_dirty(repo_path),
        engine_version=_clean_optional(metadata.get("engine_version")),
        model_revision=_clean_optional(metadata.get("model_revision")),
        cuda_version=_clean_optional(metadata.get("cuda_version")),
        gpu_type=_clean_optional(metadata.get("gpu_type")),
        gpu_count=_clean_int(metadata.get("gpu_count")),
        backend_launch_command=_sanitize_launch_command(
            _clean_optional(metadata.get("backend_launch_command"))
        ),
        config_sha256=_clean_optional(metadata.get("config_sha256")),
        workload_sha256=_clean_optional(metadata.get("workload_sha256")),
        package_versions=_package_versions(),
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "engine_version",
                "model_revision",
                "cuda_version",
                "gpu_type",
                "gpu_count",
                "backend_launch_command",
                "config_sha256",
                "workload_sha256",
            }
        },
    )


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sanitize_launch_command(command: str | None) -> str | None:
    """Remove common secret-bearing values from a backend launch command."""
    if command is None:
        return None
    parts = command.split()
    sanitized: list[str] = []
    redact_next = False
    secret_flags = {
        "--api-key",
        "--api_key",
        "--token",
        "--hf-token",
        "--huggingface-token",
        "--password",
    }
    for part in parts:
        lowered = part.lower()
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if any(lowered.startswith(f"{flag}=") for flag in secret_flags):
            flag = part.split("=", 1)[0]
            sanitized.append(f"{flag}=<redacted>")
            continue
        if lowered in secret_flags:
            sanitized.append(part)
            redact_next = True
            continue
        sanitized.append(part)
    return " ".join(sanitized)


def capture_backend_environment(
    metadata: dict[str, Any] | None = None,
    *,
    expected_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Normalize standalone backend metadata and report unavailable fields explicitly."""
    metadata = metadata or {}
    captured = {
        "engine_version": _clean_optional(metadata.get("engine_version")),
        "model_revision": _clean_optional(metadata.get("model_revision")),
        "cuda_version": _clean_optional(metadata.get("cuda_version")),
        "gpu_type": _clean_optional(metadata.get("gpu_type")),
        "gpu_count": _clean_int(metadata.get("gpu_count")),
        "backend_launch_command": _sanitize_launch_command(
            _clean_optional(metadata.get("backend_launch_command"))
        ),
    }
    expected = set(expected_fields or BACKEND_ENVIRONMENT_FIELDS)
    captured["missing_environment_fields"] = [
        {
            "field": field,
            "reason": _MISSING_BACKEND_ENVIRONMENT_REASONS[field],
        }
        for field in BACKEND_ENVIRONMENT_FIELDS
        if field in expected and captured[field] is None
    ]
    return captured


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            if package == "kvoptbench":
                versions[package] = __version__
    return versions


def _git_output(repo_path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output or None


def _git_dirty(repo_path: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return bool(completed.stdout.strip())
