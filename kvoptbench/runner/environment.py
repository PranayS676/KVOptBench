"""Environment capture helpers for reproducible benchmark results."""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from pathlib import Path

from kvoptbench import __version__
from kvoptbench.schemas import RunEnvironmentSnapshot


TRACKED_PACKAGES = ("kvoptbench", "pydantic", "pandas", "httpx", "fastapi", "uvicorn")


def capture_environment(repo_dir: str | Path | None = None) -> RunEnvironmentSnapshot:
    """Capture reproducibility metadata without storing local paths or secrets."""
    repo_path = Path(repo_dir) if repo_dir is not None else Path.cwd()
    return RunEnvironmentSnapshot(
        python_version=platform.python_version(),
        platform=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        kvoptbench_version=__version__,
        git_commit=_git_output(repo_path, "rev-parse", "HEAD"),
        git_branch=_git_output(repo_path, "branch", "--show-current"),
        git_dirty=_git_dirty(repo_path),
        package_versions=_package_versions(),
    )


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
