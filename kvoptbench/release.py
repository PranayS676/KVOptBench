"""Release-readiness checks for public KVOptBench branches."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from kvoptbench import __version__
from kvoptbench.contracts import check_schema_files
from kvoptbench.strategy.advisor import load_workload_thresholds
from kvoptbench.telemetry.profiles import load_telemetry_profiles

ReleaseStatus = Literal["ok", "fail"]


class ReleaseCheck(BaseModel):
    """One release-readiness check result."""

    name: str
    status: ReleaseStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ReleaseCheckReport(BaseModel):
    """Release-readiness report for a repository checkout."""

    ok: bool
    repo_dir: str
    checks: list[ReleaseCheck]


PUBLIC_TEXT_FILES = [
    "README.md",
    "ROADMAP.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "guides/benchmark_validity.md",
    "guides/metric_provenance.md",
    "guides/reproducibility.md",
    "guides/real_endpoint_vllm_sglang.md",
    "guides/runpod.md",
    "guides/first_real_benchmark.md",
    "guides/datasets.md",
    "guides/dataset_adapter_contract.md",
    "guides/frontier_dataset_pack.md",
    "examples/public_release/result_template.md",
    "examples/public_release/blog_report_template.md",
]

REQUIRED_PUBLIC_FILES = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "assets/kvoptbench-readme-hero.png",
    "schemas/v1/request_result.schema.json",
    "schemas/v1/telemetry_run_summary.schema.json",
    "schemas/v1/strategy_advisor.schema.json",
    "examples/example_experiment_config.yaml",
    "examples/vllm_openai_compatible_config.yaml",
    "examples/sglang_openai_compatible_config.yaml",
    "guides/benchmark_validity.md",
    "guides/metric_provenance.md",
]

INTERNAL_MARKERS = [
    "Milestone",
    "C:\\Users",
    "OneDrive",
    "YOUR_USERNAME",
    "KVOptBench_Strategic_Direction_Memo.docx",
]


def run_release_check(repo_dir: str | Path = ".") -> ReleaseCheckReport:
    """Run lightweight release-readiness checks without invoking network or GPUs."""
    root = Path(repo_dir)
    checks = [
        _check_version_consistency(root),
        _check_schema_snapshots(root),
        _check_bundled_resources(),
        _check_public_files(root),
        _check_public_text_safety(root),
    ]
    return ReleaseCheckReport(
        ok=all(check.status == "ok" for check in checks),
        repo_dir=root.resolve().as_posix(),
        checks=checks,
    )


def _check_version_consistency(root: Path) -> ReleaseCheck:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return _fail("version_consistency", "pyproject.toml is missing.")
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project_version = str(payload.get("project", {}).get("version", ""))
    if project_version != __version__:
        return _fail(
            "version_consistency",
            "pyproject.toml version does not match kvoptbench.__version__.",
            {"pyproject_version": project_version, "package_version": __version__},
        )
    return _ok(
        "version_consistency",
        f"Package version is consistent at {__version__}.",
        {"version": __version__},
    )


def _check_schema_snapshots(root: Path) -> ReleaseCheck:
    schema_dir = root / "schemas" / "v1"
    mismatches = check_schema_files(schema_dir)
    if mismatches:
        return _fail(
            "schema_snapshots",
            "Schema snapshots are missing or stale.",
            {"mismatches": mismatches},
        )
    return _ok("schema_snapshots", "Schema snapshots are current.")


def _check_bundled_resources() -> ReleaseCheck:
    try:
        profiles = load_telemetry_profiles()
        advisor_thresholds = load_workload_thresholds()
    except (OSError, ValueError) as exc:
        return _fail("bundled_resources", f"Bundled YAML resources failed to load: {exc}")
    if not profiles:
        return _fail("bundled_resources", "No telemetry profiles are bundled.")
    if not advisor_thresholds:
        return _fail("bundled_resources", "No advisor workload thresholds are bundled.")
    return _ok(
        "bundled_resources",
        "Bundled telemetry profiles and advisor thresholds load successfully.",
        {
            "telemetry_profiles": sorted(profiles),
            "advisor_profiles": sorted(advisor_thresholds),
        },
    )


def _check_public_files(root: Path) -> ReleaseCheck:
    missing = [path for path in REQUIRED_PUBLIC_FILES if not (root / path).exists()]
    if missing:
        return _fail("public_files", "Required public files are missing.", {"missing": missing})
    return _ok("public_files", "Required public files are present.")


def _check_public_text_safety(root: Path) -> ReleaseCheck:
    findings: list[dict[str, Any]] = []
    for relative_path in PUBLIC_TEXT_FILES:
        path = root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in INTERNAL_MARKERS:
            if marker in text:
                findings.append({"file": relative_path, "marker": marker})
    if findings:
        return _fail(
            "public_text_safety",
            "Public-facing text contains internal placeholders or local paths.",
            {"findings": findings},
        )
    return _ok("public_text_safety", "Public-facing text has no internal placeholders.")


def _ok(name: str, message: str, details: dict[str, Any] | None = None) -> ReleaseCheck:
    return ReleaseCheck(name=name, status="ok", message=message, details=details or {})


def _fail(name: str, message: str, details: dict[str, Any] | None = None) -> ReleaseCheck:
    return ReleaseCheck(name=name, status="fail", message=message, details=details or {})
