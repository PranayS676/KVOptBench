"""Build reproducible KVOptBench result packages."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kvoptbench import __version__
from kvoptbench.datasets.hashing import sha256_file
from kvoptbench.schemas import utc_now_iso


MISSING_METRIC_REASON = (
    "Metric was reported unavailable by the input artifacts; KVOptBench stores nulls "
    "instead of fabricating unavailable engine telemetry."
)


class PackageArtifact(BaseModel):
    """One file copied or generated inside a result package."""

    role: str
    path: str
    sha256: str
    size_bytes: int
    original_name: str


class MissingMetricEntry(BaseModel):
    """One unavailable metric documented for a packaged run."""

    metric: str
    reason: str = MISSING_METRIC_REASON


class ResultPackageManifest(BaseModel):
    """Top-level manifest for a publishable result package."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1"
    package_version: str = "1"
    kvoptbench_version: str = __version__
    created_at: str = Field(default_factory=utc_now_iso)
    run_name: str
    artifact_count: int
    summary: dict[str, Any]
    artifacts: list[PackageArtifact]
    workload_provenance: list[dict[str, Any]]
    dataset_provenance: list[dict[str, Any]]
    missing_metrics: list[MissingMetricEntry]
    metric_provenance: dict[str, Any]
    limitations: list[str]
    reproduction_notes: list[str]


class ResultPackageBuild(BaseModel):
    """Paths produced by result package generation."""

    output_dir: Path
    manifest_path: Path
    missing_metrics_path: Path
    metric_provenance_path: Path
    readme_path: Path
    artifact_count: int


def build_result_package(
    *,
    output_dir: str | Path,
    summary_path: str | Path,
    raw_input_paths: list[str | Path] | None = None,
    workload_paths: list[str | Path] | None = None,
    dataset_manifest_paths: list[str | Path] | None = None,
    report_paths: list[str | Path] | None = None,
    config_paths: list[str | Path] | None = None,
    extra_artifact_paths: list[str | Path] | None = None,
    sample_rows: int = 3,
    run_name: str | None = None,
) -> ResultPackageBuild:
    """Create a reproducibility package around completed benchmark artifacts."""
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)
    raw_input_paths = _to_paths(raw_input_paths)
    workload_paths = _to_paths(workload_paths)
    dataset_manifest_paths = _to_paths(dataset_manifest_paths)
    report_paths = _to_paths(report_paths)
    config_paths = _to_paths(config_paths)
    extra_artifact_paths = _to_paths(extra_artifact_paths)

    if sample_rows < 0:
        raise ValueError("sample_rows must be non-negative")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[PackageArtifact] = []
    dataset_provenance: list[dict[str, Any]] = []
    workload_provenance: list[dict[str, Any]] = []

    summary_artifact = _copy_file(summary_path, output_dir / "summaries", "summary", artifacts)
    summary_rows = _read_summary_rows(summary_path)
    summary = _summary_snapshot(summary_rows, summary_artifact)

    raw_files = _expand_jsonl_inputs(raw_input_paths)
    for raw_file in raw_files:
        _copy_file(raw_file, output_dir / "raw", "raw_results", artifacts)
    raw_sample = _write_jsonl_sample(
        files=raw_files,
        output_path=output_dir / "samples" / "raw_results_sample.jsonl",
        sample_rows=sample_rows,
    )
    if raw_sample is not None:
        _register_artifact(raw_sample, output_dir, "raw_sample", "raw_results_sample.jsonl", artifacts)

    for workload_path in workload_paths:
        copied = _copy_file(workload_path, output_dir / "workloads", "workload", artifacts)
        sample_path = _write_jsonl_sample(
            files=[workload_path],
            output_path=output_dir / "samples" / f"{workload_path.stem}_sample.jsonl",
            sample_rows=sample_rows,
        )
        sample_rel = None
        if sample_path is not None:
            _register_artifact(sample_path, output_dir, "workload_sample", sample_path.name, artifacts)
            sample_rel = _relative_package_path(sample_path, output_dir)
        workload_provenance.append(
            {
                "path": copied.path,
                "sample_path": sample_rel,
                "sha256": sha256_file(workload_path),
                "row_count": _count_nonempty_lines(workload_path),
                "original_name": workload_path.name,
            }
        )

    for manifest_path in dataset_manifest_paths:
        copied = _copy_file(
            manifest_path,
            output_dir / "dataset_manifests",
            "dataset_manifest",
            artifacts,
        )
        dataset_provenance.append(_dataset_provenance(manifest_path, copied))

    for report_path in report_paths:
        _copy_file(report_path, output_dir / "reports", "report", artifacts)

    for config_path in config_paths:
        redacted_path = _write_redacted_config(config_path, output_dir / "configs")
        _register_artifact(redacted_path, output_dir, "redacted_config", config_path.name, artifacts)

    for artifact_path in extra_artifact_paths:
        _copy_file(artifact_path, output_dir / "artifacts", "artifact", artifacts)

    for telemetry_path in _collect_telemetry_artifact_paths(raw_files):
        _copy_file(telemetry_path, output_dir / "telemetry", "telemetry", artifacts)

    missing_metrics = _collect_missing_metrics(summary_rows, raw_files)
    missing_metric_entries = [MissingMetricEntry(metric=name) for name in missing_metrics]
    metric_provenance = _collect_metric_provenance(summary_rows, raw_files)

    limitations = [
        "Do not publish mock metrics as real endpoint results.",
        "Unavailable engine metrics are preserved as null and listed in missing_metrics.json.",
        "Metric sources are documented in metric_provenance.json.",
        "Redacted config snapshots omit endpoint URLs and secret-bearing values.",
    ]
    reproduction_notes = [
        "Use run_manifest.json for artifact hashes and package-relative paths.",
        "Use dataset_manifests/ and workload samples to verify dataset provenance.",
        "Regenerate reports from the packaged summary and comparison CSV files when possible.",
    ]

    manifest = ResultPackageManifest(
        run_name=run_name or summary_path.stem,
        artifact_count=len(artifacts),
        summary=summary,
        artifacts=artifacts,
        workload_provenance=workload_provenance,
        dataset_provenance=dataset_provenance,
        missing_metrics=missing_metric_entries,
        metric_provenance=metric_provenance,
        limitations=limitations,
        reproduction_notes=reproduction_notes,
    )

    missing_metrics_path = output_dir / "missing_metrics.json"
    missing_metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "generated_at": utc_now_iso(),
                "missing_metrics": [
                    entry.model_dump(mode="json") for entry in missing_metric_entries
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _register_artifact(missing_metrics_path, output_dir, "missing_metrics", "missing_metrics.json", artifacts)

    metric_provenance_path = output_dir / "metric_provenance.json"
    metric_provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "generated_at": utc_now_iso(),
                "metrics": metric_provenance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _register_artifact(
        metric_provenance_path,
        output_dir,
        "metric_provenance",
        "metric_provenance.json",
        artifacts,
    )

    readme_path = output_dir / "README_result.md"
    readme_path.write_text(_render_readme(manifest), encoding="utf-8")
    _register_artifact(readme_path, output_dir, "readme", "README_result.md", artifacts)

    manifest.artifacts = artifacts
    manifest.artifact_count = len(artifacts)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    return ResultPackageBuild(
        output_dir=output_dir,
        manifest_path=manifest_path,
        missing_metrics_path=missing_metrics_path,
        metric_provenance_path=metric_provenance_path,
        readme_path=readme_path,
        artifact_count=len(artifacts),
    )


def _to_paths(paths: list[str | Path] | None) -> list[Path]:
    return [Path(path) for path in paths or []]


def _copy_file(
    source: Path,
    dest_dir: Path,
    role: str,
    artifacts: list[PackageArtifact],
) -> PackageArtifact:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Artifact file not found: {source}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(dest_dir / source.name)
    shutil.copyfile(source, dest)
    return _register_artifact(dest, dest_dir.parents[0], role, source.name, artifacts)


def _register_artifact(
    path: Path,
    output_dir: Path,
    role: str,
    original_name: str,
    artifacts: list[PackageArtifact],
) -> PackageArtifact:
    artifact = PackageArtifact(
        role=role,
        path=_relative_package_path(path, output_dir),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        original_name=original_name,
    )
    artifacts.append(artifact)
    return artifact


def _relative_package_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_snapshot(
    rows: list[dict[str, str]],
    summary_artifact: PackageArtifact,
) -> dict[str, Any]:
    return {
        "path": summary_artifact.path,
        "sha256": summary_artifact.sha256,
        "row_count": len(rows),
        "experiments": _unique_csv_values(rows, "experiment_id"),
        "providers": _unique_csv_values(rows, "provider"),
        "engines": _unique_csv_values(rows, "engine"),
        "models": _unique_csv_values(rows, "model_id"),
        "strategies": _unique_csv_values(rows, "strategy"),
        "workloads": _unique_csv_values(rows, "workload"),
        "requests": _sum_int_csv_values(rows, "requests"),
        "successes": _sum_int_csv_values(rows, "successes"),
    }


def _unique_csv_values(rows: list[dict[str, str]], field: str) -> list[str]:
    return sorted({value for row in rows if (value := row.get(field, "").strip())})


def _sum_int_csv_values(rows: list[dict[str, str]], field: str) -> int | None:
    total = 0
    found = False
    for row in rows:
        value = row.get(field, "").strip()
        if not value:
            continue
        try:
            total += int(float(value))
        except ValueError:
            continue
        found = True
    return total if found else None


def _expand_jsonl_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.glob("*.jsonl") if child.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"JSONL input not found: {path}")
    return files


def _write_jsonl_sample(
    *,
    files: list[Path],
    output_path: Path,
    sample_rows: int,
) -> Path | None:
    if sample_rows == 0 or not files:
        return None
    lines: list[str] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    lines.append(line.rstrip("\n"))
                if len(lines) >= sample_rows:
                    break
        if len(lines) >= sample_rows:
            break
    if not lines:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _dataset_provenance(manifest_path: Path, copied_artifact: PackageArtifact) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = [
        "adapter_name",
        "adapter_version",
        "dataset_name",
        "dataset_source_url",
        "source_url",
        "dataset_revision",
        "source_revision",
        "split",
        "license",
        "license_review_status",
        "redistribution_policy",
        "mode",
        "row_count",
        "workload_sha256",
        "prompt_template_hash",
        "tokenizer_id",
        "tokenizer_revision",
        "token_count_method",
    ]
    provenance = {key: payload.get(key) for key in keys if key in payload}
    provenance.update(
        {
            "path": copied_artifact.path,
            "sha256": copied_artifact.sha256,
            "original_name": manifest_path.name,
        }
    )
    return provenance


def _write_redacted_config(config_path: Path, dest_dir: Path) -> Path:
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(dest_dir / f"{config_path.stem}.redacted{config_path.suffix}")
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        dest.write_text(_redact_text(text), encoding="utf-8")
        return dest
    redacted = _redact_value(payload)
    dest.write_text(yaml.safe_dump(redacted, sort_keys=False), encoding="utf-8")
    return dest


def _redact_value(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: _redact_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, key) for item in value]
    if key is not None and _is_secret_key(key):
        return "<redacted>"
    if key is not None and _is_url_key(key) and isinstance(value, str):
        return "<redacted_url>"
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.startswith("tokenizer"):
        return False
    secret_markers = [
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "bearer_token",
        "hf_token",
        "runpod_token",
        "secret",
        "password",
        "credential",
        "authorization",
    ]
    return any(marker in lowered for marker in secret_markers)


def _is_url_key(key: str) -> bool:
    lowered = key.lower()
    return lowered == "base_url" or lowered.endswith("_url")


def _redact_text(text: str) -> str:
    redacted_lines: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if _line_has_secret_key(lowered):
            key = line.split(":", 1)[0]
            redacted_lines.append(f"{key}: <redacted>")
        elif "base_url" in lowered:
            key = line.split(":", 1)[0]
            redacted_lines.append(f"{key}: <redacted_url>")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines) + "\n"


def _line_has_secret_key(lowered_line: str) -> bool:
    if lowered_line.strip().startswith("tokenizer"):
        return False
    return any(
        marker in lowered_line
        for marker in [
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
            "bearer_token",
            "hf_token",
            "runpod_token",
            "secret",
            "password",
        ]
    )


def _collect_missing_metrics(summary_rows: list[dict[str, str]], raw_files: list[Path]) -> list[str]:
    missing: set[str] = set()
    for row in summary_rows:
        missing.update(_split_missing_metrics(row.get("missing_metrics")))
    for raw_file in raw_files:
        with raw_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                missing.update(_split_missing_metrics(payload.get("missing_metrics")))
    return sorted(missing)


def _collect_metric_provenance(
    summary_rows: list[dict[str, str]], raw_files: list[Path]
) -> dict[str, dict[str, list[str]]]:
    collected: dict[str, dict[str, set[str]]] = {}
    for row in summary_rows:
        _merge_metric_provenance_value(collected, row.get("metric_provenance"))
        _merge_metric_source_types(collected, row.get("metric_source_types"))
        _merge_unavailable_reasons(collected, row.get("unavailable_metric_reasons"))
    for raw_file in raw_files:
        with raw_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _merge_metric_provenance_value(collected, payload.get("metric_provenance"))
    return {
        metric: {
            "source_types": sorted(values["source_types"]),
            "measurement_methods": sorted(values["measurement_methods"]),
            "unavailable_reasons": sorted(values["unavailable_reasons"]),
        }
        for metric, values in sorted(collected.items())
    }


def _collect_telemetry_artifact_paths(raw_files: list[Path]) -> list[Path]:
    paths: dict[str, Path] = {}
    for raw_file in raw_files:
        with raw_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for value in _telemetry_paths_from_row(payload):
                    path = Path(value)
                    if path.exists() and path.is_file():
                        paths[path.resolve().as_posix()] = path
    return [paths[key] for key in sorted(paths)]


def _telemetry_paths_from_row(payload: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ["telemetry_summary_path", "telemetry_snapshots_path"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        telemetry = metadata.get("telemetry")
        if isinstance(telemetry, dict):
            for key in ["summary_path", "snapshots_path"]:
                value = telemetry.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append(value)
    return paths


def _merge_metric_provenance_value(
    collected: dict[str, dict[str, set[str]]], value: Any
) -> None:
    if value is None or value == "":
        return
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return
    if not isinstance(value, dict):
        return
    for metric, details in value.items():
        if not isinstance(details, dict):
            continue
        entry = _metric_provenance_entry(collected, str(metric))
        for source_type in _as_list(details.get("source_types") or details.get("source_type")):
            entry["source_types"].add(str(source_type))
        for method in _as_list(
            details.get("measurement_methods") or details.get("measurement_method")
        ):
            entry["measurement_methods"].add(str(method))
        if details.get("available") is False and details.get("missing_reason"):
            entry["unavailable_reasons"].add(str(details["missing_reason"]))
        for reason in _as_list(details.get("unavailable_reasons")):
            entry["unavailable_reasons"].add(str(reason))


def _merge_metric_source_types(
    collected: dict[str, dict[str, set[str]]], value: str | None
) -> None:
    if not value:
        return
    for piece in value.split(";"):
        if ":" not in piece:
            continue
        metric, source_types = piece.split(":", 1)
        entry = _metric_provenance_entry(collected, metric)
        for source_type in source_types.split(","):
            if source_type.strip():
                entry["source_types"].add(source_type.strip())


def _merge_unavailable_reasons(
    collected: dict[str, dict[str, set[str]]], value: str | None
) -> None:
    if not value:
        return
    for piece in value.split(";"):
        if ":" not in piece:
            continue
        metric, reasons = piece.split(":", 1)
        entry = _metric_provenance_entry(collected, metric)
        for reason in reasons.split(" | "):
            if reason.strip():
                entry["unavailable_reasons"].add(reason.strip())


def _metric_provenance_entry(
    collected: dict[str, dict[str, set[str]]], metric: str
) -> dict[str, set[str]]:
    return collected.setdefault(
        metric,
        {
            "source_types": set(),
            "measurement_methods": set(),
            "unavailable_reasons": set(),
        },
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _split_missing_metrics(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str):
        pieces = value.replace(",", ";").split(";")
        return {piece.strip() for piece in pieces if piece.strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _render_readme(manifest: ResultPackageManifest) -> str:
    summary = manifest.summary
    missing = ", ".join(entry.metric for entry in manifest.missing_metrics) or "none"
    artifacts = "\n".join(
        f"- `{artifact.path}` ({artifact.role}, sha256 `{artifact.sha256}`)"
        for artifact in manifest.artifacts
    )
    workloads = ", ".join(summary.get("workloads", [])) or "unknown"
    engines = ", ".join(summary.get("engines", [])) or "unknown"
    strategies = ", ".join(summary.get("strategies", [])) or "unknown"

    return (
        "# KVOptBench Result Package\n\n"
        "Use this package to review, reproduce, and publish a completed benchmark run.\n\n"
        "## Run Summary\n\n"
        f"- Run name: `{manifest.run_name}`\n"
        f"- Created at: `{manifest.created_at}`\n"
        f"- Engines: {engines}\n"
        f"- Strategies: {strategies}\n"
        f"- Workloads: {workloads}\n"
        f"- Summary rows: {summary.get('row_count', 0)}\n"
        f"- Requests: {summary.get('requests')}\n"
        f"- Missing metrics: {missing}\n\n"
        "## Artifacts\n\n"
        f"{artifacts}\n\n"
        "## Reproducibility Notes\n\n"
        "- `run_manifest.json` contains package-relative paths and hashes.\n"
        "- `missing_metrics.json` explains unavailable metrics instead of treating them as zero.\n"
        "- `metric_provenance.json` records whether metrics are observed, reported, imported, derived, or estimated.\n"
        "- Config snapshots are redacted before packaging.\n"
        "- Do not publish mock metrics as real endpoint results.\n"
        "- Do not publish private endpoint URLs, secrets, or private workload data.\n"
    )
