"""Dataset adapter schemas and manifest writing helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kvoptbench import __version__
from kvoptbench.datasets.hashing import sha256_file, sha256_text
from kvoptbench.schemas import WorkloadItem, utc_now_iso


class DatasetAdapterInfo(BaseModel):
    """Static metadata for one dataset adapter."""

    name: str
    dataset_name: str
    source_url: str
    supported_modes: tuple[str, ...]
    license: str | None = None
    rights_note: str | None = None
    download_supported: bool = False
    fixture_supported: bool = True


class DatasetPrepareOptions(BaseModel):
    """User options for preparing a dataset workload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    mode: str
    out: Path
    manifest: Path
    source_path: Path | None = None
    split: str | None = None
    max_items: int | None = Field(default=None, ge=1)
    seed: int = 7
    target_input_tokens: int = Field(default=32768, ge=1)
    target_output_tokens: int = Field(default=256, ge=1)
    context_buckets: tuple[int, ...] = ()
    book_ids: tuple[str, ...] = ()
    download: bool = False
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    token_count_method: str = "char_approx_4"


class DatasetPrepareResult(BaseModel):
    """Paths and counts produced by one adapter run."""

    output_path: Path
    manifest_path: Path
    row_count: int
    workload_sha256: str
    excluded_count: int = 0


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


class DatasetManifest(BaseModel):
    """Reproducibility manifest for a generated dataset workload."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1"
    manifest_version: str = "1"
    kvoptbench_version: str = __version__
    git_commit: str = Field(default_factory=_git_commit)
    adapter_name: str
    adapter_version: str
    dataset_name: str
    dataset: str
    dataset_source_url: str
    source_url: str
    dataset_revision: str | None = None
    source_revision: str | None = None
    split: str | None = None
    license: str | None = None
    rights_note: str | None = None
    license_review_status: str
    redistribution_policy: str
    adapter: str
    mode: str
    generation_command: str
    generation_started_at: str | None = None
    generation_finished_at: str | None = None
    generation_duration_sec: float | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    token_count_method: str
    sampling_method: str = "deterministic"
    seed: int | None = None
    max_items_requested: int | None = None
    max_items_emitted: int = 0
    row_count: int
    excluded_count: int = 0
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)
    target_input_tokens: int | None = None
    target_output_tokens: int | None = None
    context_buckets: list[int] = Field(default_factory=list)
    min_input_tokens: int | None = None
    max_input_tokens: int | None = None
    avg_input_tokens: float | None = None
    input_token_histogram: dict[str, int] = Field(default_factory=dict)
    prefix_group_count: int = 0
    items_per_prefix_group: dict[str, int] = Field(default_factory=dict)
    workload_sha256: str
    workload_hash: str | None = None
    source_sha256: str | None = None
    prompt_template: str
    prompt_template_hash: str
    normalization_rules: list[str] = Field(default_factory=list)
    truncation_policy: str = "none"
    truncation_count: int = 0
    known_limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def write_workload_jsonl(items: list[WorkloadItem], output_path: str | Path) -> str:
    """Write workload rows and return the output file hash."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
    return sha256_file(path)


def write_manifest(manifest: DatasetManifest, manifest_path: str | Path) -> Path:
    """Write a dataset manifest JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def build_manifest(
    *,
    info: DatasetAdapterInfo,
    options: DatasetPrepareOptions,
    items: list[WorkloadItem],
    workload_sha256: str,
    prompt_template: str,
    prompt_template_text: str,
    license_review_status: str,
    redistribution_policy: str,
    excluded_reasons: list[str] | None = None,
    rights_note: str | None = None,
    source_sha256: str | None = None,
    generation_started_at: str | None = None,
    generation_finished_at: str | None = None,
    generation_duration_sec: float | None = None,
    known_limitations: list[str] | None = None,
) -> DatasetManifest:
    """Construct a manifest from generated workload rows."""
    excluded_reasons = excluded_reasons or []
    measured_inputs = [
        int(item.metadata.get("measured_input_tokens", item.target_input_tokens)) for item in items
    ]
    prefix_groups = Counter(item.prefix_group_id for item in items if item.prefix_group_id)
    truncation_count = sum(1 for item in items if item.metadata.get("truncated") is True)
    context_buckets = sorted(
        {
            int(value)
            for item in items
            for value in [
                item.metadata.get("context_bucket"),
                item.metadata.get("context_token_bucket"),
            ]
            if value is not None
        }
    )
    if not context_buckets:
        context_buckets = [options.target_input_tokens]

    return DatasetManifest(
        adapter_name=info.name,
        adapter_version="0.1.0",
        dataset_name=info.dataset_name,
        dataset=info.name,
        dataset_source_url=info.source_url,
        source_url=info.source_url,
        split=options.split,
        license=info.license,
        rights_note=rights_note or info.rights_note,
        license_review_status=license_review_status,
        redistribution_policy=redistribution_policy,
        adapter=info.name,
        mode=options.mode,
        generation_command=generation_command(options),
        generation_started_at=generation_started_at,
        generation_finished_at=generation_finished_at,
        generation_duration_sec=generation_duration_sec,
        tokenizer_id=options.tokenizer_id,
        tokenizer_revision=options.tokenizer_revision,
        token_count_method=options.token_count_method,
        seed=options.seed,
        max_items_requested=options.max_items,
        max_items_emitted=len(items),
        row_count=len(items),
        excluded_count=len(excluded_reasons),
        exclusion_reasons=dict(Counter(excluded_reasons)),
        target_input_tokens=options.target_input_tokens,
        target_output_tokens=options.target_output_tokens,
        context_buckets=context_buckets,
        min_input_tokens=min(measured_inputs) if measured_inputs else None,
        max_input_tokens=max(measured_inputs) if measured_inputs else None,
        avg_input_tokens=round(sum(measured_inputs) / len(measured_inputs), 3)
        if measured_inputs
        else None,
        prefix_group_count=len(prefix_groups),
        items_per_prefix_group=dict(prefix_groups),
        workload_sha256=workload_sha256,
        workload_hash=workload_sha256,
        source_sha256=source_sha256,
        prompt_template=prompt_template,
        prompt_template_hash=sha256_text(prompt_template_text),
        truncation_policy=_manifest_truncation_policy(items),
        truncation_count=truncation_count,
        known_limitations=known_limitations or [],
        notes=["Raw public dataset files are not committed by KVOptBench adapters."],
    )


def generation_command(options: DatasetPrepareOptions) -> str:
    """Render a reproducible command shape for a dataset preparation run."""
    parts = [
        "kvoptbench dataset prepare",
        f"--source {options.source}",
        f"--mode {options.mode}",
        f"--out {options.out}",
        f"--manifest {options.manifest}",
    ]
    if options.source_path is not None:
        parts.append(f"--source-path {options.source_path}")
    if options.split:
        parts.append(f"--split {options.split}")
    if options.max_items:
        parts.append(f"--max-items {options.max_items}")
    if options.context_buckets:
        parts.append("--context-buckets " + ",".join(str(value) for value in options.context_buckets))
    if options.book_ids:
        parts.append("--book-ids " + ",".join(options.book_ids))
    parts.extend(
        [
            f"--target-input-tokens {options.target_input_tokens}",
            f"--target-output-tokens {options.target_output_tokens}",
            f"--seed {options.seed}",
        ]
    )
    return " ".join(parts)


def source_hash(path: Path | None) -> str | None:
    """Hash a source file when one concrete source file was provided."""
    if path is None or not path.exists() or path.is_dir():
        return None
    return sha256_file(path)


def _manifest_truncation_policy(items: list[WorkloadItem]) -> str:
    policies = {
        str(item.metadata.get("truncation_policy", "none"))
        for item in items
        if item.metadata.get("truncated") is True
    }
    if not policies:
        return "none"
    return ",".join(sorted(policies))
