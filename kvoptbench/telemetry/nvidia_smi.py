"""Offline NVIDIA/DCGM/nvidia-smi telemetry normalization helpers."""

from __future__ import annotations

import csv
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvoptbench.telemetry.metrics import MissingMetric, TelemetrySnapshot
from kvoptbench.telemetry.prometheus import parse_prometheus_samples


_USED_ALIASES = {
    "gpu_memory_used_gb",
    "gpu_memory_used_gib",
    "gpu_memory_used",
    "memory.used",
    "memory_used",
    "memory.used [mib]",
    "memory.used_[mib]",
    "memory_used_mib",
    "memory_used_[mib]",
    "used_memory_mib",
    "fb_memory_usage.used",
}
_PEAK_ALIASES = {
    "gpu_memory_peak_gb",
    "gpu_memory_peak_gib",
    "gpu_memory_peak",
    "memory_peak",
    "memory_peak_mib",
    "max_memory_used_mib",
    "peak_memory_mib",
    "gpu_memory_peak_mib",
}
_GPU_MEMORY_FIELDS = ["gpu_memory_used_gb", "gpu_memory_peak_gb"]
_MIB_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*MiB", re.IGNORECASE)
GpuSampleSource = str | Path | dict[str, Any] | list[dict[str, Any]]
GpuSampleProvider = Callable[[], GpuSampleSource | None]


def collect_gpu_metrics() -> dict:
    return {
        "gpu_memory_used_gb": None,
        "gpu_memory_peak_gb": None,
        "reason": "GPU telemetry is not collected in local mock mode.",
    }


class GpuSampler:
    """Run-window GPU sampler that is testable with fake sample providers."""

    def __init__(
        self,
        sample_provider: GpuSampleProvider | None = None,
        *,
        source_type: str = "nvidia_smi",
        expected_metrics: list[str] | tuple[str, ...] | None = None,
        sample_interval_seconds: float | None = None,
    ) -> None:
        self.sample_provider = sample_provider or nvidia_smi_sample_provider
        self.source_type = source_type
        self.expected_metrics = list(expected_metrics or _GPU_MEMORY_FIELDS)
        self.sample_interval_seconds = sample_interval_seconds
        self._started_at: str | None = None
        self._samples: list[dict[str, Any]] = []
        self._missing_metrics: list[MissingMetric] = []

    def start(self) -> None:
        """Mark the start of a benchmark sampling window."""
        self._started_at = _utc_now()
        self._samples = []
        self._missing_metrics = []

    def sample_once(self) -> TelemetrySnapshot:
        """Collect one provider sample and normalize it without requiring a GPU in tests."""
        if self._started_at is None:
            self.start()
        try:
            source = self.sample_provider()
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            snapshot = self._unavailable_snapshot(f"GPU sample provider failed: {exc}")
            self._missing_metrics.extend(snapshot.missing_metrics)
            return snapshot

        if source is None:
            snapshot = self._unavailable_snapshot("GPU sample provider returned no sample.")
            self._missing_metrics.extend(snapshot.missing_metrics)
            return snapshot

        snapshot = normalize_gpu_metrics(source)
        self._samples.extend(snapshot.samples)
        self._missing_metrics.extend(snapshot.missing_metrics)
        return snapshot

    def stop(self) -> TelemetrySnapshot:
        """Summarize all samples collected during the benchmark window."""
        started_at = self._started_at or _utc_now()
        finished_at = _utc_now()
        if not self._samples:
            snapshot = self._unavailable_snapshot(
                "No GPU telemetry samples were collected during the benchmark window."
            )
        else:
            snapshot = normalize_gpu_metrics(self._samples)
            snapshot.missing_metrics = _dedupe_missing_metrics(
                [*snapshot.missing_metrics, *self._missing_metrics]
            )
        snapshot.source_type = self.source_type
        snapshot.metadata.update(
            {
                "sample_started_at": started_at,
                "sample_finished_at": finished_at,
                "sample_interval_seconds": self.sample_interval_seconds,
            }
        )
        return snapshot

    def _unavailable_snapshot(self, reason: str) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            metrics={metric: None for metric in self.expected_metrics},
            missing_metrics=[
                MissingMetric(metric=metric, reason=reason, source=self.source_type)
                for metric in self.expected_metrics
            ],
            source_type=self.source_type,
            samples=[],
        )


def nvidia_smi_sample_provider(timeout_seconds: float = 2.0) -> str:
    """Return live ``nvidia-smi`` CSV output for optional runtime use."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.used,memory.total",
            "--format=csv",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.stdout


def normalize_gpu_metrics(source: str | Path | dict[str, Any] | list[dict[str, Any]]) -> TelemetrySnapshot:
    """Normalize supplied NVIDIA/DCGM/nvidia-smi samples without live GPU access."""
    samples, source_type, source_path = _load_samples(source)
    used_values = [value for sample in samples if (value := _extract_used_gb(sample)) is not None]
    peak_values = [value for sample in samples if (value := _extract_peak_gb(sample)) is not None]

    metrics: dict[str, float | None] = {
        "gpu_memory_used_gb": _round_gb(used_values[-1]) if used_values else None,
        "gpu_memory_peak_gb": _round_gb(max(peak_values or used_values)) if peak_values or used_values else None,
    }
    missing_metrics = [
        MissingMetric(
            metric=name,
            reason=f"{name} was not present in the supplied GPU telemetry sample.",
            source=source_path or source_type,
        )
        for name in _GPU_MEMORY_FIELDS
        if metrics[name] is None
    ]

    return TelemetrySnapshot(
        metrics=metrics,
        missing_metrics=missing_metrics,
        source_type=source_type,
        source_path=source_path,
        samples=samples,
    )


def _load_samples(
    source: str | Path | dict[str, Any] | list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str | None]:
    if isinstance(source, dict):
        return [source], "dict", None
    if isinstance(source, list):
        return [sample for sample in source if isinstance(sample, dict)], "dict", None

    text, source_path = _read_text_source(source)
    if "DCGM_FI_DEV_FB_USED" in text:
        return _load_dcgm_samples(text, source_path), "dcgm_text", source_path
    if _looks_like_csv(text):
        return _load_csv_samples(text), "nvidia_smi_csv", source_path
    return _load_text_samples(text), "nvidia_smi_text", source_path


def _read_text_source(source: str | Path) -> tuple[str, str | None]:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8"), source.name
    if "\n" not in source and "\r" not in source:
        candidate = Path(source)
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.read_text(encoding="utf-8"), candidate.name
        except OSError:
            pass
    return source, None


def _looks_like_csv(text: str) -> bool:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return "," in first_line and any(marker in first_line.lower() for marker in ["memory", "gpu"])


def _load_csv_samples(text: str) -> list[dict[str, Any]]:
    handle = text.splitlines()
    return [dict(row) for row in csv.DictReader(handle, skipinitialspace=True)]


def _load_text_samples(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for match in _MIB_RE.finditer(text):
        samples.append({"memory.used [MiB]": match.group("value")})
    return samples or [{"raw_text": text}]


def _load_dcgm_samples(text: str, source_path: str | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in parse_prometheus_samples(text):
        if record.name == "DCGM_FI_DEV_FB_USED":
            sample = dict(record.labels)
            sample["memory_used_mib"] = record.value
            sample["source_path"] = source_path
            samples.append(sample)
    return samples


def _extract_used_gb(sample: dict[str, Any]) -> float | None:
    return _extract_memory_gb(sample, _USED_ALIASES)


def _extract_peak_gb(sample: dict[str, Any]) -> float | None:
    return _extract_memory_gb(sample, _PEAK_ALIASES)


def _extract_memory_gb(sample: dict[str, Any], aliases: set[str]) -> float | None:
    for key, value in sample.items():
        normalized_key = _normalize_key(key)
        if normalized_key in aliases:
            return _memory_value_to_gb(value, key)
    return None


def _normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _memory_value_to_gb(value: Any, key: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        match = re.search(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>mib|gb|gib)?", text, re.IGNORECASE)
        if match is None:
            return None
        number = float(match.group("number"))
        unit = (match.group("unit") or _unit_from_key(key)).lower()
        return _convert_to_gb(number, unit)
    number = float(value)
    return _convert_to_gb(number, _unit_from_key(key))


def _unit_from_key(key: Any) -> str:
    lowered = str(key).lower()
    if "mib" in lowered or lowered.endswith("_mb"):
        return "mib"
    if "gib" in lowered or lowered.endswith("_gb") or lowered.endswith("_gib"):
        return "gib"
    return "mib"


def _convert_to_gb(value: float, unit: str) -> float:
    if unit == "mib":
        return value / 1024
    return value


def _round_gb(value: float) -> float:
    return round(value, 4)


def _dedupe_missing_metrics(missing_metrics: list[MissingMetric]) -> list[MissingMetric]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[MissingMetric] = []
    for missing in missing_metrics:
        key = (missing.metric, missing.reason, missing.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(missing)
    return deduped


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

