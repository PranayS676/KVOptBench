"""Run-lifecycle telemetry collection and artifact writing."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kvoptbench.schemas import (
    ExperimentConfig,
    LmcacheTelemetrySource,
    PrometheusTelemetrySource,
    TelemetryConfig,
    utc_now_iso,
)
from kvoptbench.telemetry.lmcache import normalize_lmcache_metrics
from kvoptbench.telemetry.metrics import MissingMetric, TelemetrySnapshot
from kvoptbench.telemetry.nvidia_smi import GpuSampleProvider, GpuSampler
from kvoptbench.telemetry.prometheus import PrometheusScrapeResult, scrape_prometheus_endpoint


class TelemetryRunSummary(BaseModel):
    """Run-level telemetry artifact summary."""

    schema_version: str = "1"
    run_id: str
    enabled: bool
    started_at: str | None = None
    finished_at: str | None = None
    output_dir: str | None = None
    snapshots_path: str | None = None
    summary_path: str | None = None
    snapshot_count: int = 0
    metrics: dict[str, float | None] = Field(default_factory=dict)
    missing_metrics: list[MissingMetric] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)


class TelemetryRunCollector:
    """Collect optional run-level telemetry around an experiment."""

    def __init__(
        self,
        config: TelemetryConfig | None,
        *,
        run_id: str,
        output_file: Path,
        prometheus_clients: dict[str, Any] | None = None,
        lmcache_clients: dict[str, Any] | None = None,
        gpu_sample_provider: GpuSampleProvider | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.enabled = bool(config and config.enabled)
        self.prometheus_clients = prometheus_clients or {}
        self.lmcache_clients = lmcache_clients or {}
        self.gpu_sample_provider = gpu_sample_provider
        self.run_dir = _telemetry_run_dir(config, output_file, run_id) if self.enabled else None
        self.snapshots_path = self.run_dir / "telemetry_snapshots.jsonl" if self.run_dir else None
        self.summary_path = self.run_dir / "telemetry_summary.json" if self.run_dir else None
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._metrics: dict[str, float | None] = {}
        self._missing_by_metric: dict[str, MissingMetric] = {}
        self._periodic_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._gpu_sampler: GpuSampler | None = None

    async def start(self) -> None:
        """Start run telemetry and capture before-run snapshots."""
        if not self.enabled or self.config is None or self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = utc_now_iso()
        if self.snapshots_path is not None:
            self.snapshots_path.write_text("", encoding="utf-8")
        if self.config.gpu and self.config.gpu.enabled:
            self._gpu_sampler = GpuSampler(
                sample_provider=self.gpu_sample_provider,
                expected_metrics=self.config.gpu.expected_metrics,
                sample_interval_seconds=self.config.gpu.sample_interval_seconds,
                source_type=self.config.gpu.provider,
            )
            self._gpu_sampler.start()
            await self._sample_gpu("before_run")
        await self.capture_phase("before_run")
        if self._needs_periodic_collection():
            self._stop_event = asyncio.Event()
            self._periodic_task = asyncio.create_task(self._periodic_loop())

    async def capture_phase(self, phase: str) -> None:
        """Capture all configured sources for a named run phase."""
        if not self.enabled or self.config is None:
            return
        for source in self.config.prometheus:
            if phase in source.scrape_phases:
                result = await asyncio.to_thread(self._scrape_prometheus, source)
                self._record_prometheus_snapshot(phase, result)
        for source in self.config.lmcache:
            if phase in source.scrape_phases:
                snapshot = await asyncio.to_thread(self._collect_lmcache, source)
                self._record_snapshot(
                    phase=phase,
                    source_name=source.name,
                    source_type=snapshot.source_type,
                    success=not snapshot.missing_metrics,
                    metrics=snapshot.metrics,
                    missing_metrics=snapshot.missing_metrics,
                    samples=snapshot.samples,
                    source_path=snapshot.source_path,
                )

    async def stop(self) -> TelemetryRunSummary:
        """Stop run telemetry, write summary artifacts, and return the summary."""
        if not self.enabled or self.config is None:
            return self.summary()
        if self._periodic_task is not None:
            if self._stop_event is not None:
                self._stop_event.set()
            self._periodic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._periodic_task
        await self.capture_phase("after_run")
        if self._gpu_sampler is not None:
            await self._sample_gpu("after_run")
            final_snapshot = await asyncio.to_thread(self._gpu_sampler.stop)
            self._record_snapshot(
                phase="after_run",
                source_name="gpu",
                source_type=final_snapshot.source_type,
                success=not final_snapshot.missing_metrics,
                metrics=final_snapshot.metrics,
                missing_metrics=final_snapshot.missing_metrics,
                samples=final_snapshot.samples,
                metadata=final_snapshot.metadata,
            )
        self._finished_at = utc_now_iso()
        summary = self.summary()
        if self.summary_path is not None:
            self.summary_path.write_text(
                json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
        return summary

    def summary(self) -> TelemetryRunSummary:
        """Return the current run-level telemetry summary."""
        return TelemetryRunSummary(
            run_id=self.run_id,
            enabled=self.enabled,
            started_at=self._started_at,
            finished_at=self._finished_at,
            output_dir=self.run_dir.as_posix() if self.run_dir else None,
            snapshots_path=self.snapshots_path.as_posix() if self.snapshots_path else None,
            summary_path=self.summary_path.as_posix() if self.summary_path else None,
            snapshot_count=len(self._snapshots),
            metrics=dict(sorted(self._metrics.items())),
            missing_metrics=sorted(self._missing_by_metric.values(), key=lambda item: item.metric),
            sources=self._source_summaries(),
        )

    async def _periodic_loop(self) -> None:
        interval = self._minimum_interval()
        if interval is None:
            return
        while self._stop_event is not None and not self._stop_event.is_set():
            await asyncio.sleep(interval)
            if self._stop_event.is_set():
                break
            if self._gpu_sampler is not None:
                await self._sample_gpu("during_run")
            await self.capture_phase("during_run")

    async def _sample_gpu(self, phase: str) -> None:
        if self._gpu_sampler is None:
            return
        snapshot = await asyncio.to_thread(self._gpu_sampler.sample_once)
        self._record_snapshot(
            phase=phase,
            source_name="gpu",
            source_type=snapshot.source_type,
            success=not snapshot.missing_metrics,
            metrics=snapshot.metrics,
            missing_metrics=snapshot.missing_metrics,
            samples=snapshot.samples,
            metadata=snapshot.metadata,
        )

    def _scrape_prometheus(self, source: PrometheusTelemetrySource) -> PrometheusScrapeResult:
        return scrape_prometheus_endpoint(
            source.url,
            source_name=source.name,
            timeout_seconds=source.timeout_seconds,
            expected_metrics=source.expected_metrics,
            metric_aliases=source.metric_aliases,
            client=self.prometheus_clients.get(source.name),
        )

    def _collect_lmcache(self, source: LmcacheTelemetrySource) -> TelemetrySnapshot:
        if source.format == "prometheus" and source.url:
            result = scrape_prometheus_endpoint(
                source.url,
                source_name=source.name,
                timeout_seconds=source.timeout_seconds,
                expected_metrics=[],
                client=self.lmcache_clients.get(source.name),
            )
            if not result.success:
                return TelemetrySnapshot(
                    metrics={metric: None for metric in source.expected_metrics},
                    missing_metrics=result.missing_metrics,
                    source_type="lmcache_prometheus",
                    samples=[],
                )
            return normalize_lmcache_metrics(
                result.raw_text or "",
                expected_metrics=source.expected_metrics,
                metric_aliases=source.metric_aliases,
            )
        if source.path is not None:
            return normalize_lmcache_metrics(
                source.path,
                expected_metrics=source.expected_metrics,
                metric_aliases=source.metric_aliases,
            )
        return TelemetrySnapshot(
            metrics={metric: None for metric in source.expected_metrics},
            missing_metrics=[
                MissingMetric(
                    metric=metric,
                    reason="LMCache telemetry source did not define url or path.",
                    source=source.name,
                )
                for metric in source.expected_metrics
            ],
            source_type="lmcache",
            samples=[],
        )

    def _record_prometheus_snapshot(
        self,
        phase: str,
        result: PrometheusScrapeResult,
    ) -> None:
        metrics = {record.name: record.value for record in result.records}
        self._record_snapshot(
            phase=phase,
            source_name=result.source_name,
            source_type="prometheus",
            success=result.success,
            metrics=metrics,
            missing_metrics=result.missing_metrics,
            records=[record.model_dump(mode="json") for record in result.records],
            status_code=result.status_code,
            error_type=result.error_type,
            error_message=result.error_message,
        )

    def _record_snapshot(
        self,
        *,
        phase: str,
        source_name: str,
        source_type: str,
        success: bool,
        metrics: dict[str, float | None],
        missing_metrics: list[MissingMetric],
        records: list[dict[str, Any]] | None = None,
        samples: list[dict[str, Any]] | None = None,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        snapshot = {
            "schema_version": "1",
            "run_id": self.run_id,
            "phase": phase,
            "source_name": source_name,
            "source_type": source_type,
            "collected_at": utc_now_iso(),
            "success": success,
            "metrics": metrics,
            "missing_metrics": [item.model_dump(mode="json") for item in missing_metrics],
            "records": records or [],
            "samples": samples or [],
            "source_path": source_path,
            "metadata": metadata or {},
            "status_code": status_code,
            "error_type": error_type,
            "error_message": error_message,
        }
        self._snapshots.append(snapshot)
        self._merge_metrics(metrics)
        self._merge_missing(missing_metrics)
        if self.snapshots_path is not None:
            with self.snapshots_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    def _merge_metrics(self, metrics: dict[str, float | None]) -> None:
        for metric, value in metrics.items():
            if value is None:
                continue
            current = self._metrics.get(metric)
            if metric.endswith("_peak_gb") and current is not None:
                self._metrics[metric] = max(float(current), float(value))
            else:
                self._metrics[metric] = float(value)
            self._missing_by_metric.pop(metric, None)

    def _merge_missing(self, missing_metrics: list[MissingMetric]) -> None:
        for missing in missing_metrics:
            if self._metrics.get(missing.metric) is not None:
                continue
            self._missing_by_metric[missing.metric] = missing

    def _needs_periodic_collection(self) -> bool:
        if self.config is None:
            return False
        if self.config.gpu and self.config.gpu.enabled and self.config.gpu.sample_interval_seconds:
            return True
        return any(
            source.scrape_interval_seconds or "during_run" in source.scrape_phases
            for source in [*self.config.prometheus, *self.config.lmcache]
        )

    def _minimum_interval(self) -> float | None:
        if self.config is None:
            return None
        intervals: list[float] = []
        if self.config.gpu and self.config.gpu.enabled and self.config.gpu.sample_interval_seconds:
            intervals.append(float(self.config.gpu.sample_interval_seconds))
        for source in [*self.config.prometheus, *self.config.lmcache]:
            if source.scrape_interval_seconds:
                intervals.append(float(source.scrape_interval_seconds))
        return min(intervals) if intervals else 1.0

    def _source_summaries(self) -> list[dict[str, Any]]:
        if self.config is None:
            return []
        sources: list[dict[str, Any]] = []
        for source in self.config.prometheus:
            sources.append(
                {
                    "name": source.name,
                    "type": "prometheus",
                    "expected_metrics": source.expected_metrics,
                    "scrape_phases": source.scrape_phases,
                }
            )
        for source in self.config.lmcache:
            sources.append(
                {
                    "name": source.name,
                    "type": "lmcache",
                    "format": source.format,
                    "expected_metrics": source.expected_metrics,
                    "scrape_phases": source.scrape_phases,
                }
            )
        if self.config.gpu and self.config.gpu.enabled:
            sources.append(
                {
                    "name": "gpu",
                    "type": self.config.gpu.provider,
                    "expected_metrics": self.config.gpu.expected_metrics,
                }
            )
        return sources


def build_telemetry_collector(
    config: ExperimentConfig,
    *,
    run_id: str,
) -> TelemetryRunCollector:
    """Build a default run collector for the experiment runner."""
    return TelemetryRunCollector(
        config.telemetry,
        run_id=run_id,
        output_file=config.output_file,
    )


def _telemetry_run_dir(config: TelemetryConfig | None, output_file: Path, run_id: str) -> Path:
    if config is not None and config.output_dir is not None:
        return config.output_dir / run_id
    raw_parent = output_file.parent
    if raw_parent.name == "raw":
        return raw_parent.parent / "telemetry" / run_id
    return raw_parent / "telemetry" / run_id
