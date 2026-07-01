import asyncio
import json
from pathlib import Path

from kvoptbench.schemas import (
    GpuTelemetryConfig,
    LmcacheTelemetrySource,
    PrometheusTelemetrySource,
    TelemetryConfig,
)
from kvoptbench.telemetry.runtime import TelemetryRunCollector


def test_telemetry_collector_writes_run_artifacts(tmp_path: Path) -> None:
    config = TelemetryConfig(
        enabled=True,
        output_dir=tmp_path / "telemetry",
        prometheus=[
            PrometheusTelemetrySource(
                name="vllm",
                url="http://metrics.local/metrics",
                expected_metrics=["engine_reported_cache_hit_rate"],
                metric_aliases={
                    "vllm:prefix_cache_hit_rate": "engine_reported_cache_hit_rate"
                },
            )
        ],
        gpu=GpuTelemetryConfig(enabled=True, sample_interval_seconds=None),
        lmcache=[
            LmcacheTelemetrySource(
                name="lmcache",
                url="http://lmcache.local/metrics",
                expected_metrics=["lmcache_cache_hit_rate"],
            )
        ],
    )
    gpu_samples = iter(
        [
            {"memory.used [MiB]": "1024 MiB"},
            {"memory.used [MiB]": "2048 MiB"},
            {"memory.used [MiB]": "3072 MiB"},
        ]
    )
    collector = TelemetryRunCollector(
        config,
        run_id="run-123",
        output_file=tmp_path / "results" / "raw" / "run.jsonl",
        prometheus_clients={
            "vllm": _FakeClient("vllm:prefix_cache_hit_rate 0.7\n"),
        },
        lmcache_clients={
            "lmcache": _FakeClient(
                "lmcache_cache_hit_total 9\nlmcache_cache_miss_total 1\n"
            ),
        },
        gpu_sample_provider=lambda: next(gpu_samples),
    )

    summary = asyncio.run(_run_collector(collector))

    run_dir = tmp_path / "telemetry" / "run-123"
    snapshots_path = run_dir / "telemetry_snapshots.jsonl"
    summary_path = run_dir / "telemetry_summary.json"
    assert snapshots_path.exists()
    assert summary_path.exists()
    assert summary.metrics["engine_reported_cache_hit_rate"] == 0.7
    assert summary.metrics["lmcache_cache_hit_rate"] == 0.9
    assert summary.metrics["gpu_memory_peak_gb"] == 2.0
    assert summary.telemetry_profile is None
    assert summary.snapshots_path == snapshots_path.as_posix()
    assert summary.summary_path == summary_path.as_posix()
    snapshots = [
        json.loads(line) for line in snapshots_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {snapshot["phase"] for snapshot in snapshots} >= {"before_run", "after_run"}
    assert "gpu_memory_peak_gb" in summary_path.read_text(encoding="utf-8")


def test_telemetry_collector_summary_preserves_profile_name(tmp_path: Path) -> None:
    config = TelemetryConfig(
        enabled=True,
        profile="gpu_only",
        output_dir=tmp_path / "telemetry",
        gpu=GpuTelemetryConfig(enabled=True, sample_interval_seconds=None),
    )
    collector = TelemetryRunCollector(
        config,
        run_id="run-profile",
        output_file=tmp_path / "results" / "raw" / "run.jsonl",
        gpu_sample_provider=lambda: {"memory.used [MiB]": "1024 MiB"},
    )

    summary = asyncio.run(_run_collector(collector))

    assert summary.telemetry_profile == "gpu_only"
    assert summary.metrics["gpu_memory_peak_gb"] == 1.0
    persisted = json.loads(Path(summary.summary_path or "").read_text(encoding="utf-8"))
    assert persisted["telemetry_profile"] == "gpu_only"


async def _run_collector(collector: TelemetryRunCollector):
    await collector.start()
    await collector.capture_phase("during_run")
    return await collector.stop()


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self, url: str, *, timeout: float) -> "_FakeResponse":
        return _FakeResponse(self.text)


class _FakeResponse:
    status_code = 200

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None
