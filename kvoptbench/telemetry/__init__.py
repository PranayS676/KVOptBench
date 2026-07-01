"""Telemetry adapter foundations."""

from kvoptbench.telemetry.metrics import MetricRecord, MissingMetric, TelemetrySnapshot
from kvoptbench.telemetry.lmcache import normalize_lmcache_metrics, parse_lmcache_jsonl
from kvoptbench.telemetry.nvidia_smi import GpuSampler, normalize_gpu_metrics
from kvoptbench.telemetry.prometheus import (
    PrometheusScrapeResult,
    parse_prometheus_file,
    parse_prometheus_samples,
    scrape_prometheus_endpoint,
    scrape_prometheus_endpoints,
)
from kvoptbench.telemetry.profiles import (
    TelemetryProfile,
    get_telemetry_profile,
    load_telemetry_profiles,
)

__all__ = [
    "GpuSampler",
    "MetricRecord",
    "MissingMetric",
    "PrometheusScrapeResult",
    "TelemetrySnapshot",
    "TelemetryProfile",
    "get_telemetry_profile",
    "load_telemetry_profiles",
    "normalize_gpu_metrics",
    "normalize_lmcache_metrics",
    "parse_lmcache_jsonl",
    "parse_prometheus_file",
    "parse_prometheus_samples",
    "scrape_prometheus_endpoint",
    "scrape_prometheus_endpoints",
]

