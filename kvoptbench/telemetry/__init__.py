"""Telemetry adapter foundations."""

from kvoptbench.telemetry.metrics import MetricRecord, MissingMetric, TelemetrySnapshot
from kvoptbench.telemetry.nvidia_smi import GpuSampler, normalize_gpu_metrics
from kvoptbench.telemetry.prometheus import (
    PrometheusScrapeResult,
    parse_prometheus_file,
    parse_prometheus_samples,
    scrape_prometheus_endpoint,
    scrape_prometheus_endpoints,
)

__all__ = [
    "GpuSampler",
    "MetricRecord",
    "MissingMetric",
    "PrometheusScrapeResult",
    "TelemetrySnapshot",
    "normalize_gpu_metrics",
    "parse_prometheus_file",
    "parse_prometheus_samples",
    "scrape_prometheus_endpoint",
    "scrape_prometheus_endpoints",
]

