"""Telemetry adapter foundations."""

from kvoptbench.telemetry.metrics import MetricRecord, MissingMetric, TelemetrySnapshot
from kvoptbench.telemetry.nvidia_smi import normalize_gpu_metrics
from kvoptbench.telemetry.prometheus import parse_prometheus_file, parse_prometheus_samples

__all__ = [
    "MetricRecord",
    "MissingMetric",
    "TelemetrySnapshot",
    "normalize_gpu_metrics",
    "parse_prometheus_file",
    "parse_prometheus_samples",
]

