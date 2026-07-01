"""Shared telemetry records for offline adapter normalization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MissingMetric(BaseModel):
    """One metric that was unavailable in a supplied telemetry artifact."""

    metric: str
    reason: str
    source: str | None = None


class MetricRecord(BaseModel):
    """One normalized metric sample from a telemetry source."""

    model_config = ConfigDict(extra="allow")

    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)
    source_type: str
    source_path: str | None = None
    raw_name: str | None = None
    metric_type: str | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TelemetrySnapshot(BaseModel):
    """Normalized metric values plus explicit missing metric reasons."""

    model_config = ConfigDict(extra="allow")

    metrics: dict[str, float | None]
    missing_metrics: list[MissingMetric] = Field(default_factory=list)
    source_type: str
    source_path: str | None = None
    samples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
