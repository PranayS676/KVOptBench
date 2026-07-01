"""Offline Prometheus telemetry parsing helpers."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from kvoptbench.telemetry.metrics import MetricRecord, MissingMetric


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)"
    r"(?:\s+(?P<timestamp>[+-]?\d+(?:\.\d+)?))?$"
)


def collect_prometheus_metrics() -> dict:
    return {"reason": "Prometheus telemetry is not collected in local mock mode."}


class PrometheusScrapeResult(BaseModel):
    """Result of one bounded Prometheus text scrape."""

    source_name: str
    success: bool
    scrape_started_at: str
    scrape_finished_at: str
    records: list[MetricRecord] = Field(default_factory=list)
    missing_metrics: list[MissingMetric] = Field(default_factory=list)
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    raw_text: str | None = None


def scrape_prometheus_endpoint(
    url: str,
    *,
    source_name: str = "prometheus",
    timeout_seconds: float = 2.0,
    expected_metrics: list[str] | tuple[str, ...] | None = None,
    metric_aliases: dict[str, str] | None = None,
    client: Any | None = None,
) -> PrometheusScrapeResult:
    """Fetch and parse Prometheus text exposition from a configured endpoint.

    The optional ``client`` only needs a ``get(url, timeout=...)`` method, which keeps
    unit tests on fake HTTP clients and avoids any live network dependency.
    """
    started_at = _utc_now()
    expected = list(expected_metrics or [])
    try:
        response = _prometheus_get(url, timeout_seconds=timeout_seconds, client=client)
        response.raise_for_status()
        text = response.text
        records = normalize_prometheus_records(
            parse_prometheus_samples(text),
            metric_aliases=metric_aliases,
        )
        missing_metrics = _missing_expected_prometheus_metrics(
            records,
            expected_metrics=expected,
            source_name=source_name,
        )
        return PrometheusScrapeResult(
            source_name=source_name,
            success=True,
            scrape_started_at=started_at,
            scrape_finished_at=_utc_now(),
            records=records,
            missing_metrics=missing_metrics,
            status_code=getattr(response, "status_code", None),
            raw_text=text,
        )
    except httpx.TimeoutException as exc:
        return _failed_scrape_result(
            source_name=source_name,
            started_at=started_at,
            expected_metrics=expected,
            error_type="timeout",
            error_message=f"Prometheus scrape timed out for source '{source_name}': {exc}",
        )
    except httpx.HTTPStatusError as exc:
        return _failed_scrape_result(
            source_name=source_name,
            started_at=started_at,
            expected_metrics=expected,
            error_type="http_status",
            error_message=f"Prometheus scrape failed for source '{source_name}': {exc}",
            status_code=exc.response.status_code,
        )
    except httpx.HTTPError as exc:
        return _failed_scrape_result(
            source_name=source_name,
            started_at=started_at,
            expected_metrics=expected,
            error_type="http_error",
            error_message=f"Prometheus scrape failed for source '{source_name}': {exc}",
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return _failed_scrape_result(
            source_name=source_name,
            started_at=started_at,
            expected_metrics=expected,
            error_type="parse_error",
            error_message=f"Prometheus scrape could not be parsed for source '{source_name}': {exc}",
        )


def scrape_prometheus_endpoints(
    endpoints: list[dict[str, Any]],
    *,
    client: Any | None = None,
) -> list[PrometheusScrapeResult]:
    """Scrape multiple configured Prometheus endpoints without runner wiring."""
    results: list[PrometheusScrapeResult] = []
    for endpoint in endpoints:
        results.append(
            scrape_prometheus_endpoint(
                str(endpoint["url"]),
                source_name=str(endpoint.get("name") or "prometheus"),
                timeout_seconds=float(endpoint.get("timeout_seconds") or 2.0),
                expected_metrics=endpoint.get("expected_metrics"),
                metric_aliases=endpoint.get("metric_aliases"),
                client=client,
            )
        )
    return results


def normalize_prometheus_records(
    records: list[MetricRecord],
    *,
    metric_aliases: dict[str, str] | None = None,
) -> list[MetricRecord]:
    """Apply optional raw-to-normalized metric aliases after offline parsing."""
    aliases = metric_aliases or {}
    normalized: list[MetricRecord] = []
    for record in records:
        raw_name = record.raw_name or record.name
        normalized_name = aliases.get(raw_name, aliases.get(record.name, record.name))
        if normalized_name == record.name and raw_name == record.raw_name:
            normalized.append(record)
            continue
        normalized.append(record.model_copy(update={"name": normalized_name, "raw_name": raw_name}))
    return normalized


def parse_prometheus_samples(source: str | Path) -> list[MetricRecord]:
    """Parse Prometheus text exposition or JSON API-like samples without fetching.

    ``source`` may be a raw string payload or a path to a local fixture. The parser
    intentionally does not perform network requests or require a live Prometheus
    server.
    """
    text, source_path = _read_source(source)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return _parse_json_samples(stripped, source_path)
    return _parse_text_samples(text, source_path)


def parse_prometheus_file(path: str | Path) -> list[MetricRecord]:
    """Parse Prometheus samples from a local file."""
    return parse_prometheus_samples(Path(path))


def _prometheus_get(url: str, *, timeout_seconds: float, client: Any | None) -> Any:
    if client is not None:
        return client.get(url, timeout=timeout_seconds)
    return httpx.get(url, timeout=timeout_seconds)


def _failed_scrape_result(
    *,
    source_name: str,
    started_at: str,
    expected_metrics: list[str],
    error_type: str,
    error_message: str,
    status_code: int | None = None,
) -> PrometheusScrapeResult:
    missing = [
        MissingMetric(metric=metric, reason=error_message, source=source_name)
        for metric in expected_metrics
    ]
    if not missing:
        missing.append(
            MissingMetric(
                metric="prometheus_scrape",
                reason=error_message,
                source=source_name,
            )
        )
    return PrometheusScrapeResult(
        source_name=source_name,
        success=False,
        scrape_started_at=started_at,
        scrape_finished_at=_utc_now(),
        records=[],
        missing_metrics=missing,
        status_code=status_code,
        error_type=error_type,
        error_message=error_message,
    )


def _missing_expected_prometheus_metrics(
    records: list[MetricRecord],
    *,
    expected_metrics: list[str],
    source_name: str,
) -> list[MissingMetric]:
    present = {record.name for record in records}
    return [
        MissingMetric(
            metric=metric,
            reason=f"{metric} was not present in the Prometheus scrape.",
            source=source_name,
        )
        for metric in expected_metrics
        if metric not in present
    ]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_source(source: str | Path) -> tuple[str, str | None]:
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


def _parse_text_samples(text: str, source_path: str | None) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    metric_types: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            _record_metric_type(line, metric_types)
            continue

        match = _SAMPLE_RE.match(line)
        if match is None:
            continue

        name = match.group("name")
        records.append(
            MetricRecord(
                name=name,
                raw_name=name,
                value=_parse_float(match.group("value")),
                labels=_parse_labels(match.group("labels") or ""),
                metric_type=metric_types.get(name),
                timestamp=_parse_optional_float(match.group("timestamp")),
                source_type="prometheus_text",
                source_path=source_path,
            )
        )

    return records


def _record_metric_type(line: str, metric_types: dict[str, str]) -> None:
    parts = line.split()
    if len(parts) >= 4 and parts[1] == "TYPE":
        metric_types[parts[2]] = parts[3]


def _parse_json_samples(text: str, source_path: str | None) -> list[MetricRecord]:
    payload = json.loads(text)
    records: list[MetricRecord] = []
    for sample in _iter_json_samples(payload):
        records.extend(_record_from_json_sample(sample, source_path))
    return records


def _iter_json_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    result = payload.get("data", {}).get("result") if isinstance(payload.get("data"), dict) else None
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]

    for key in ["result", "samples", "metrics"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return [payload]


def _record_from_json_sample(sample: dict[str, Any], source_path: str | None) -> list[MetricRecord]:
    metric_payload = sample.get("metric")
    labels: dict[str, str] = {}
    name = sample.get("name") or sample.get("metric_name")

    if isinstance(metric_payload, dict):
        name = name or metric_payload.get("__name__")
        labels = {str(key): str(value) for key, value in metric_payload.items() if key != "__name__"}

    if isinstance(sample.get("labels"), dict):
        labels.update({str(key): str(value) for key, value in sample["labels"].items()})

    if not name:
        return []

    records: list[MetricRecord] = []
    value = sample.get("value")
    if isinstance(value, list) and len(value) >= 2:
        records.append(_json_record(str(name), value[1], labels, source_path, value[0]))
    elif value is not None:
        records.append(_json_record(str(name), value, labels, source_path, sample.get("timestamp")))

    values = sample.get("values")
    if isinstance(values, list):
        for item in values:
            if isinstance(item, list) and len(item) >= 2:
                records.append(_json_record(str(name), item[1], labels, source_path, item[0]))

    return records


def _json_record(
    name: str,
    value: Any,
    labels: dict[str, str],
    source_path: str | None,
    timestamp: Any,
) -> MetricRecord:
    return MetricRecord(
        name=name,
        raw_name=name,
        value=_parse_float(value),
        labels=dict(labels),
        timestamp=_parse_optional_float(timestamp),
        source_type="prometheus_json",
        source_path=source_path,
    )


def _parse_labels(payload: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in _split_label_items(payload):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        value = raw_value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        labels[key.strip()] = _unescape_label_value(value)
    return labels


def _split_label_items(payload: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    in_quotes = False
    escaped = False
    for char in payload:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
        if char == "," and not in_quotes:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())
    return items


def _unescape_label_value(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")


def _parse_float(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"+Inf", "Inf"}:
            return math.inf
        if normalized == "-Inf":
            return -math.inf
        if normalized == "NaN":
            return math.nan
        return float(normalized)
    return float(value)


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _parse_float(value)

