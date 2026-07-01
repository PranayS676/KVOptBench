"""LMCache telemetry normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvoptbench.telemetry.metrics import MetricRecord, MissingMetric, TelemetrySnapshot
from kvoptbench.telemetry.prometheus import parse_prometheus_samples


DEFAULT_LMCACHE_ALIASES = {
    "lmcache_cache_hit_total": "lmcache_cache_hits",
    "lmcache_cache_hits_total": "lmcache_cache_hits",
    "cache_hit": "lmcache_cache_hits",
    "cache_hits": "lmcache_cache_hits",
    "hits": "lmcache_cache_hits",
    "lmcache_cache_miss_total": "lmcache_cache_misses",
    "lmcache_cache_misses_total": "lmcache_cache_misses",
    "cache_miss": "lmcache_cache_misses",
    "cache_misses": "lmcache_cache_misses",
    "misses": "lmcache_cache_misses",
    "lmcache_cache_hit_rate": "lmcache_cache_hit_rate",
    "cache_hit_rate": "lmcache_cache_hit_rate",
    "hit_rate": "lmcache_cache_hit_rate",
    "lmcache_cache_load_total": "lmcache_cache_loads",
    "lmcache_cache_loads_total": "lmcache_cache_loads",
    "cache_load": "lmcache_cache_loads",
    "cache_loads": "lmcache_cache_loads",
    "lmcache_cache_store_total": "lmcache_cache_stores",
    "lmcache_cache_stores_total": "lmcache_cache_stores",
    "cache_store": "lmcache_cache_stores",
    "cache_stores": "lmcache_cache_stores",
    "lmcache_kv_transfer_bytes_total": "lmcache_kv_transfer_bytes",
    "kv_transfer_bytes": "lmcache_kv_transfer_bytes",
    "transfer_bytes": "lmcache_kv_transfer_bytes",
    "lmcache_kv_transfer_seconds_sum": "lmcache_kv_transfer_ms",
    "kv_transfer_ms": "lmcache_kv_transfer_ms",
    "transfer_ms": "lmcache_kv_transfer_ms",
    "lmcache_offload_seconds_sum": "lmcache_offload_ms",
    "offload_ms": "lmcache_offload_ms",
    "lmcache_load_seconds_sum": "lmcache_load_ms",
    "load_ms": "lmcache_load_ms",
}


def normalize_lmcache_metrics(
    source: str | Path | dict[str, Any] | list[dict[str, Any]],
    *,
    expected_metrics: list[str] | tuple[str, ...] | None = None,
    metric_aliases: dict[str, str] | None = None,
) -> TelemetrySnapshot:
    """Normalize LMCache metrics from Prometheus text, JSON, or JSONL-like records."""
    records, source_type, source_path = _load_records(source)
    aliases = {**DEFAULT_LMCACHE_ALIASES, **(metric_aliases or {})}
    metrics: dict[str, float | None] = {}
    samples: list[dict[str, Any]] = []

    for record in records:
        normalized_name = aliases.get(record.name, aliases.get(record.raw_name or "", record.name))
        value = _convert_units(normalized_name, record.value, record.name)
        metrics[normalized_name] = value
        samples.append(
            {
                "name": normalized_name,
                "raw_name": record.raw_name or record.name,
                "value": value,
                "labels": record.labels,
                "source_type": record.source_type,
                "source_path": record.source_path,
            }
        )

    _derive_hit_rate(metrics)
    expected = list(expected_metrics or [])
    for metric in expected:
        metrics.setdefault(metric, None)

    missing_metrics = [
        MissingMetric(
            metric=metric,
            reason=f"{metric} was not present in LMCache telemetry.",
            source=source_path or source_type,
        )
        for metric in expected
        if metrics.get(metric) is None
    ]

    return TelemetrySnapshot(
        metrics=metrics,
        missing_metrics=missing_metrics,
        source_type=source_type,
        source_path=source_path,
        samples=samples,
    )


def parse_lmcache_jsonl(
    path: str | Path,
    *,
    expected_metrics: list[str] | tuple[str, ...] | None = None,
    metric_aliases: dict[str, str] | None = None,
) -> TelemetrySnapshot:
    """Normalize a structured LMCache JSONL telemetry export."""
    return normalize_lmcache_metrics(
        _read_jsonl_records(Path(path)),
        expected_metrics=expected_metrics,
        metric_aliases=metric_aliases,
    )


def _load_records(
    source: str | Path | dict[str, Any] | list[dict[str, Any]],
) -> tuple[list[MetricRecord], str, str | None]:
    if isinstance(source, dict):
        return _records_from_json_items([source], None), "lmcache_json", None
    if isinstance(source, list):
        return _records_from_json_items(source, None), "lmcache_json", None
    text, source_path = _read_text_source(source)
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return _records_from_json_items([json.loads(stripped)], source_path), "lmcache_json", source_path
    if stripped.startswith("["):
        payload = json.loads(stripped)
        items = payload if isinstance(payload, list) else []
        return _records_from_json_items(items, source_path), "lmcache_json", source_path
    if _looks_like_jsonl(text):
        return _records_from_json_items(_parse_jsonl_text(text), source_path), "lmcache_jsonl", source_path
    records = parse_prometheus_samples(text)
    return records, "lmcache_prometheus", source_path


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


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return _parse_jsonl_text(path.read_text(encoding="utf-8"))


def _looks_like_jsonl(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("{") for line in lines)


def _parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _records_from_json_items(items: list[Any], source_path: str | None) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        records.extend(_records_from_json_item(item, source_path))
    return records


def _records_from_json_item(item: dict[str, Any], source_path: str | None) -> list[MetricRecord]:
    if isinstance(item.get("metrics"), dict):
        return [
            _json_record(str(name), value, source_path)
            for name, value in item["metrics"].items()
            if _is_number_like(value)
        ]
    name = item.get("metric") or item.get("name") or item.get("metric_name")
    value = item.get("value")
    if name is None or not _is_number_like(value):
        return []
    return [_json_record(str(name), value, source_path)]


def _json_record(name: str, value: Any, source_path: str | None) -> MetricRecord:
    return MetricRecord(
        name=name,
        raw_name=name,
        value=float(value),
        source_type="lmcache_json",
        source_path=source_path,
    )


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _convert_units(normalized_name: str, value: float, raw_name: str) -> float:
    if normalized_name.endswith("_ms") and raw_name.endswith("_seconds_sum"):
        return round(value * 1000.0, 3)
    return value


def _derive_hit_rate(metrics: dict[str, float | None]) -> None:
    if metrics.get("lmcache_cache_hit_rate") is not None:
        return
    hits = metrics.get("lmcache_cache_hits")
    misses = metrics.get("lmcache_cache_misses")
    if hits is None or misses is None:
        return
    denominator = hits + misses
    if denominator <= 0:
        return
    metrics["lmcache_cache_hit_rate"] = round(hits / denominator, 6)
