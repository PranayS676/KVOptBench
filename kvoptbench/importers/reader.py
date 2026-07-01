"""Shared CSV/JSON/JSONL reader for offline benchmark importers."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceRecordSet:
    """Records and public-safe metadata for one imported source artifact."""

    records: list[dict[str, Any]]
    source_format: str
    source_file_name: str
    source_sha256: str


def read_source_records(source: str | Path) -> SourceRecordSet:
    """Read a supported source file into dictionaries without normalizing values."""
    source_path = Path(source)
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        records = _read_jsonl(source_path)
        source_format = "jsonl"
    elif suffix == ".json":
        records = _read_json(source_path)
        source_format = "json"
    elif suffix == ".csv":
        records = _read_csv(source_path)
        source_format = "csv"
    else:
        raise ValueError(f"Unsupported import format: {source_path.suffix}")

    if not records:
        raise ValueError(f"No import rows found in {source_path.name}")

    return SourceRecordSet(
        records=records,
        source_format=source_format,
        source_file_name=source_path.name,
        source_sha256=_sha256(source_path),
    )


def _read_jsonl(source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {source_path.name} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid JSONL in {source_path.name} on line {line_number}: "
                    "expected an object"
                )
            records.append(payload)
    return records


def _read_json(source_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source_path.name}: {exc.msg}") from exc

    if isinstance(payload, list):
        return _dict_items(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON in {source_path.name}: expected an object or list")

    for key in ["requests", "results", "rows", "samples", "data"]:
        value = payload.get(key)
        if isinstance(value, list):
            return _dict_items(value)

    for section in ["summary", "metrics", "result"]:
        value = payload.get(section)
        if isinstance(value, dict):
            return [value]

    return [payload]


def _read_csv(source_path: Path) -> list[dict[str, Any]]:
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV import source {source_path.name} has no header row")
        return [dict(row) for row in reader]


def _dict_items(items: list[Any]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict)]


def _sha256(source_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
