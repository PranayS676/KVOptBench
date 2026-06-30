"""Local cache helpers for public dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvoptbench.datasets.hashing import sha256_file

DEFAULT_DATASET_CACHE_DIR = Path("data/raw")


def resolve_dataset_cache_dir(source: str, cache_dir: Path | None = None) -> Path:
    """Return the cache directory for one dataset source."""
    root = cache_dir or DEFAULT_DATASET_CACHE_DIR
    return Path(root) / source


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON file containing either a list or an object with records."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("records") or [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list of records in {path}")
    return [dict(record) for record in payload]


def write_json_payload(path: str | Path, payload: Any, *, force: bool = False) -> Path:
    """Write a JSON payload to cache unless an existing file should be reused."""
    target = Path(path)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def write_json_records(path: str | Path, records: list[dict[str, Any]], *, force: bool = False) -> Path:
    """Write JSON records to cache."""
    return write_json_payload(path, records, force=force)


def cache_file_sha256(path: str | Path) -> str | None:
    """Return a file hash when the cache file exists."""
    target = Path(path)
    if not target.exists() or target.is_dir():
        return None
    return sha256_file(target)
