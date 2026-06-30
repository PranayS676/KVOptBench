"""Optional Hugging Face dataset loader."""

from __future__ import annotations

import json
from typing import Any

from kvoptbench.datasets.download import DatasetDownloadError


def load_hf_dataset_records(
    dataset_name: str,
    *,
    split: str,
    subset: str | None = None,
    revision: str | None = None,
    max_items: int | None = None,
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    """Load records from Hugging Face when optional data dependencies are installed."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetDownloadError(
            "Hugging Face dataset downloads require optional data dependencies. "
            "Install with `pip install -e .[data]`."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        dataset = (
            load_dataset(dataset_name, subset, **kwargs)
            if subset is not None
            else load_dataset(dataset_name, **kwargs)
        )
    except Exception as exc:
        raise DatasetDownloadError(
            f"Failed to load Hugging Face dataset {dataset_name}"
            f"{'/' + subset if subset else ''}: {exc}"
        ) from exc

    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if max_items is not None and index >= max_items:
            break
        records.append(_json_safe_dict(row))
    return records


def _json_safe_dict(row: Any) -> dict[str, Any]:
    """Convert dataset rows into plain JSON-compatible dictionaries."""
    return dict(json.loads(json.dumps(dict(row), ensure_ascii=False, default=str)))
