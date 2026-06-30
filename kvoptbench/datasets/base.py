"""Dataset adapter protocol."""

from __future__ import annotations

from typing import Protocol

from kvoptbench.datasets.manifest import (
    DatasetAdapterInfo,
    DatasetPrepareOptions,
    DatasetPrepareResult,
)


class DatasetAdapter(Protocol):
    """Protocol implemented by public dataset adapters."""

    info: DatasetAdapterInfo

    def prepare(self, options: DatasetPrepareOptions) -> DatasetPrepareResult:
        """Prepare a workload JSONL file and manifest."""
        ...
