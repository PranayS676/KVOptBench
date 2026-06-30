"""Registry for dataset adapters."""

from __future__ import annotations

from kvoptbench.datasets.base import DatasetAdapter
from kvoptbench.datasets.gutenberg import GutenbergAdapter
from kvoptbench.datasets.manifest import DatasetAdapterInfo
from kvoptbench.datasets.qasper import QasperAdapter


def _adapters() -> dict[str, DatasetAdapter]:
    adapters: list[DatasetAdapter] = [QasperAdapter(), GutenbergAdapter()]
    return {adapter.info.name: adapter for adapter in adapters}


def list_dataset_adapters() -> list[DatasetAdapterInfo]:
    """List public dataset adapters known to the CLI."""
    return [adapter.info for adapter in _adapters().values()]


def get_dataset_adapter(name: str) -> DatasetAdapter:
    """Return a dataset adapter by name."""
    adapters = _adapters()
    if name not in adapters:
        valid = ", ".join(sorted(adapters))
        raise ValueError(f"Unknown dataset adapter '{name}'. Valid adapters: {valid}")
    return adapters[name]
