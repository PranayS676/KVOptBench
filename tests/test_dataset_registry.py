import pytest

from kvoptbench.datasets.registry import get_dataset_adapter, list_dataset_adapters


def test_dataset_registry_lists_public_dataset_adapters() -> None:
    adapters = {adapter.name: adapter for adapter in list_dataset_adapters()}

    assert "qasper" in adapters
    assert "gutenberg" in adapters
    assert "longbench" in adapters
    assert "beir_scifact" in adapters
    assert "bfcl" in adapters
    assert "shared_prefix" in adapters["qasper"].supported_modes
    assert "needle" in adapters["gutenberg"].supported_modes
    assert "long_context_qa" in adapters["longbench"].supported_modes
    assert "rag" in adapters["beir_scifact"].supported_modes
    assert "tool_calling" in adapters["bfcl"].supported_modes


def test_dataset_registry_rejects_unknown_adapter() -> None:
    with pytest.raises(ValueError, match="Unknown dataset adapter"):
        get_dataset_adapter("missing")
