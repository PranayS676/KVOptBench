import zipfile
from pathlib import Path

import pytest

from kvoptbench.datasets.download import DatasetDownloadError, download_file, extract_zip


def test_download_file_reuses_existing_cache_without_network(tmp_path: Path) -> None:
    cached = tmp_path / "cached.txt"
    cached.write_text("already here", encoding="utf-8")

    result = download_file("https://example.test/dataset.txt", cached)

    assert result.reused_cache is True
    assert result.downloaded_at is None
    assert result.sha256


def test_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zip_handle:
        zip_handle.writestr("../bad.txt", "bad")

    with pytest.raises(DatasetDownloadError, match="Unsafe path"):
        extract_zip(archive, tmp_path / "out")
