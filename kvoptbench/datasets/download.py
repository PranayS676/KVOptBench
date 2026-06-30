"""Network download helpers for optional public dataset ingestion."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from kvoptbench.datasets.hashing import sha256_file
from kvoptbench.schemas import utc_now_iso


class DatasetDownloadError(ValueError):
    """Raised when an optional dataset download cannot be completed."""


@dataclass(frozen=True)
class DownloadResult:
    """Metadata for one cached download."""

    path: Path
    url: str
    sha256: str | None
    downloaded_at: str | None
    reused_cache: bool = False


def download_file(
    url: str,
    output_path: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> DownloadResult:
    """Download one URL into the local cache."""
    target = Path(output_path)
    if target.exists() and not force:
        return DownloadResult(
            path=target,
            url=url,
            sha256=sha256_file(target),
            downloaded_at=None,
            reused_cache=True,
        )

    try:
        response = httpx.get(url, follow_redirects=True, timeout=timeout_seconds)
    except httpx.HTTPError as exc:
        raise DatasetDownloadError(f"Failed to download {url}: {exc}") from exc
    if response.status_code == 404:
        raise DatasetDownloadError(f"Dataset URL was not found: {url}")
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DatasetDownloadError(f"Failed to download {url}: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return DownloadResult(
        path=target,
        url=url,
        sha256=sha256_file(target),
        downloaded_at=utc_now_iso(),
        reused_cache=False,
    )


def download_first_available(
    urls: Iterable[str],
    output_path: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> DownloadResult:
    """Try candidate URLs until one succeeds."""
    errors: list[str] = []
    for url in urls:
        try:
            return download_file(
                url,
                output_path,
                force=force,
                timeout_seconds=timeout_seconds,
            )
        except DatasetDownloadError as exc:
            errors.append(str(exc))
    raise DatasetDownloadError("; ".join(errors) or "No dataset URLs were provided")


def extract_zip(zip_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a zip file without allowing path traversal."""
    archive = Path(zip_path)
    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    resolved_root = target_root.resolve()
    with zipfile.ZipFile(archive) as zip_handle:
        for member in zip_handle.infolist():
            target = (target_root / member.filename).resolve()
            if not target.is_relative_to(resolved_root):
                raise DatasetDownloadError(f"Unsafe path in dataset archive: {member.filename}")
        zip_handle.extractall(target_root)
    return target_root
