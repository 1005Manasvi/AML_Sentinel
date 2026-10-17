"""Load processed parquet caches after verifying them against their manifest.

Security note:
    The parquet file is only read after its SHA-256 matches the value recorded by
    ``prepare-data``, so a swapped or corrupted cache fails loudly instead of
    silently changing results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.integrity import verify_checksum


def processed_paths(processed_dir: Path, dataset: DatasetName) -> tuple[Path, Path]:
    """Return the parquet and manifest paths for a dataset.

    Args:
        processed_dir: Directory written by ``prepare-data``.
        dataset: Dataset variant.

    Returns:
        ``(parquet_path, manifest_path)``.
    """
    return (
        processed_dir / f"{dataset.value}.parquet",
        processed_dir / f"{dataset.value}.manifest.json",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a processed-data manifest.

    Args:
        path: Manifest JSON path.

    Returns:
        The parsed manifest.

    Raises:
        DataNotFoundError: If the manifest is missing.
        DataValidationError: If it is not a JSON object.
    """
    if not path.is_file():
        raise DataNotFoundError(f"{path.name} not found; run `make data` or `make sample` first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DataValidationError(f"{path.name} is not a JSON object")
    return manifest


def load_processed(processed_dir: Path, dataset: DatasetName) -> pd.DataFrame:
    """Load a processed dataset after checksum verification.

    Args:
        processed_dir: Directory written by ``prepare-data``.
        dataset: Dataset variant.

    Returns:
        The processed transactions DataFrame.

    Raises:
        DataNotFoundError: If the parquet or manifest is missing.
        DataValidationError: If the manifest has no checksum for the parquet.
        IntegrityError: If the parquet does not match its recorded checksum.
    """
    parquet, manifest_path = processed_paths(processed_dir, dataset)
    manifest = load_manifest(manifest_path)
    expected = manifest.get("sha256", {}).get(parquet.name)
    if not isinstance(expected, str):
        raise DataValidationError(f"{manifest_path.name} has no checksum for {parquet.name}")
    verify_checksum(parquet, expected)
    return pd.read_parquet(parquet)
