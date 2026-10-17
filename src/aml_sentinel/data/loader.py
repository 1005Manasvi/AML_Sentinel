"""Load and validate IBM AMLworld ``*_Trans.csv`` files into typed DataFrames.

Pipeline: check header -> parse with explicit dtypes -> coerce -> validate ->
derive account node IDs -> sort by time.

Security note:
    The CSV is parsed as data only (no ``eval``, no pickle). Bank codes are read as
    strings so leading zeros, which are part of the identifier, survive.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aml_sentinel.data.schema import (
    CURRENCIES,
    PAYMENT_FORMATS,
    RAW_COLUMNS,
    RAW_HEADER,
    TIMESTAMP_FORMAT,
    Col,
    DatasetName,
)
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError

logger = logging.getLogger(__name__)

_STRING_COLUMNS = (
    Col.FROM_BANK,
    Col.FROM_ACCOUNT,
    Col.TO_BANK,
    Col.TO_ACCOUNT,
    Col.RECEIVING_CURRENCY,
    Col.PAYMENT_CURRENCY,
    Col.PAYMENT_FORMAT,
)
_AMOUNT_COLUMNS = (Col.AMOUNT_RECEIVED, Col.AMOUNT_PAID)


def transactions_path(raw_dir: Path, dataset: DatasetName) -> Path:
    """Return the expected path of a dataset's transactions file.

    Args:
        raw_dir: Directory containing the downloaded files.
        dataset: Dataset variant.

    Returns:
        Path to ``<dataset>_Trans.csv``.
    """
    return raw_dir / f"{dataset.value}_Trans.csv"


def patterns_path(raw_dir: Path, dataset: DatasetName) -> Path:
    """Return the expected path of a dataset's laundering-patterns file.

    Args:
        raw_dir: Directory containing the downloaded files.
        dataset: Dataset variant.

    Returns:
        Path to ``<dataset>_Patterns.txt``.
    """
    return raw_dir / f"{dataset.value}_Patterns.txt"


def read_header(path: Path) -> tuple[str, ...]:
    """Read only the header row of a CSV file.

    Args:
        path: CSV file.

    Returns:
        Header fields, whitespace-stripped.

    Raises:
        DataNotFoundError: If the file does not exist.
        DataValidationError: If the file is empty.
    """
    if not path.is_file():
        raise DataNotFoundError(f"file not found: {path.name} (see data/README.md)")
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.reader(fh), None)
    if row is None:
        raise DataValidationError(f"{path.name} is empty")
    return tuple(field.strip() for field in row)


def load_transactions(path: Path) -> pd.DataFrame:
    """Load, coerce and validate a ``*_Trans.csv`` file.

    Args:
        path: Path to the CSV.

    Returns:
        DataFrame with canonical columns (see :class:`Col`), plus ``tx_id`` (the
        0-based row number in the source file, a stable transaction key) and
        ``from_id`` / ``to_id`` account node IDs, sorted by timestamp.

    Raises:
        DataNotFoundError: If the file does not exist.
        DataValidationError: If the header or any value violates the schema.
    """
    header = read_header(path)
    if header != RAW_HEADER:
        raise DataValidationError(f"{path.name}: unexpected header {header!r}")

    # The C engine, not pyarrow: pyarrow infers bank codes as integers *before*
    # applying the string dtype, silently stripping their leading zeros.
    frame = pd.read_csv(
        path,
        header=0,
        names=[c.value for c in RAW_COLUMNS],
        dtype=dict.fromkeys([Col.TIMESTAMP.value, *(c.value for c in _STRING_COLUMNS)], "string"),
        engine="c",
    )
    frame = coerce_transactions(frame)
    validate_transactions(frame)
    logger.info(
        "loaded transactions",
        extra={
            "file": path.name,
            "rows": len(frame),
            "positives": int(frame[Col.IS_LAUNDERING].sum()),
        },
    )
    return frame


def coerce_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw string/number columns to canonical dtypes and derive IDs.

    Args:
        frame: DataFrame with exactly the :data:`RAW_COLUMNS`.

    Returns:
        A new DataFrame (input is not modified) with typed columns, ``tx_id``,
        ``from_id`` and ``to_id``, stably sorted by timestamp.

    Raises:
        DataValidationError: If timestamps or numbers cannot be parsed.
    """
    out = frame.copy()
    out.insert(0, Col.TX_ID.value, np.arange(len(out), dtype=np.int64))
    try:
        out[Col.TIMESTAMP] = pd.to_datetime(out[Col.TIMESTAMP], format=TIMESTAMP_FORMAT)
        for col in _AMOUNT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(np.float64)
        labels = pd.to_numeric(out[Col.IS_LAUNDERING], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"unparseable value: {type(exc).__name__}") from None
    if not labels.isin([0, 1]).all():
        raise DataValidationError("is_laundering must be 0 or 1")
    out[Col.IS_LAUNDERING] = labels.astype(np.int8)

    for col in _STRING_COLUMNS:
        out[col] = out[col].astype("string").str.strip()
    out[Col.FROM_ID] = out[Col.FROM_BANK] + ":" + out[Col.FROM_ACCOUNT]
    out[Col.TO_ID] = out[Col.TO_BANK] + ":" + out[Col.TO_ACCOUNT]
    # Categories cut memory by roughly 5x on the multi-million-row files.
    for col in (*_STRING_COLUMNS, Col.FROM_ID, Col.TO_ID):
        out[col] = out[col].astype("category")

    return out.sort_values(Col.TIMESTAMP.value, kind="stable").reset_index(drop=True)


def validate_transactions(frame: pd.DataFrame) -> None:
    """Check value-level constraints on a coerced transactions DataFrame.

    Args:
        frame: Output of :func:`coerce_transactions`.

    Raises:
        DataValidationError: On missing columns, nulls, negative or non-finite
            amounts, unknown currencies/payment formats, or unsorted timestamps.
            Messages report counts only, never offending values.
    """
    required = {c.value for c in RAW_COLUMNS} | {Col.TX_ID, Col.FROM_ID, Col.TO_ID}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")

    null_counts = frame[sorted(required)].isna().sum()
    if int(null_counts.sum()) > 0:
        bad = null_counts[null_counts > 0].to_dict()
        raise DataValidationError(f"null values per column: {bad}")

    for col in _AMOUNT_COLUMNS:
        values = frame[col].to_numpy(dtype=np.float64)
        n_bad = int((~np.isfinite(values) | (values < 0)).sum())
        if n_bad:
            raise DataValidationError(f"{col}: {n_bad} negative or non-finite amounts")

    for col in (Col.RECEIVING_CURRENCY, Col.PAYMENT_CURRENCY):
        _check_allowed(frame[col], CURRENCIES, col)
    _check_allowed(frame[Col.PAYMENT_FORMAT], PAYMENT_FORMATS, Col.PAYMENT_FORMAT)

    if not frame[Col.TIMESTAMP].is_monotonic_increasing:
        raise DataValidationError("timestamps are not sorted")


def _check_allowed(series: pd.Series, allowed: frozenset[str], name: str) -> None:
    n_bad = int((~series.astype("string").isin(allowed)).sum())
    if n_bad:
        raise DataValidationError(f"{name}: {n_bad} rows with unknown values")
