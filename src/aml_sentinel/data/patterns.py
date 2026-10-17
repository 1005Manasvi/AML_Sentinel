"""Parse ``*_Patterns.txt`` files: ground-truth laundering attempts with typology.

File format (one block per attempt)::

    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    <csv rows in the *_Trans.csv layout, without header>
    END LAUNDERING ATTEMPT - FAN-OUT

Only a subset of laundering transactions belongs to a documented attempt, so a
missing typology does **not** mean "benign".
"""

from __future__ import annotations

import csv
import re
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

from aml_sentinel.data.loader import coerce_transactions
from aml_sentinel.data.schema import RAW_COLUMNS, Col
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError

ATTEMPT_ID: Final = "attempt_id"
TYPOLOGY: Final = "typology"


class Typology(StrEnum):
    """Laundering typologies used by the IBM AMLworld simulator."""

    FAN_IN = "FAN-IN"
    FAN_OUT = "FAN-OUT"
    CYCLE = "CYCLE"
    GATHER_SCATTER = "GATHER-SCATTER"
    SCATTER_GATHER = "SCATTER-GATHER"
    BIPARTITE = "BIPARTITE"
    STACK = "STACK"
    RANDOM = "RANDOM"


_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)\s*(?::.*)?$")
_END = re.compile(r"^END LAUNDERING ATTEMPT - ([A-Z-]+)\s*$")
# Columns that identify a transaction row (everything except the label).
MATCH_KEY: Final = tuple(c.value for c in RAW_COLUMNS if c is not Col.IS_LAUNDERING)


def load_patterns(path: Path) -> pd.DataFrame:
    """Parse a patterns file into one row per (attempt, transaction).

    Args:
        path: Path to ``<dataset>_Patterns.txt``.

    Returns:
        Coerced transaction rows (same dtypes as :func:`load_transactions`) plus
        ``attempt_id`` (0-based, file order) and ``typology``. ``tx_id`` here is the
        row number within the patterns file, *not* the transactions file.

    Raises:
        DataNotFoundError: If the file does not exist.
        DataValidationError: On malformed or unbalanced BEGIN/END blocks.
    """
    if not path.is_file():
        raise DataNotFoundError(f"file not found: {path.name} (see data/README.md)")

    rows: list[list[str]] = []
    attempt_ids: list[int] = []
    typologies: list[str] = []
    current: str | None = None
    attempt = -1

    with path.open(newline="", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            if begin := _BEGIN.match(text):
                if current is not None:
                    raise DataValidationError(f"{path.name}:{lineno}: nested BEGIN")
                current = _parse_typology(begin.group(1), path, lineno)
                attempt += 1
            elif end := _END.match(text):
                if current is None or end.group(1) != current:
                    raise DataValidationError(f"{path.name}:{lineno}: unmatched END")
                current = None
            else:
                if current is None:
                    raise DataValidationError(f"{path.name}:{lineno}: row outside a block")
                fields = next(csv.reader([text]))
                if len(fields) != len(RAW_COLUMNS):
                    raise DataValidationError(f"{path.name}:{lineno}: wrong field count")
                rows.append(fields)
                attempt_ids.append(attempt)
                typologies.append(current)
    if current is not None:
        raise DataValidationError(f"{path.name}: unterminated block at end of file")

    raw = pd.DataFrame(rows, columns=[c.value for c in RAW_COLUMNS], dtype="string")
    raw[ATTEMPT_ID] = attempt_ids
    raw[TYPOLOGY] = typologies
    # coerce_transactions sorts by time; keep attempt info aligned via the frame.
    meta = raw[[ATTEMPT_ID, TYPOLOGY]]
    coerced = coerce_transactions(raw.drop(columns=[ATTEMPT_ID, TYPOLOGY]))
    coerced[ATTEMPT_ID] = meta[ATTEMPT_ID].to_numpy()[coerced[Col.TX_ID].to_numpy()]
    coerced[TYPOLOGY] = pd.Categorical(
        meta[TYPOLOGY].to_numpy()[coerced[Col.TX_ID].to_numpy()],
        categories=[t.value for t in Typology],
    )
    return coerced


def _parse_typology(name: str, path: Path, lineno: int) -> str:
    try:
        return Typology(name).value
    except ValueError:
        raise DataValidationError(f"{path.name}:{lineno}: unknown typology") from None


def attach_typology(transactions: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Label each transaction with the typology of the attempt it belongs to.

    Rows are matched on every raw field except the label. When one transaction
    appears in several attempts, the first attempt in file order wins.

    Args:
        transactions: Output of :func:`load_transactions`.
        patterns: Output of :func:`load_patterns`.

    Returns:
        Copy of ``transactions`` with ``attempt_id`` (nullable Int64) and
        ``typology`` (categorical, NaN when not in any documented attempt).
    """
    lookup = patterns.sort_values(ATTEMPT_ID, kind="stable").drop_duplicates(
        subset=list(MATCH_KEY), keep="first"
    )[[*MATCH_KEY, ATTEMPT_ID, TYPOLOGY]]
    key_types = {k: "string" for k in MATCH_KEY if k not in _NON_STRING_KEYS}
    left = transactions.astype(key_types)
    right = lookup.astype(key_types)
    merged = left.merge(right, how="left", on=list(MATCH_KEY), validate="many_to_one")
    out = transactions.copy()
    out[ATTEMPT_ID] = merged[ATTEMPT_ID].astype("Int64").to_numpy()
    out[TYPOLOGY] = pd.Categorical(
        merged[TYPOLOGY].astype("string").to_numpy(), categories=[t.value for t in Typology]
    )
    return out


_NON_STRING_KEYS: Final = frozenset(
    {Col.TIMESTAMP.value, Col.AMOUNT_PAID.value, Col.AMOUNT_RECEIVED.value}
)
