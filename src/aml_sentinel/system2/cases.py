"""Build bounded, past-only case subgraphs around escalated alerts.

A *case* is the set of transactions an investigator would pull for one alert:

* every transaction touching the alert's sender or receiver, then their
  counterparties, up to ``hops`` levels out;
* restricted to the ``lookback`` window ending at the alert;
* by default **nothing after the alert**. Rows are taken only up to the alert's
  own position in the time-sorted frame, the same no-lookahead rule as Phase 3.
  ``horizon`` > 0 models an investigation that happens later (for example, next
  day) and may see data up to ``alert time + horizon``. That is a different,
  explicitly labelled scenario, never the default (ADR-015).

Hub accounts have up to tens of thousands of counterparties, so each account
contributes at most ``max_rows_per_account`` of its most recent rows and the case
is capped at ``max_rows``. Truncation is recorded on the case, never silent.

Labels are never copied into a case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError

CASE_COLUMNS: Final = (
    Col.TX_ID.value,
    Col.TIMESTAMP.value,
    Col.FROM_ID.value,
    Col.TO_ID.value,
    Col.AMOUNT_PAID_USD.value,
    Col.PAYMENT_FORMAT.value,
)
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CaseConfig:
    """Case-extraction limits.

    Attributes:
        lookback: Window before the alert.
        hops: Neighbourhood depth around the alert's two accounts.
        max_rows_per_account: Most recent rows taken per account (hub cap).
        max_rows: Hard cap on rows per case.
        horizon: How far past the alert the investigation may look (0 = strict).
    """

    lookback: timedelta = timedelta(hours=72)
    hops: int = 2
    max_rows_per_account: int = 50
    max_rows: int = 3_000
    horizon: timedelta = timedelta(0)


@dataclass(frozen=True, slots=True)
class Case:
    """Transactions gathered for one alert.

    Attributes:
        alert_tx_id: The alert's transaction id.
        focal: The alert's sender and receiver account ids.
        edges: Case transactions (columns :data:`CASE_COLUMNS`), time-sorted.
        truncated: True if any cap removed rows.
    """

    alert_tx_id: int
    focal: tuple[str, str]
    edges: pd.DataFrame
    truncated: bool


class CaseIndex:
    """Account-indexed view of a time-sorted transaction frame for fast case lookups.

    Args:
        frame: Processed transactions sorted by time (``prepare-data`` output).

    Raises:
        DataValidationError: If the frame is not time-sorted or lacks columns.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(CASE_COLUMNS) - set(frame.columns))
        if missing:
            raise DataValidationError(f"case index missing columns: {missing}")
        if not frame[Col.TIMESTAMP].is_monotonic_increasing:
            raise DataValidationError("case index needs a time-sorted frame")
        self._frame = frame.reset_index(drop=True)
        both = pd.concat(
            [self._frame[Col.FROM_ID].astype("string"), self._frame[Col.TO_ID].astype("string")],
            ignore_index=True,
        )
        codes, self._accounts = pd.factorize(both)
        n = len(self._frame)
        self._src, self._dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
        self._code_of = {str(a): i for i, a in enumerate(self._accounts)}
        self._ts = self._frame[Col.TIMESTAMP].to_numpy()
        self._tx_pos = pd.Index(self._frame[Col.TX_ID])
        # Rows grouped by account, each group in time order (stable sort keeps it).
        self._out_order = np.argsort(self._src, kind="stable")
        self._in_order = np.argsort(self._dst, kind="stable")
        self._out_bounds = np.searchsorted(
            self._src[self._out_order], np.arange(len(self._accounts) + 1)
        )
        self._in_bounds = np.searchsorted(
            self._dst[self._in_order], np.arange(len(self._accounts) + 1)
        )

    def _rows_of(
        self, code: int, lo_ts: np.datetime64, hi_pos: int, hi_ts: np.datetime64 | None, cap: int
    ) -> tuple[IntArray, bool]:
        rows = np.concatenate(
            [
                self._out_order[self._out_bounds[code] : self._out_bounds[code + 1]],
                self._in_order[self._in_bounds[code] : self._in_bounds[code + 1]],
            ]
        )
        upper = rows <= hi_pos if hi_ts is None else self._ts[rows] <= hi_ts
        rows = rows[upper & (self._ts[rows] > lo_ts)]
        rows = np.unique(rows)  # sorted = time order
        return (rows[-cap:], True) if rows.size > cap else (rows, False)

    def build(self, alert_tx_id: int, config: CaseConfig | None = None) -> Case:
        """Gather the case for one alert.

        Args:
            alert_tx_id: ``tx_id`` of the alerted transaction.
            config: Extraction limits.

        Returns:
            The case.

        Raises:
            DataValidationError: If the transaction id is unknown.
        """
        config = config or CaseConfig()
        pos = int(self._tx_pos.get_indexer([alert_tx_id])[0])
        if pos < 0:
            raise DataValidationError("unknown alert transaction")
        lo = self._ts[pos] - np.timedelta64(int(config.lookback.total_seconds()), "s")
        hi_ts = (
            None
            if config.horizon <= timedelta(0)
            else self._ts[pos] + np.timedelta64(int(config.horizon.total_seconds()), "s")
        )
        focal = (int(self._src[pos]), int(self._dst[pos]))
        seen, frontier = set(focal), list(dict.fromkeys(focal))
        rows: set[int] = {pos}
        truncated = False
        for _ in range(config.hops):
            nxt: list[int] = []
            for code in frontier:
                r, cut = self._rows_of(code, lo, pos, hi_ts, config.max_rows_per_account)
                truncated |= cut
                rows.update(int(x) for x in r)
                for other in np.concatenate([self._src[r], self._dst[r]]):
                    if int(other) not in seen:
                        seen.add(int(other))
                        nxt.append(int(other))
                if len(rows) >= config.max_rows:
                    break
            if len(rows) >= config.max_rows:
                truncated = True
                break
            frontier = nxt
        chosen = np.sort(np.fromiter(rows, dtype=np.int64))
        if chosen.size > config.max_rows:
            # Keep the alert plus the most recent rows.
            chosen = np.union1d(chosen[-(config.max_rows - 1) :], [pos])
            truncated = True
        edges = self._frame.iloc[chosen][list(CASE_COLUMNS)].reset_index(drop=True)
        edges[Col.FROM_ID.value] = edges[Col.FROM_ID.value].astype("string")
        edges[Col.TO_ID.value] = edges[Col.TO_ID.value].astype("string")
        return Case(
            alert_tx_id=int(alert_tx_id),
            focal=(str(self._accounts[focal[0]]), str(self._accounts[focal[1]])),
            edges=edges,
            truncated=truncated,
        )
