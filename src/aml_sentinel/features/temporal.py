"""Past-only trailing-window aggregates (counts and sums) per entity.

For row ``i`` with entity ``k`` at minute ``t``, the window is ``(t - w, t]``. Rows in
the same minute are ordered by their position in the frame and only earlier ones
count, so no row ever sees anything after itself. Pass frames sorted by time, as
the loader produces them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class TrailingAggregates:
    """Trailing-window aggregates aligned with the input rows.

    Attributes:
        counts: Rows in the window, including the current one (>= 1).
        sums: Sum of the value column over the same rows (zeros if no values).
    """

    counts: npt.NDArray[np.int64]
    sums: npt.NDArray[np.float64]


def trailing_aggregates(
    frame: pd.DataFrame,
    *,
    key: str,
    window: timedelta,
    values: str | None = None,
) -> TrailingAggregates:
    """Compute past-only trailing counts (and optionally sums) per entity.

    Args:
        frame: Rows with ``timestamp`` and ``key`` (and ``values`` if given).
        key: Entity column, e.g. ``from_id``.
        window: Window length; minute resolution, at least one minute.
        values: Optional numeric column to sum over the window.

    Returns:
        Aggregates aligned with ``frame`` rows.

    Raises:
        DataValidationError: If ``key`` has missing values.
    """
    n = len(frame)
    if n == 0:
        return TrailingAggregates(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    codes, _ = pd.factorize(frame[key], use_na_sentinel=True)
    if (codes < 0).any():
        raise DataValidationError(f"{key} has missing values")
    minutes = frame[Col.TIMESTAMP].to_numpy().astype("datetime64[m]").astype(np.int64)
    minutes = minutes - np.min(minutes)
    win = max(1, int(window.total_seconds() // 60))
    stride = int(np.max(minutes)) + win + 1  # keeps different keys' ranges disjoint

    order = np.lexsort((np.arange(n), minutes, codes))
    sorted_keys = codes[order].astype(np.int64) * stride + minutes[order]
    first = np.searchsorted(sorted_keys, sorted_keys - win, side="right")
    positions = np.arange(n, dtype=np.int64)

    counts = np.empty(n, dtype=np.int64)
    counts[order] = positions - first + 1

    sums = np.zeros(n, dtype=np.float64)
    if values is not None:
        sorted_vals = frame[values].to_numpy(dtype=np.float64)[order]
        # Window sum = prefix[i + 1] - prefix[first]; one cumsum serves every row.
        prefix = np.concatenate([[0.0], np.cumsum(sorted_vals)])
        sums[order] = prefix[positions + 1] - prefix[first]
    return TrailingAggregates(counts, sums)
