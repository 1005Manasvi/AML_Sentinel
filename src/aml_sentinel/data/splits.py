"""Time-based train / validation / test splits and leakage guards.

Splits are half-open intervals on the transaction timestamp::

    train: [-inf,      train_end)
    val:   [train_end, val_end)     <- fusion / calibration fitting only
    test:  [val_end,   test_end)
    tail:  [test_end,  +inf)        <- excluded from evaluation

Why a tail: in both IBM AMLworld files, normal activity stops after 2022-09-10 while
laundering continues, so the last days hold a few hundred rows that are mostly
positives. Evaluating on them would report a meaningless precision (ADR-003).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import ConfigurationError, LeakageError


class SplitName(StrEnum):
    """Names of the time-based partitions."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """Exclusive upper boundaries of the train, validation and test periods.

    Attributes:
        train_end: First timestamp *not* in train.
        val_end: First timestamp *not* in validation.
        test_end: First timestamp *not* in test.
    """

    train_end: datetime
    val_end: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        """Reject boundaries that are not strictly increasing.

        Raises:
            ConfigurationError: If ``train_end < val_end < test_end`` does not hold.
        """
        if not self.train_end < self.val_end < self.test_end:
            raise ConfigurationError("split boundaries must be strictly increasing")

    @classmethod
    def from_settings(cls, settings: Settings) -> TimeSplit:
        """Build the split from application settings.

        Args:
            settings: Validated settings.

        Returns:
            The configured split.
        """
        return cls(settings.train_end, settings.val_end, settings.test_end)

    def name_of(self, timestamps: pd.Series) -> pd.Series:
        """Assign every timestamp to a partition.

        Args:
            timestamps: Datetime series.

        Returns:
            Categorical series of :class:`SplitName` values, aligned with the input.
        """
        inner = [pd.Timestamp(t) for t in (self.train_end, self.val_end, self.test_end)]
        bins = [pd.Timestamp.min, *inner, pd.Timestamp.max]
        labels = [s.value for s in SplitName]
        # right=False makes every bin half-open [a, b), matching the docstring.
        return pd.cut(timestamps, bins=bins, labels=labels, right=False)

    def partition(self, frame: pd.DataFrame) -> dict[SplitName, pd.DataFrame]:
        """Split a transactions frame into time partitions.

        Args:
            frame: DataFrame with a ``timestamp`` column.

        Returns:
            Mapping from every :class:`SplitName` to its rows (index reset).
        """
        names = self.name_of(frame[Col.TIMESTAMP])
        return {
            split: frame.loc[(names == split.value).to_numpy()].reset_index(drop=True)
            for split in SplitName
        }


def history_before(frame: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Return only rows strictly earlier than ``cutoff``.

    Every statistic fitted for a period (FX rates, feature scalers, model weights,
    calibrators) must be computed from ``history_before(frame, period_start)`` or
    from that period's own training partition.

    Args:
        frame: DataFrame with a ``timestamp`` column.
        cutoff: Exclusive upper bound.

    Returns:
        The filtered rows (a view-safe copy).
    """
    return frame.loc[frame[Col.TIMESTAMP] < cutoff].copy()


def assert_no_future_rows(frame: pd.DataFrame, cutoff: datetime, *, what: str) -> None:
    """Raise if ``frame`` contains any row at or after ``cutoff``.

    Args:
        frame: DataFrame with a ``timestamp`` column.
        cutoff: Exclusive upper bound that must hold.
        what: Short description for the error message (e.g. ``"fx fit data"``).

    Raises:
        LeakageError: If any timestamp is ``>= cutoff``.
    """
    n_future = int((frame[Col.TIMESTAMP] >= cutoff).sum())
    if n_future:
        raise LeakageError(f"{what}: {n_future} rows at or after the cutoff")
