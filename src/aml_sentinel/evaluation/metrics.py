"""Binary alerting metrics at the transaction level.

All metrics are computed from explicit confusion counts, so every reported number
can be recomputed by hand. Ratios with an empty denominator are ``None`` rather
than 0, because "no alerts" is not the same as "zero precision".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.exceptions import DataValidationError

UNDOCUMENTED = "undocumented"


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Confusion counts plus derived ratios for one alerting decision.

    Attributes:
        n: Number of transactions evaluated.
        positives: Number of laundering transactions.
        alerts: Number of transactions flagged.
        tp: Flagged laundering transactions.
        fp: Flagged benign transactions.
        fn: Missed laundering transactions.
        tn: Unflagged benign transactions.
    """

    n: int
    positives: int
    alerts: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        """TP / alerts, or ``None`` when nothing was flagged."""
        return _ratio(self.tp, self.alerts)

    @property
    def recall(self) -> float | None:
        """TP / positives, or ``None`` when there are no positives."""
        return _ratio(self.tp, self.positives)

    @property
    def false_positive_rate(self) -> float | None:
        """FP / negatives, or ``None`` when there are no negatives."""
        return _ratio(self.fp, self.fp + self.tn)

    @property
    def alert_rate(self) -> float | None:
        """Alerts / n, or ``None`` when ``n`` is 0."""
        return _ratio(self.alerts, self.n)

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall, or ``None`` if undefined."""
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict[str, Any]:
        """Return counts and ratios as a JSON-serialisable dict.

        Returns:
            Counts plus ``precision``, ``recall``, ``false_positive_rate``,
            ``alert_rate`` and ``f1``.
        """
        return {
            **asdict(self),
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
            "alert_rate": self.alert_rate,
            "f1": self.f1,
        }


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def binary_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> BinaryMetrics:
    """Compute confusion counts for binary labels and binary alerts.

    Args:
        y_true: Ground-truth labels (0/1 or bool).
        y_pred: Alert decisions (0/1 or bool), same length.

    Returns:
        The metrics.

    Raises:
        DataValidationError: If lengths differ or values are not binary.
    """
    t = _as_bool(y_true, "y_true")
    p = _as_bool(y_pred, "y_pred")
    if t.shape != p.shape:
        raise DataValidationError("y_true and y_pred have different lengths")
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    return BinaryMetrics(
        n=int(t.size),
        positives=int(t.sum()),
        alerts=int(p.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=int(t.size) - tp - fp - fn,
    )


def _as_bool(values: npt.ArrayLike, name: str) -> npt.NDArray[np.bool_]:
    arr = np.asarray(values)
    if arr.dtype != np.bool_:
        if not np.isin(arr, [0, 1]).all():
            raise DataValidationError(f"{name} must be binary")
        arr = arr.astype(np.bool_)
    return arr.ravel()


def recall_by_group(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: pd.Series
) -> dict[str, dict[str, float | int | None]]:
    """Recall of laundering transactions, broken down by group (e.g. typology).

    Args:
        y_true: Ground-truth labels.
        y_pred: Alert decisions.
        groups: Group label per row; missing values are reported as
            ``"undocumented"``.

    Returns:
        ``{group: {"positives": int, "caught": int, "recall": float | None}}`` for
        every group that has at least one positive.
    """
    t = _as_bool(y_true, "y_true")
    p = _as_bool(y_pred, "y_pred")
    labels = groups.astype("string").fillna(UNDOCUMENTED).to_numpy()
    out: dict[str, dict[str, float | int | None]] = {}
    for group in sorted(set(labels[t])):
        mask = t & (labels == group)
        positives, caught = int(mask.sum()), int((mask & p).sum())
        out[str(group)] = {
            "positives": positives,
            "caught": caught,
            "recall": _ratio(caught, positives),
        }
    return out
