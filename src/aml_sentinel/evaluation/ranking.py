"""Ranking metrics and fair comparisons against the rule baseline.

A model that alerts on more transactions trivially catches more laundering, so
"false-positive reduction" is reported only at **matched operating points**:

* matched alert volume: the model flags exactly as many transactions as the baseline;
* matched recall: the model flags the fewest transactions that reach the baseline's
  recall.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from aml_sentinel.evaluation.metrics import BinaryMetrics, binary_metrics
from aml_sentinel.exceptions import DataValidationError

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _labels(y: npt.ArrayLike) -> BoolArray:
    return np.asarray(y).ravel().astype(bool)


def top_k_mask(scores: npt.ArrayLike, k: int) -> BoolArray:
    """Flag the ``k`` highest-scoring rows (ties broken by row order).

    Args:
        scores: Model scores.
        k: Number of rows to flag; clipped to ``[0, n]``.

    Returns:
        Boolean alert mask.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    k = int(min(max(k, 0), s.size))
    mask = np.zeros(s.size, dtype=bool)
    if k:
        # Stable sort on -score keeps the earliest row first among equal scores.
        mask[np.argsort(-s, kind="stable")[:k]] = True
    return mask


def ranking_summary(y: npt.ArrayLike, scores: npt.ArrayLike) -> dict[str, float | None]:
    """Threshold-free ranking quality.

    Args:
        y: Binary labels.
        scores: Scores (any monotone transform gives the same values).

    Returns:
        ``average_precision`` (PR-AUC), ``roc_auc`` and ``base_rate``; AUCs are
        ``None`` when only one class is present.
    """
    t = _labels(y)
    s = np.asarray(scores, dtype=np.float64).ravel()
    both = 0 < t.sum() < t.size
    return {
        "average_precision": float(average_precision_score(t, s)) if both else None,
        "roc_auc": float(roc_auc_score(t, s)) if both else None,
        "base_rate": float(t.mean()) if t.size else None,
    }


def precision_at_k(
    y: npt.ArrayLike, scores: npt.ArrayLike, ks: Sequence[int]
) -> dict[str, dict[str, Any]]:
    """Precision and recall when analysts review only the top ``k`` alerts.

    Args:
        y: Binary labels.
        scores: Model scores.
        ks: Review budgets.

    Returns:
        ``{str(k): metrics_dict}``.
    """
    t = _labels(y)
    return {str(k): binary_metrics(t, top_k_mask(scores, k)).to_dict() for k in ks}


def _comparison(model: BinaryMetrics, baseline: BinaryMetrics) -> dict[str, Any]:
    fp_reduction = None if baseline.fp == 0 else 1.0 - model.fp / baseline.fp
    return {
        "model": model.to_dict(),
        "baseline": baseline.to_dict(),
        "fp_reduction": fp_reduction,
    }


def compare_at_matched_volume(
    y: npt.ArrayLike, scores: npt.ArrayLike, baseline_alerts: npt.ArrayLike
) -> dict[str, Any]:
    """Compare model and baseline when both raise the same number of alerts.

    Args:
        y: Binary labels.
        scores: Model scores.
        baseline_alerts: Baseline alert mask on the same rows.

    Returns:
        Model and baseline metrics plus ``fp_reduction`` (``1 - FP_model/FP_base``).

    Raises:
        DataValidationError: On length mismatch.
    """
    t, base = _labels(y), _labels(baseline_alerts)
    if t.shape != base.shape:
        raise DataValidationError("labels and baseline alerts differ in length")
    model = binary_metrics(t, top_k_mask(scores, int(base.sum())))
    return _comparison(model, binary_metrics(t, base))


def compare_at_matched_recall(
    y: npt.ArrayLike, scores: npt.ArrayLike, baseline_alerts: npt.ArrayLike
) -> dict[str, Any]:
    """Compare at the smallest model alert count that reaches the baseline's recall.

    Args:
        y: Binary labels.
        scores: Model scores.
        baseline_alerts: Baseline alert mask on the same rows.

    Returns:
        Model and baseline metrics plus ``fp_reduction``; ``reachable`` is false if
        the model cannot reach the baseline recall at any alert count.
    """
    t, base = _labels(y), _labels(baseline_alerts)
    if t.shape != base.shape:
        raise DataValidationError("labels and baseline alerts differ in length")
    baseline = binary_metrics(t, base)
    needed = baseline.tp
    s = np.asarray(scores, dtype=np.float64).ravel()
    caught = np.cumsum(t[np.argsort(-s, kind="stable")])
    reachable = needed > 0 and caught.size > 0 and caught[-1] >= needed
    k = int(np.searchsorted(caught, needed) + 1) if reachable else t.size
    out = _comparison(binary_metrics(t, top_k_mask(s, k)), baseline)
    out["reachable"] = bool(reachable)
    return out


def operating_points(
    y: npt.ArrayLike, scores: npt.ArrayLike, alert_rates: Sequence[float]
) -> dict[str, dict[str, Any]]:
    """Metrics when escalating the top ``rate`` fraction of transactions.

    Args:
        y: Binary labels.
        scores: Model scores.
        alert_rates: Fractions in (0, 1], e.g. the 1-5% escalation target.

    Returns:
        ``{str(rate): metrics_dict}``.
    """
    t = _labels(y)
    return {
        str(r): binary_metrics(t, top_k_mask(scores, round(r * t.size))).to_dict()
        for r in alert_rates
    }


def pr_curve_points(
    y: npt.ArrayLike, scores: npt.ArrayLike, *, max_points: int = 400
) -> dict[str, list[float]]:
    """Down-sampled precision-recall curve for plotting.

    Args:
        y: Binary labels.
        scores: Model scores.
        max_points: Maximum points kept.

    Returns:
        ``{"precision": [...], "recall": [...]}``.
    """
    precision, recall, _ = precision_recall_curve(_labels(y), np.asarray(scores, dtype=np.float64))
    # Sample on an even *recall* grid. sklearn's points crowd the high-recall end,
    # so index-even sampling would draw the high-precision region as one straight line.
    grid = np.linspace(0.0, 1.0, max(2, max_points))
    # recall is non-increasing, so searching its negation finds, for each grid value,
    # the first point whose recall has dropped to (or below) it.
    idx = np.unique(np.clip(np.searchsorted(-recall, -grid, side="left"), 0, recall.size - 1))
    return {"precision": precision[idx].tolist(), "recall": recall[idx].tolist()}
