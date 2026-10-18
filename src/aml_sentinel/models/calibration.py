"""Probability calibration (Platt, isotonic) and calibration metrics.

Raw model scores are **not** probabilities. Every score shown to a user goes through a
calibrator fitted on the validation split only, and calibration quality is measured
on test.

Calibrators serialise to plain JSON (two floats for Platt, breakpoints for isotonic),
so loading one never unpickles anything.

Smoothed isotonic (Phase 8): plain isotonic regression gives a block that holds only
positives a probability of exactly 1.0, which overstates certainty for the riskiest
alerts (ADR-018). The smoothed variant shrinks each block towards the base rate with a
weak Beta prior of :data:`SMOOTHING_STRENGTH` pseudo-observations, then re-imposes
monotonicity, so no output is ever exactly 0 or 1. Whether it is used is decided by
validation cross-fitting like every other calibrator choice, never by test results.

ECE note: at 0.1% prevalence, equal-width bins put almost every row in the first
bin, so the headline ECE uses equal-mass ("quantile") bins. Equal-width ECE is
reported alongside it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from aml_sentinel.exceptions import ConfigurationError, DataValidationError

FloatArray = npt.NDArray[np.float64]
BinStrategy = Literal["quantile", "uniform"]
_EPS = 1e-7
# Two pseudo-observations at the base rate per isotonic block: a weak prior, fixed in
# advance (not tuned), that only matters for small blocks such as the top one.
SMOOTHING_STRENGTH: Final = 2.0


class Calibrator(Protocol):
    """Maps raw scores in [0, 1] to calibrated probabilities."""

    method: str

    def transform(self, scores: npt.ArrayLike) -> FloatArray:
        """Calibrate scores."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe types."""
        ...


def _logit(p: FloatArray) -> FloatArray:
    clipped = np.clip(p, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


def _check(scores: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    if s.shape != t.shape or s.size == 0:
        raise DataValidationError("scores and labels must be non-empty and equal length")
    if not np.isin(t, (0.0, 1.0)).all():
        raise DataValidationError("labels must be binary")
    if np.min(t) == np.max(t):
        raise DataValidationError("calibration needs both classes")
    return s, t


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """Logistic recalibration on the logit of the raw score: ``sigmoid(a*logit(s)+b)``.

    Attributes:
        a: Slope.
        b: Intercept.
        method: Always ``"platt"``.
    """

    a: float
    b: float
    method: str = field(default="platt", init=False)

    @classmethod
    def fit(cls, scores: npt.ArrayLike, y: npt.ArrayLike) -> PlattCalibrator:
        """Fit on validation scores and labels.

        Args:
            scores: Raw scores in [0, 1].
            y: Binary labels.

        Returns:
            The fitted calibrator.
        """
        s, t = _check(scores, y)
        # Very weak regularisation: two parameters, so plain maximum likelihood is the intent.
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(_logit(s).reshape(-1, 1), t)
        return cls(a=float(model.coef_[0, 0]), b=float(model.intercept_[0]))

    def transform(self, scores: npt.ArrayLike) -> FloatArray:
        """Calibrate scores.

        Args:
            scores: Raw scores in [0, 1].

        Returns:
            Calibrated probabilities.
        """
        z = self.a * _logit(np.asarray(scores, dtype=np.float64)) + self.b
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serialise.

        Returns:
            ``{"method": "platt", "a": ..., "b": ...}``.
        """
        return {"method": self.method, "a": self.a, "b": self.b}


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Monotone step-wise calibration, stored as interpolation breakpoints.

    Attributes:
        x: Increasing raw-score breakpoints.
        y: Calibrated probability at each breakpoint.
        prior_strength: Beta-prior pseudo-observations per block (0 = plain isotonic).
        method: ``"isotonic"``, or ``"isotonic_smoothed"`` when ``prior_strength > 0``.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    prior_strength: float = 0.0
    method: str = field(default="isotonic", init=False)

    def __post_init__(self) -> None:
        """Name the method after the smoothing setting."""
        if self.prior_strength > 0:
            object.__setattr__(self, "method", "isotonic_smoothed")

    @classmethod
    def fit(
        cls, scores: npt.ArrayLike, y: npt.ArrayLike, *, prior_strength: float = 0.0
    ) -> IsotonicCalibrator:
        """Fit on validation scores and labels.

        Args:
            scores: Raw scores in [0, 1].
            y: Binary labels.
            prior_strength: Pseudo-observations at the base rate added to every block.

        Returns:
            The fitted calibrator.
        """
        s, t = _check(scores, y)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, t)
        xs = np.asarray(iso.X_thresholds_, dtype=np.float64)
        ys = np.asarray(iso.y_thresholds_, dtype=np.float64)
        if prior_strength > 0:
            ys = _smooth_blocks(ys, np.asarray(iso.predict(s)), float(t.mean()), prior_strength)
        return cls(
            x=tuple(float(v) for v in xs),
            y=tuple(float(v) for v in ys),
            prior_strength=float(prior_strength),
        )

    def transform(self, scores: npt.ArrayLike) -> FloatArray:
        """Calibrate scores (clipped to the fitted range).

        Args:
            scores: Raw scores in [0, 1].

        Returns:
            Calibrated probabilities.
        """
        return np.interp(np.asarray(scores, dtype=np.float64), self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        """Serialise.

        Returns:
            ``{"method": "isotonic", "x": [...], "y": [...]}``.
        """
        out: dict[str, Any] = {"method": self.method, "x": list(self.x), "y": list(self.y)}
        if self.prior_strength > 0:
            out["prior_strength"] = self.prior_strength
        return out


def _smooth_blocks(
    ys: FloatArray, fitted: FloatArray, base_rate: float, strength: float
) -> FloatArray:
    """Shrink isotonic block values towards the base rate, keeping them monotone."""
    blocks = np.unique(ys)
    idx = np.clip(np.searchsorted(blocks, fitted), 0, blocks.size - 1)
    counts = np.bincount(idx, minlength=blocks.size).astype(np.float64)
    shrunk = (blocks * counts + strength * base_rate) / (counts + strength)
    # Small blocks shrink more than large ones, which can reverse their order; a
    # weighted isotonic pass restores monotonicity.
    mono = IsotonicRegression().fit(blocks, shrunk, sample_weight=counts + strength)
    smoothed = np.asarray(mono.predict(blocks), dtype=np.float64)
    return smoothed[np.searchsorted(blocks, ys)]


def calibrator_from_dict(data: dict[str, Any]) -> PlattCalibrator | IsotonicCalibrator:
    """Rebuild a calibrator from :meth:`to_dict` output.

    Args:
        data: Serialised calibrator.

    Returns:
        The calibrator.

    Raises:
        ConfigurationError: On an unknown method or malformed fields.
    """
    try:
        if data["method"] == "platt":
            return PlattCalibrator(a=float(data["a"]), b=float(data["b"]))
        if data["method"] in ("isotonic", "isotonic_smoothed"):
            x = tuple(float(v) for v in data["x"])
            y = tuple(float(v) for v in data["y"])
            if len(x) != len(y) or not x or any(b < a for a, b in itertools.pairwise(x)):
                raise ValueError("bad breakpoints")
            strength = float(data.get("prior_strength", 0.0))
            if (data["method"] == "isotonic_smoothed") != (strength > 0):
                raise ValueError("method and prior_strength disagree")
            return IsotonicCalibrator(x=x, y=y, prior_strength=strength)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid calibrator: {type(exc).__name__}") from None
    raise ConfigurationError("unknown calibration method")


Fitter = Callable[[npt.ArrayLike, npt.ArrayLike], PlattCalibrator | IsotonicCalibrator]


def _fit_smoothed(scores: npt.ArrayLike, y: npt.ArrayLike) -> IsotonicCalibrator:
    return IsotonicCalibrator.fit(scores, y, prior_strength=SMOOTHING_STRENGTH)


# The System 1 default. It is kept unchanged so that re-running Phases 3-4 reproduces
# their reported numbers; the final ranker also considers the smoothed variant.
FITTERS: dict[str, Fitter] = {"platt": PlattCalibrator.fit, "isotonic": IsotonicCalibrator.fit}
FITTERS_WITH_SMOOTHING: dict[str, Fitter] = {**FITTERS, "isotonic_smoothed": _fit_smoothed}


def select_method(
    scores: npt.ArrayLike, y: npt.ArrayLike, *, fitters: dict[str, Fitter] | None = None
) -> tuple[str, dict[str, float]]:
    """Choose Platt vs isotonic by time-ordered 2-fold cross-fitting on validation.

    Each method is fitted on one half of the validation rows (in time order) and scored
    by quantile-ECE on the other half, then the two directions are averaged. Picking by
    in-sample ECE would always favour isotonic, which can memorise its fitting data.

    Args:
        scores: Validation raw scores, in time order.
        y: Validation labels.
        fitters: Candidates (default :data:`FITTERS`).

    Returns:
        ``(best_method, {method: mean held-out ECE})``.
    """
    s, t = _check(scores, y)
    fitters = FITTERS if fitters is None else fitters
    half = s.size // 2
    folds = [(slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))]
    results: dict[str, float] = {}
    for name, fit in fitters.items():
        eces = []
        for fit_idx, eval_idx in folds:
            if np.min(t[fit_idx]) == np.max(t[fit_idx]) or t[eval_idx].size == 0:
                continue
            cal = fit(s[fit_idx], t[fit_idx])
            eces.append(expected_calibration_error(cal.transform(s[eval_idx]), t[eval_idx]))
        results[name] = float(np.mean(eces)) if eces else float("inf")
    return min(results, key=lambda k: results[k]), results


def _bin_ids(probs: FloatArray, n_bins: int, strategy: BinStrategy) -> npt.NDArray[np.int64]:
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, n_bins - 1)
    # Equal-mass bins by rank, so ties never collapse bins.
    order = np.argsort(probs, kind="stable")
    ids = np.empty(probs.size, dtype=np.int64)
    ids[order] = (np.arange(probs.size) * n_bins) // max(probs.size, 1)
    return ids


def reliability_table(
    probs: npt.ArrayLike, y: npt.ArrayLike, *, n_bins: int = 15, strategy: BinStrategy = "quantile"
) -> list[dict[str, float]]:
    """Per-bin mean predicted probability vs observed positive rate.

    Args:
        probs: Predicted probabilities.
        y: Binary labels.
        n_bins: Number of bins.
        strategy: ``"quantile"`` (equal mass) or ``"uniform"`` (equal width).

    Returns:
        One dict per non-empty bin with ``count``, ``mean_pred`` and ``frac_pos``.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    ids = _bin_ids(p, n_bins, strategy)
    rows = []
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            rows.append(
                {
                    "count": float(mask.sum()),
                    "mean_pred": float(p[mask].mean()),
                    "frac_pos": float(t[mask].mean()),
                }
            )
    return rows


def expected_calibration_error(
    probs: npt.ArrayLike, y: npt.ArrayLike, *, n_bins: int = 15, strategy: BinStrategy = "quantile"
) -> float:
    """Count-weighted mean |mean_pred - frac_pos| over bins.

    Args:
        probs: Predicted probabilities.
        y: Binary labels.
        n_bins: Number of bins.
        strategy: Binning strategy (see :func:`reliability_table`).

    Returns:
        The ECE (0 is perfect).
    """
    table = reliability_table(probs, y, n_bins=n_bins, strategy=strategy)
    total = sum(r["count"] for r in table)
    if total == 0:
        return 0.0
    return sum(r["count"] * abs(r["mean_pred"] - r["frac_pos"]) for r in table) / total


def brier_score(probs: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Mean squared error between probabilities and labels.

    Args:
        probs: Predicted probabilities.
        y: Binary labels.

    Returns:
        The Brier score.
    """
    p = np.asarray(probs, dtype=np.float64)
    return float(np.mean((p - np.asarray(y, dtype=np.float64)) ** 2))


def log_loss(probs: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Binary cross-entropy with clipping.

    Args:
        probs: Predicted probabilities.
        y: Binary labels.

    Returns:
        Mean log loss.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1 - _EPS)
    t = np.asarray(y, dtype=np.float64)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
