"""Interpretable fusion of System 1 signals into a calibrated ``p_triage``.

The stacker is a logistic regression over a handful of named inputs, for example the
logit of LightGBM's probability, the logit of Laya's P(escalate), Laya's expected
risk and P(benign). Its coefficients can be read directly, which is the point: an
analyst can see how much each signal moved the score.

Fitting uses the validation alerts only:

1. Time-ordered 2-fold cross-fitting produces out-of-fold stacker scores.
2. A calibrator (Platt or isotonic, chosen by :func:`select_method`) is fitted on
   those out-of-fold scores, so it never calibrates in-sample predictions.
3. The final stacker is refitted on all validation alerts.

The model serialises to JSON: coefficients, scaling and calibrator breakpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.models.calibration import (
    FITTERS,
    IsotonicCalibrator,
    PlattCalibrator,
    calibrator_from_dict,
    select_method,
)

FloatArray = npt.NDArray[np.float64]
_EPS = 1e-6


def logit(p: npt.ArrayLike) -> FloatArray:
    """Numerically safe logit.

    Args:
        p: Probabilities.

    Returns:
        ``log(p / (1 - p))`` with clipping.
    """
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def stack_inputs(columns: Mapping[str, npt.ArrayLike]) -> pd.DataFrame:
    """Assemble named fusion inputs, converting ``p_*`` probabilities to logits.

    Args:
        columns: ``{name: values}``; names starting with ``p_`` are treated as
            probabilities and logit-transformed, others are used as-is.

    Returns:
        Float DataFrame with columns ``logit_<name>`` or ``<name>``.

    Raises:
        DataValidationError: If lengths differ or values are non-finite.
    """
    lengths = {len(np.asarray(v)) for v in columns.values()}
    if len(lengths) != 1:
        raise DataValidationError("fusion inputs differ in length")
    out = {}
    for name, values in columns.items():
        arr = np.asarray(values, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise DataValidationError(f"fusion input {name} has non-finite values")
        out[f"logit_{name}" if name.startswith("p_") else name] = (
            logit(arr) if name.startswith("p_") else arr
        )
    return pd.DataFrame(out)


@dataclass(frozen=True, slots=True)
class FusionModel:
    """Fitted logistic-regression stacker plus calibrator.

    Attributes:
        inputs: Input column names, in order.
        mean: Per-input mean used for standardisation.
        scale: Per-input standard deviation used for standardisation.
        coef: Logistic-regression coefficients on standardised inputs.
        intercept: Logistic-regression intercept.
        calibrator: Calibrator fitted on out-of-fold stacker scores.
    """

    inputs: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    calibrator: PlattCalibrator | IsotonicCalibrator

    @staticmethod
    def _fit_lr(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit((x - mean) / scale, y)
        return mean, scale, lr.coef_[0].astype(np.float64), float(lr.intercept_[0])

    @staticmethod
    def _raw(
        x: FloatArray, mean: FloatArray, scale: FloatArray, coef: FloatArray, intercept: float
    ) -> FloatArray:
        z = ((x - mean) / scale) @ coef + intercept
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    @classmethod
    def fit(cls, inputs: pd.DataFrame, y: npt.ArrayLike) -> FusionModel:
        """Fit the stacker and its calibrator (see :meth:`fit_oof`).

        Args:
            inputs: Validation-alert inputs, time-ordered.
            y: Labels.

        Returns:
            The fitted fusion model.
        """
        return cls.fit_oof(inputs, y)[0]

    @classmethod
    def fit_oof(cls, inputs: pd.DataFrame, y: npt.ArrayLike) -> tuple[FusionModel, FloatArray]:
        """Fit the stacker and its calibrator on validation alerts (in time order).

        Args:
            inputs: Output of :func:`stack_inputs` for validation alerts, time-ordered.
            y: Labels.

        Returns:
            ``(model, oof)``: the model refitted on all rows, plus calibrated
            **out-of-fold** scores for the fitting rows. Downstream models fitted
            on the same rows must use ``oof``, never in-sample predictions.

        Raises:
            DataValidationError: If there are too few rows or only one class.
        """
        x = inputs.to_numpy(dtype=np.float64)
        t = np.asarray(y, dtype=np.float64)
        if x.shape[0] < 20 or np.min(t) == np.max(t):
            raise DataValidationError("fusion needs >= 20 rows and both classes")
        half = x.shape[0] // 2
        oof = np.empty(x.shape[0], dtype=np.float64)
        for fit_idx, pred_idx in (
            (slice(0, half), slice(half, None)),
            (slice(half, None), slice(0, half)),
        ):
            if np.min(t[fit_idx]) == np.max(t[fit_idx]):
                raise DataValidationError("a fusion fold has only one class")
            params = cls._fit_lr(x[fit_idx], t[fit_idx])
            oof[pred_idx] = cls._raw(x[pred_idx], *params)
        method, _ = select_method(oof, t)
        calibrator = FITTERS[method](oof, t)
        mean, scale, coef, intercept = cls._fit_lr(x, t)
        model = cls(
            inputs=tuple(inputs.columns),
            mean=tuple(float(v) for v in mean),
            scale=tuple(float(v) for v in scale),
            coef=tuple(float(v) for v in coef),
            intercept=intercept,
            calibrator=calibrator,
        )
        return model, calibrator.transform(oof)

    def predict(self, inputs: pd.DataFrame) -> FloatArray:
        """Calibrated ``p_triage``.

        Args:
            inputs: Output of :func:`stack_inputs` with the same columns.

        Returns:
            Calibrated probabilities.

        Raises:
            DataValidationError: If the input columns differ from training.
        """
        if tuple(inputs.columns) != self.inputs:
            raise DataValidationError("fusion inputs do not match the fitted model")
        raw = self._raw(
            inputs.to_numpy(dtype=np.float64),
            np.array(self.mean),
            np.array(self.scale),
            np.array(self.coef),
            self.intercept,
        )
        return self.calibrator.transform(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe types.

        Returns:
            Plain dict; inverse of :meth:`from_dict`.
        """
        return {
            "inputs": list(self.inputs),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coef": list(self.coef),
            "intercept": self.intercept,
            "calibrator": self.calibrator.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FusionModel:
        """Rebuild from :meth:`to_dict` output.

        Args:
            data: Serialised model.

        Returns:
            The model.

        Raises:
            DataValidationError: If the arrays have inconsistent lengths.
        """
        n = len(data["inputs"])
        if not len(data["mean"]) == len(data["scale"]) == len(data["coef"]) == n:
            raise DataValidationError("fusion model arrays have inconsistent lengths")
        return cls(
            inputs=tuple(str(v) for v in data["inputs"]),
            mean=tuple(float(v) for v in data["mean"]),
            scale=tuple(float(v) for v in data["scale"]),
            coef=tuple(float(v) for v in data["coef"]),
            intercept=float(data["intercept"]),
            calibrator=calibrator_from_dict(dict(data["calibrator"])),
        )

    def explain(self) -> dict[str, float]:
        """Standardised coefficients by input name (larger magnitude = more influence).

        Returns:
            ``{input: coefficient}``.
        """
        return dict(zip(self.inputs, self.coef, strict=True))
