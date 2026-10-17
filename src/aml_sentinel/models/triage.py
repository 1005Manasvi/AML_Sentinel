"""The common ``Triage`` protocol shared by every System 1 model.

Rules, LightGBM, Laya (Phase 4) and the benchmark-only Jev adapter all implement
:class:`Triage`, so they can be swapped, stacked by the fusion layer, or removed for
ablations without touching the callers.

Models that answer typed questions (Laya, Jev) fill the optional ``typology``,
``risk`` and ``escalate`` fields. Score-only models (rules, LightGBM) leave them unset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.models.calibration import Calibrator

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AlertBatch:
    """Inputs for one triage call; each model reads only what it needs.

    Attributes:
        features: Feature matrix (one row per transaction).
        transactions: Matching processed transaction rows, if the model needs raw
            fields (e.g. the rule engine needs bank codes).
        summaries: Plain-text alert summaries for text models (Phase 4).
        is_synthetic: True only when every row is known to be synthetic. It defaults
            to False, so data must be explicitly marked before any hosted third-party
            model (Jev) will accept it.
    """

    features: pd.DataFrame
    transactions: pd.DataFrame | None = None
    summaries: tuple[str, ...] | None = None
    is_synthetic: bool = False

    def __len__(self) -> int:
        """Number of transactions in the batch."""
        return len(self.features)


@dataclass(frozen=True, slots=True)
class TriageOutput:
    """Per-row triage results.

    Attributes:
        scores: Risk scores in [0, 1]. They are probabilities only if the model is
            wrapped in :class:`CalibratedTriage`.
        typology: Predicted typology per row (typed-question models only).
        risk: Risk on a 1-5 scale (typed-question models only).
        escalate: Escalate-to-System-2 decision (typed-question models only).
        extras: Additional per-row numeric outputs, e.g. per-label probabilities, which
            the fusion layer may use as inputs.
    """

    scores: FloatArray
    typology: tuple[str, ...] | None = None
    risk: npt.NDArray[np.int64] | None = None
    escalate: npt.NDArray[np.bool_] | None = None
    extras: dict[str, FloatArray] | None = None


@runtime_checkable
class Triage(Protocol):
    """A System 1 model: maps an :class:`AlertBatch` to a :class:`TriageOutput`."""

    @property
    def name(self) -> str:
        """Short identifier used in reports and ablation tables."""
        ...

    def triage(self, batch: AlertBatch) -> TriageOutput:
        """Score every row of ``batch``."""
        ...


class CalibratedTriage:
    """Wrap a triage model so its scores become calibrated probabilities.

    Args:
        inner: The model to wrap.
        calibrator: Calibrator fitted on the validation split.
    """

    def __init__(self, inner: Triage, calibrator: Calibrator) -> None:
        self._inner = inner
        self._calibrator = calibrator

    @property
    def name(self) -> str:
        """``<inner>+<method>``, e.g. ``lightgbm+isotonic``."""
        return f"{self._inner.name}+{self._calibrator.method}"

    def triage(self, batch: AlertBatch) -> TriageOutput:
        """Score with the inner model, then calibrate the scores.

        Args:
            batch: Input batch.

        Returns:
            The inner output with calibrated ``scores``.

        Raises:
            DataValidationError: If the inner model returns the wrong number of scores.
        """
        out = self._inner.triage(batch)
        if out.scores.shape != (len(batch),):
            raise DataValidationError("inner model returned wrong number of scores")
        return TriageOutput(
            scores=self._calibrator.transform(out.scores),
            typology=out.typology,
            risk=out.risk,
            escalate=out.escalate,
            extras=out.extras,
        )
