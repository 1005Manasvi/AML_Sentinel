"""LightGBM tabular triage model (``p_tabular``), with checksum-verified persistence.

The model is saved in LightGBM's native text format, never pickled, together with a
JSON manifest holding the feature order, parameters and the file's SHA-256. Loading
refuses a file whose hash does not match the manifest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.features.builder import CATEGORICAL_FEATURES
from aml_sentinel.integrity import sha256_file, verify_checksum
from aml_sentinel.models.triage import AlertBatch, TriageOutput

logger = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LgbmParams:
    """Training hyperparameters (fixed rather than searched, to keep the test split clean).

    Attributes:
        num_leaves: Maximum leaves per tree.
        learning_rate: Shrinkage.
        max_rounds: Upper bound on boosting rounds.
        early_stopping_rounds: Patience on the early-stopping slice.
        min_child_samples: Minimum rows per leaf (guards against memorising rare rows).
        feature_fraction: Column subsampling.
        bagging_fraction: Row subsampling.
        lambda_l2: L2 regularisation.
        seed: Random seed.
        num_threads: Fixed thread count; together with ``deterministic`` it makes runs
            reproducible.
    """

    num_leaves: int = 63
    learning_rate: float = 0.05
    max_rounds: int = 2000
    early_stopping_rounds: int = 100
    min_child_samples: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    seed: int = 42
    num_threads: int = 4

    def to_lgb(self) -> dict[str, Any]:
        """Translate to LightGBM's parameter dict.

        Returns:
            Parameters for :func:`lightgbm.train`.
        """
        return {
            "objective": "binary",
            "metric": "average_precision",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "num_threads": self.num_threads,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }


class LightGBMTriage:
    """Gradient-boosted trees over System 1 features; implements ``Triage``.

    Args:
        booster: Trained booster.
        features: Input column order.
        variant: Short tag, e.g. ``"all"`` or ``"no_shortcuts"``.
    """

    def __init__(self, booster: lgb.Booster, features: list[str], variant: str) -> None:
        self._booster = booster
        self.features = features
        self.variant = variant

    @property
    def name(self) -> str:
        """``lightgbm[<variant>]``."""
        return f"lightgbm[{self.variant}]"

    @property
    def best_iteration(self) -> int:
        """Boosting round chosen by early stopping."""
        return int(self._booster.best_iteration or self._booster.current_iteration())

    @classmethod
    def fit(
        cls,
        x_fit: pd.DataFrame,
        y_fit: npt.ArrayLike,
        x_stop: pd.DataFrame,
        y_stop: npt.ArrayLike,
        *,
        features: list[str],
        variant: str,
        params: LgbmParams | None = None,
    ) -> LightGBMTriage:
        """Train with early stopping on a held-out *training-period* slice.

        Early stopping uses the last training day rather than validation, so the
        validation split stays untouched for calibration and fusion.

        Args:
            x_fit: Features for fitting.
            y_fit: Labels for fitting.
            x_stop: Features for early stopping.
            y_stop: Labels for early stopping.
            features: Columns to use, in order.
            variant: Variant tag.
            params: Hyperparameters.

        Returns:
            The trained model.

        Raises:
            DataValidationError: If a feature column is missing.
        """
        params = params or LgbmParams()
        missing = sorted(set(features) - set(x_fit.columns))
        if missing:
            raise DataValidationError(f"missing feature columns: {missing}")
        cats = [c for c in CATEGORICAL_FEATURES if c in features]
        train_set = lgb.Dataset(
            x_fit[features], label=np.asarray(y_fit), categorical_feature=cats, free_raw_data=True
        )
        stop_set = lgb.Dataset(
            x_stop[features],
            label=np.asarray(y_stop),
            categorical_feature=cats,
            reference=train_set,
        )
        booster = lgb.train(
            params.to_lgb(),
            train_set,
            num_boost_round=params.max_rounds,
            valid_sets=[stop_set],
            callbacks=[lgb.early_stopping(params.early_stopping_rounds, verbose=False)],
        )
        model = cls(booster, features, variant)
        logger.info("trained lightgbm", extra={"variant": variant, "rounds": model.best_iteration})
        return model

    def predict(self, features: pd.DataFrame) -> FloatArray:
        """Raw (uncalibrated) scores in [0, 1].

        Args:
            features: Feature frame containing :attr:`features`.

        Returns:
            Scores aligned with the rows.
        """
        raw = self._booster.predict(features[self.features], num_iteration=self.best_iteration)
        return np.asarray(raw, dtype=np.float64)

    def triage(self, batch: AlertBatch) -> TriageOutput:
        """Score a batch (``Triage`` protocol).

        Args:
            batch: Input batch; only ``features`` is used.

        Returns:
            Raw scores; wrap in ``CalibratedTriage`` for probabilities.
        """
        return TriageOutput(scores=self.predict(batch.features))

    def feature_importance(self) -> dict[str, float]:
        """Total split gain per feature, highest first.

        Returns:
            ``{feature: gain}``.
        """
        gains = self._booster.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.features, gains, strict=True), key=lambda p: -p[1])
        return {name: float(g) for name, g in pairs}

    def save(self, path: Path, *, params: LgbmParams) -> Path:
        """Write the native model file and its manifest (``<path>.manifest.json``).

        Args:
            path: Model file path (``.txt``).
            params: Parameters used for training (recorded for reproducibility).

        Returns:
            The manifest path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path), num_iteration=self.best_iteration)
        manifest = path.with_name(path.name + ".manifest.json")
        manifest.write_text(
            json.dumps(
                {
                    "variant": self.variant,
                    "features": self.features,
                    "best_iteration": self.best_iteration,
                    "params": asdict(params),
                    "sha256": sha256_file(path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest

    @classmethod
    def load(cls, path: Path) -> LightGBMTriage:
        """Load a model only after its checksum matches the manifest.

        Args:
            path: Model file written by :meth:`save`.

        Returns:
            The model.

        Raises:
            DataNotFoundError: If the model or manifest is missing.
            IntegrityError: If the checksum does not match.
        """
        manifest_path = path.with_name(path.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_checksum(path, str(manifest["sha256"]))
        booster = lgb.Booster(model_file=str(path))
        return cls(booster, [str(f) for f in manifest["features"]], str(manifest["variant"]))
