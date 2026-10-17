"""System 1a: train LightGBM, calibrate it and evaluate against the rule baseline.

Data use per split (no split is used for two jobs):

* ``train`` minus its last day: fit trees;
* last training day: early stopping;
* ``val``: choose and fit the calibrator (later also fusion);
* ``test``: report only.

Two feature variants are trained (ADR-010): ``all`` features, and ``no_shortcuts``,
which drops the simulator artifacts (payment format, cross-currency flag).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.data.splits import SplitName
from aml_sentinel.evaluation.ranking import (
    compare_at_matched_recall,
    compare_at_matched_volume,
    operating_points,
    pr_curve_points,
    precision_at_k,
    ranking_summary,
)
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.features.builder import build_features, feature_columns
from aml_sentinel.models.calibration import (
    FITTERS,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_table,
    select_method,
)
from aml_sentinel.models.lgbm import LgbmParams, LightGBMTriage
from aml_sentinel.models.rule_triage import RuleTriage
from aml_sentinel.models.triage import AlertBatch
from aml_sentinel.rules.engine import RuleConfig

logger = logging.getLogger(__name__)

VARIANTS: Final = {"all": True, "no_shortcuts": False}
TOP_K: Final = (100, 1_000, 10_000)
ALERT_RATES: Final = (0.01, 0.02, 0.05)
SPLIT_COL: Final = "split"


def split_masks(frame: pd.DataFrame, train_end: datetime) -> dict[str, npt.NDArray[np.bool_]]:
    """Boolean row masks for every job a split is used for.

    Args:
        frame: Processed transactions with a ``split`` column.
        train_end: Exclusive end of the training period.

    Returns:
        Masks ``fit``, ``stop``, ``val`` and ``test``.

    Raises:
        DataValidationError: If any of those masks is empty.
    """
    split = frame[SPLIT_COL].astype("string")
    is_train = (split == SplitName.TRAIN.value).to_numpy()
    last_day = (frame[Col.TIMESTAMP] >= train_end - timedelta(days=1)).to_numpy()
    masks = {
        "fit": is_train & ~last_day,
        "stop": is_train & last_day,
        "val": (split == SplitName.VAL.value).to_numpy(),
        "test": (split == SplitName.TEST.value).to_numpy(),
    }
    empty = [name for name, m in masks.items() if not m.any()]
    if empty:
        raise DataValidationError(f"empty data slices: {empty}")
    return masks


def load_baseline_config(reports_dir: Path, dataset: DatasetName) -> tuple[str, RuleConfig]:
    """Read the validation-selected rule baseline written by ``aml-sentinel baseline``.

    Args:
        reports_dir: Reports root.
        dataset: Dataset variant.

    Returns:
        ``(variant_name, config)``.

    Raises:
        DataNotFoundError: If the baseline report does not exist yet.
    """
    path = reports_dir / "baseline" / f"{dataset.value}.json"
    if not path.is_file():
        raise DataNotFoundError(f"{path.name} not found; run `aml-sentinel baseline` first")
    report = json.loads(path.read_text(encoding="utf-8"))
    name = str(report["selected"])
    return name, RuleConfig.from_dict(report["variants"][name]["config"])


def calibration_metrics(probs: npt.ArrayLike, y: npt.ArrayLike) -> dict[str, float]:
    """Calibration summary for one set of probabilities.

    Args:
        probs: Probabilities.
        y: Labels.

    Returns:
        Quantile and uniform ECE, Brier score, log loss and mean prediction.
    """
    p = np.asarray(probs, dtype=np.float64)
    return {
        "ece_quantile": expected_calibration_error(p, y, strategy="quantile"),
        "ece_uniform": expected_calibration_error(p, y, strategy="uniform", n_bins=10),
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "mean_pred": float(p.mean()),
        "observed_rate": float(np.mean(np.asarray(y, dtype=np.float64))),
    }


def run_variant(
    features: pd.DataFrame,
    y: npt.NDArray[np.int8],
    masks: dict[str, npt.NDArray[np.bool_]],
    baseline_alerts: npt.NDArray[np.bool_],
    *,
    variant: str,
    params: LgbmParams,
) -> tuple[LightGBMTriage, dict[str, Any], dict[str, Any], dict[str, npt.NDArray[np.float64]]]:
    """Train, calibrate and evaluate one feature variant.

    Args:
        features: Full feature frame.
        y: Labels for all rows.
        masks: Output of :func:`split_masks`.
        baseline_alerts: Rule-baseline alert mask on the **test** rows.
        variant: Key of :data:`VARIANTS`.
        params: LightGBM parameters.

    Returns:
        ``(model, calibrator_dict, report, scores)``. ``scores`` holds raw and
        calibrated val/test scores for persistence.
    """
    cols = feature_columns(include_shortcuts=VARIANTS[variant])
    model = LightGBMTriage.fit(
        features.loc[masks["fit"]],
        y[masks["fit"]],
        features.loc[masks["stop"]],
        y[masks["stop"]],
        features=cols,
        variant=variant,
        params=params,
    )
    raw_val = model.predict(features.loc[masks["val"]])
    raw_test = model.predict(features.loc[masks["test"]])
    y_val, y_test = y[masks["val"]], y[masks["test"]]

    method, cv_ece = select_method(raw_val, y_val)
    calibrator = FITTERS[method](raw_val, y_val)
    cal_test = calibrator.transform(raw_test)
    alternatives = {
        name: calibration_metrics(fit(raw_val, y_val).transform(raw_test), y_test)
        for name, fit in FITTERS.items()
    }

    report = {
        "features": cols,
        "best_iteration": model.best_iteration,
        "ranking": ranking_summary(y_test, raw_test),
        "precision_at_k": precision_at_k(y_test, raw_test, TOP_K),
        "operating_points": operating_points(y_test, raw_test, ALERT_RATES),
        "vs_baseline": {
            "matched_volume": compare_at_matched_volume(y_test, raw_test, baseline_alerts),
            "matched_recall": compare_at_matched_recall(y_test, raw_test, baseline_alerts),
        },
        "calibration": {
            "selected": method,
            "val_crossfit_ece": cv_ece,
            "test_raw": calibration_metrics(raw_test, y_test),
            "test_calibrated": calibration_metrics(cal_test, y_test),
            "test_by_method": alternatives,
            "reliability_test_calibrated": reliability_table(cal_test, y_test),
        },
        "pr_curve": pr_curve_points(y_test, raw_test),
        "feature_importance_gain": dict(list(model.feature_importance().items())[:15]),
    }
    scores = {
        "val_raw": raw_val,
        "val_cal": calibrator.transform(raw_val),
        "test_raw": raw_test,
        "test_cal": cal_test,
    }
    return model, calibrator.to_dict(), report, scores


def train_system1a(
    frame: pd.DataFrame,
    baseline: RuleConfig,
    *,
    train_end: datetime,
    params: LgbmParams,
) -> tuple[dict[str, Any], dict[str, tuple[LightGBMTriage, dict[str, Any]]], pd.DataFrame]:
    """Run System 1a for every feature variant.

    Args:
        frame: Processed transactions (sorted, with ``split``).
        baseline: Rule config to compare against.
        train_end: Exclusive end of the training period.
        params: LightGBM parameters.

    Returns:
        ``(report, {variant: (model, calibrator_dict)}, scores_frame)``. The scores
        frame holds val/test ``tx_id``, ``split`` and per-variant raw and calibrated
        scores; Phase 4 fusion reads it.
    """
    masks = split_masks(frame, train_end)
    features = build_features(frame)
    y = frame[Col.IS_LAUNDERING].to_numpy().astype(np.int8)

    test_rows = frame.loc[masks["test"]]
    test_batch = AlertBatch(features=features.loc[masks["test"]], transactions=test_rows)
    baseline_scores = RuleTriage(baseline).triage(test_batch).scores
    baseline_alerts = baseline_scores > 0

    report: dict[str, Any] = {
        "rows": {k: int(m.sum()) for k, m in masks.items()},
        "positives": {k: int(y[m].sum()) for k, m in masks.items()},
        "baseline_test": {
            "ranking": ranking_summary(y[masks["test"]], baseline_scores),
            "alerts": int(baseline_alerts.sum()),
        },
        "variants": {},
    }
    models: dict[str, tuple[LightGBMTriage, dict[str, Any]]] = {}
    eval_mask = masks["val"] | masks["test"]
    scores = frame.loc[eval_mask, [Col.TX_ID.value, SPLIT_COL]].reset_index(drop=True)
    for variant in VARIANTS:
        model, calibrator, variant_report, s = run_variant(
            features, y, masks, baseline_alerts, variant=variant, params=params
        )
        report["variants"][variant] = variant_report
        models[variant] = (model, calibrator)
        is_val = (scores[SPLIT_COL].astype("string") == SplitName.VAL.value).to_numpy()
        for kind in ("raw", "cal"):
            col = np.empty(len(scores), dtype=np.float64)
            col[is_val], col[~is_val] = s[f"val_{kind}"], s[f"test_{kind}"]
            scores[f"p_{variant}_{kind}"] = col
    return report, models, scores
