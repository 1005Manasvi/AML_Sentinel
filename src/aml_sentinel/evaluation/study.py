"""Phase 8 evaluation studies: ablation, calibration, attempt overlap, reliability bands.

Pure functions over arrays and frames; :mod:`aml_sentinel.evaluation.report` does the
loading, timing and writing.

Every model in the ablation is fitted on **validation** alerts only and scored once
on test alerts, exactly like the production fusion (ADR-013/018). Each step is
compared with the previous one by a paired bootstrap over test alerts. It resamples
rows only, so the intervals are optimistic (alerts from one laundering attempt are
not independent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.evaluation.ranking import bootstrap_ap_difference, ranking_summary, top_k_mask
from aml_sentinel.models.calibration import (
    FITTERS_WITH_SMOOTHING,
    IsotonicCalibrator,
    PlattCalibrator,
    reliability_table,
    select_method,
)
from aml_sentinel.models.fusion import FusionModel, stack_inputs
from aml_sentinel.models.system1a import calibration_metrics
from aml_sentinel.models.system1b import (
    ESCALATE_SHARE,
    escalation_metrics,
    queue_size,
    score_metrics,
)
from aml_sentinel.rules.engine import N_RULES, RuleConfig, apply_rules, trailing_counts

FloatArray = npt.NDArray[np.float64]
Columns = Mapping[str, npt.ArrayLike]
_PROB_EPS: Final = 1e-6


# ---------------------------------------------------------------- inputs
def rule_scores(frame: pd.DataFrame, config: RuleConfig, rows: npt.ArrayLike) -> FloatArray:
    """Rule-baseline score (fraction of enabled rules fired) for selected rows.

    Velocity is counted over the whole time-sorted frame, so each row sees exactly
    the history the baseline saw (current and earlier rows only).

    Args:
        frame: Processed transactions, time-sorted.
        config: Validation-selected rule configuration.
        rows: Positional indices of the rows to score.

    Returns:
        Scores in [0, 1].
    """
    idx = np.asarray(rows, dtype=np.int64)
    velocity = trailing_counts(frame, window=config.velocity_window)
    flags = apply_rules(frame.iloc[idx], config, velocity[idx])
    return flags[N_RULES].to_numpy(dtype=np.float64) / len(config.enabled)


def evidence_columns(evidence: pd.DataFrame, names: Sequence[str]) -> dict[str, FloatArray]:
    """Log-transformed System 2 evidence, as in the final ranker.

    Args:
        evidence: Evidence rows aligned with the alerts.
        names: Evidence columns to use.

    Returns:
        ``{"ev_<name>": log1p(values)}``.
    """
    return {f"ev_{n}": np.log1p(evidence[n].to_numpy(dtype=np.float64)) for n in names}


def as_probability(scores: npt.ArrayLike) -> FloatArray:
    """Clip scores away from 0 and 1 so their logit is finite.

    Args:
        scores: Scores in [0, 1].

    Returns:
        Clipped scores.
    """
    return np.clip(np.asarray(scores, dtype=np.float64), _PROB_EPS, 1 - _PROB_EPS)


# ---------------------------------------------------------------- ablation
@dataclass(frozen=True, slots=True)
class AblationStep:
    """One cumulative ablation row.

    Attributes:
        name: Row label.
        val: Inputs added at this step, for validation alerts.
        test: The same inputs for test alerts.
        note: Caveat shown with the row.
        skipped: Reason the step could not run (its inputs are then empty).
    """

    name: str
    val: Columns
    test: Columns
    note: str = ""
    skipped: str | None = None


def _top_share_counts(y: npt.ArrayLike, p: npt.ArrayLike, share: float) -> dict[str, int]:
    t = np.asarray(y).astype(bool)
    k = max(1, round(share * t.size))
    mask = top_k_mask(p, k)
    return {"k": k, "tp": int((mask & t).sum()), "fp": int((mask & ~t).sum())}


def run_ablation(
    y_val: npt.ArrayLike, y_test: npt.ArrayLike, steps: Sequence[AblationStep], *, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, FloatArray]]:
    """Fit one validation stacker per cumulative step and score it on test.

    Args:
        y_val: Validation-alert labels (time-ordered, matching the inputs).
        y_test: Test-alert labels.
        steps: Steps in order; each adds inputs to all earlier ones.
        seed: Bootstrap seed.

    Returns:
        ``(rows, test_scores)``. Rows hold metrics, the calibrator, coefficients,
        the false positives among the top 5% of test alerts, and the paired-bootstrap
        PR-AUC change versus the previous step that ran.
    """
    yv, yt = np.asarray(y_val), np.asarray(y_test)
    val_cols: dict[str, npt.ArrayLike] = {}
    test_cols: dict[str, npt.ArrayLike] = {}
    rows: list[dict[str, Any]] = []
    scores: dict[str, FloatArray] = {}
    prev: tuple[str, FloatArray] | None = None
    first_fp: int | None = None
    for step in steps:
        if step.skipped:
            rows.append({"name": step.name, "skipped": step.skipped, "note": step.note})
            continue
        val_cols.update(step.val)
        test_cols.update(step.test)
        model, p_val = FusionModel.fit_oof(stack_inputs(val_cols), yv)
        p_test = model.predict(stack_inputs(test_cols))
        top = _top_share_counts(yt, p_test, ESCALATE_SHARE)
        first_fp = top["fp"] if first_fp is None else first_fp
        row: dict[str, Any] = {
            "name": step.name,
            "note": step.note,
            "inputs": list(model.inputs),
            **score_metrics(yt, p_test),
            "escalation": escalation_metrics(yv, p_val, yt, p_test),
            "top_5pct": top,
            "fp_reduction_vs_first": None if not first_fp else 1 - top["fp"] / first_fp,
            "calibrator": model.calibrator.method,
            "coefficients": model.explain(),
            "vs_previous": None
            if prev is None
            else {"previous": prev[0], **bootstrap_ap_difference(yt, p_test, prev[1], seed=seed)},
        }
        rows.append(row)
        scores[step.name] = p_test
        prev = (step.name, p_test)
    return rows, scores


def reference_row(
    name: str, y_test: npt.ArrayLike, p_test: npt.ArrayLike, note: str
) -> dict[str, Any]:
    """Metrics of an already-fitted production score on the same test alerts.

    Args:
        name: Row label.
        y_test: Test-alert labels.
        p_test: The production score.
        note: Where the score comes from.

    Returns:
        A row comparable with :func:`run_ablation` rows.
    """
    p = np.asarray(p_test, dtype=np.float64)
    return {
        "name": name,
        "note": note,
        **score_metrics(y_test, p),
        "top_5pct": _top_share_counts(y_test, p, ESCALATE_SHARE),
    }


# ---------------------------------------------------------------- calibration
def calibration_study(
    raw_val: npt.ArrayLike,
    y_val: npt.ArrayLike,
    raw_test: npt.ArrayLike,
    y_test: npt.ArrayLike,
    *,
    recent_val: npt.ArrayLike,
) -> dict[str, Any]:
    """Compare calibrators fitted on validation, including a recent-window refit.

    Selection evidence is the time-ordered cross-fit ECE on validation; test metrics
    are reported for every candidate, so the selection can be audited but is never
    made on test.

    Args:
        raw_val: Raw validation scores (time-ordered).
        y_val: Validation labels.
        raw_test: Raw test scores.
        y_test: Test labels.
        recent_val: Boolean mask of the most recent validation window (the
            "rolling recalibration" candidate is fitted on it alone).

    Returns:
        ``{"val_crossfit_ece", "selected", "test": {candidate: metrics}}``.
    """
    sv, yv = np.asarray(raw_val, dtype=np.float64), np.asarray(y_val)
    st, yt = np.asarray(raw_test, dtype=np.float64), np.asarray(y_test)
    recent = np.asarray(recent_val, dtype=bool)
    selected, crossfit = select_method(sv, yv, fitters=FITTERS_WITH_SMOOTHING)
    fitted: dict[str, PlattCalibrator | IsotonicCalibrator] = {
        name: fit(sv, yv) for name, fit in FITTERS_WITH_SMOOTHING.items()
    }
    if recent.any() and 0 < int(yv[recent].sum()) < int(recent.sum()):
        fitted["isotonic_recent_window"] = IsotonicCalibrator.fit(sv[recent], yv[recent])
        fitted["platt_recent_window"] = PlattCalibrator.fit(sv[recent], yv[recent])
    test: dict[str, Any] = {"raw (uncalibrated)": _cal_summary(st, yt)}
    for name, cal in fitted.items():
        test[name] = _cal_summary(cal.transform(st), yt)
    return {
        "val_crossfit_ece": crossfit,
        "selected_on_validation": selected,
        "recent_window_rows": int(recent.sum()),
        "test": test,
    }


def _cal_summary(p: FloatArray, y: npt.ArrayLike) -> dict[str, Any]:
    return {
        **calibration_metrics(p, y),
        "max_pred": float(np.max(p)),
        "share_at_one": float(np.mean(p >= 1.0)),
        "reliability": reliability_table(p, y, n_bins=50),
    }


def recent_mask(
    timestamps: pd.Series, val_end: datetime, window: timedelta
) -> npt.NDArray[np.bool_]:
    """Rows inside ``[val_end - window, val_end)``.

    Args:
        timestamps: Validation timestamps.
        val_end: Exclusive end of validation.
        window: Window length.

    Returns:
        Boolean mask.
    """
    ts = pd.to_datetime(timestamps)
    return ((ts >= val_end - window) & (ts < val_end)).to_numpy()


def reliability_bands(p: npt.ArrayLike, y: npt.ArrayLike, n_bins: int = 10) -> list[dict[str, Any]]:
    """Observed laundering rate per score band (equal-mass bins), with band edges.

    Used by the API to show how often alerts in a score band were laundering on the
    held-out test period, instead of presenting a score such as 1.000 at face value.

    Args:
        p: Scores.
        y: Labels.
        n_bins: Number of bands.

    Returns:
        Bands ordered by score, each with ``lo``, ``hi``, ``count``, ``mean_pred``
        and ``frac_pos``.
    """
    s = np.asarray(p, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    ids = np.empty(s.size, dtype=np.int64)
    ids[order] = (np.arange(s.size) * n_bins) // max(s.size, 1)
    out = []
    for b in range(n_bins):
        m = ids == b
        if m.any():
            out.append(
                {
                    "lo": float(s[m].min()),
                    "hi": float(s[m].max()),
                    "count": int(m.sum()),
                    "mean_pred": float(s[m].mean()),
                    "frac_pos": float(t[m].mean()),
                }
            )
    return out


# ---------------------------------------------------------------- attempts
def attempt_overlap(
    frame: pd.DataFrame, test_tx: npt.ArrayLike, test_scores: npt.ArrayLike
) -> dict[str, Any]:
    """How many test positives continue laundering seen before the test period.

    ADR-003: attempts can cross split boundaries, so a model may partly recognise
    *known* actors. Positives are grouped by whether their attempt, or one of their
    accounts, already appeared in training/validation laundering, and the model's
    recall inside the alert queue (top 1%) and PR-AUC are reported per group
    (each group against all test negatives).

    Args:
        frame: Processed transactions with ``attempt_id``, ``split`` and labels.
        test_tx: Test transaction ids that were scored.
        test_scores: Model scores for them.

    Returns:
        Group sizes and per-group metrics.
    """
    split = frame["split"].astype("string")
    y_all = frame[Col.IS_LAUNDERING].to_numpy() == 1
    earlier = ((split == "train") | (split == "val")).to_numpy()
    seen_attempts = set(frame.loc[earlier & y_all, "attempt_id"].dropna().astype(int))
    train_bad = earlier & y_all
    seen_accounts = set(frame.loc[train_bad, Col.FROM_ID].astype("string")) | set(
        frame.loc[train_bad, Col.TO_ID].astype("string")
    )

    pos = pd.Index(frame[Col.TX_ID]).get_indexer(np.asarray(test_tx))
    rows = frame.iloc[pos]
    y = rows[Col.IS_LAUNDERING].to_numpy() == 1
    p = np.asarray(test_scores, dtype=np.float64)
    attempt = rows["attempt_id"]
    documented = attempt.notna().to_numpy()
    continuing = documented & attempt.fillna(-1).astype(int).isin(seen_attempts).to_numpy()
    known_account = (
        rows[Col.FROM_ID].astype("string").isin(seen_accounts)
        | rows[Col.TO_ID].astype("string").isin(seen_accounts)
    ).to_numpy()
    groups = {
        "attempt_seen_before_test": y & continuing,
        "attempt_new_in_test": y & documented & ~continuing,
        "undocumented_positive": y & ~documented,
        "account_seen_laundering_before": y & known_account,
        "account_new": y & ~known_account,
    }
    queue = top_k_mask(p, queue_size(p.size))
    out: dict[str, Any] = {"test_positives": int(y.sum()), "queue_size": int(queue.sum())}
    for name, g in groups.items():
        keep = g | ~y
        out[name] = {
            "positives": int(g.sum()),
            "share_of_positives": float(g.sum() / max(1, y.sum())),
            "recall_in_queue": None if not g.any() else float((queue & g).sum() / g.sum()),
            "pr_auc_vs_all_negatives": None
            if not g.any()
            else ranking_summary(g[keep], p[keep])["average_precision"],
        }
    return out
