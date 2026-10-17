"""System 1b: text triage (Laya, optional Jev) over the alert queue, plus fusion.

Flow and interpretation (ADR-013):

1. **Alert queue.** The top 1% of each split by LightGBM[all]'s calibrated probability
   (System 1a) forms the alert queue, about 8.6k test alerts on HI-Small.
2. **Text triage.** Laya answers typology, risk and escalation for a fixed seeded
   sample of 4,000 alerts per split. Every text model sees exactly the same alerts.
   The sample keeps the run tractable at about 120 ms per alert on a 4 GB laptop GPU.
3. **Fusion.** A logistic-regression stacker over LightGBM and Laya outputs is fitted
   on validation alerts and evaluated on test alerts.
4. **Escalation.** The top 5% of alerts by ``p_triage`` go to System 2. The threshold
   is chosen on validation, and the test escalation rate is reported as measured.

Every metric here is computed *within the alert queue*. Base rates inside the queue
are far higher than in the full data, so these numbers are not comparable with
Phase 3's full-split metrics.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.patterns import TYPOLOGY
from aml_sentinel.data.schema import Col
from aml_sentinel.evaluation.metrics import binary_metrics
from aml_sentinel.evaluation.ranking import (
    bootstrap_ap_difference,
    precision_at_k,
    ranking_summary,
)
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.models.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from aml_sentinel.models.decisions import TYPOLOGY_LABELS
from aml_sentinel.models.fusion import FusionModel, stack_inputs
from aml_sentinel.models.summarizer import summarize
from aml_sentinel.models.triage import AlertBatch, Triage, TriageOutput
from aml_sentinel.models.typology import laya_label

ALERT_RATE: Final = 0.01
# Floor on queue size so small datasets (the bundled demo sample) still give fusion
# enough alerts to fit; IBM-sized splits are unaffected (1% is already ~8.6k alerts).
MIN_ALERTS: Final = 200
EVAL_SAMPLE: Final = 4_000
ESCALATE_SHARE: Final = 0.05
TOP_K: Final = (100, 250, 500)
TABULAR: Final = {"lgbm_all": "p_all_cal", "lgbm_no_shortcuts": "p_no_shortcuts_cal"}
EVAL_SPLITS: Final = ("val", "test")
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AlertSet:
    """The sampled alerts of one split.

    Attributes:
        rows: Positional indices into the processed frame, in time order.
        y: Laundering labels.
        typology: Laya typology label per alert (``None`` = undocumented positive).
        tabular: LightGBM calibrated probabilities per variant.
        queue_size: Alerts in the full queue before sampling.
    """

    rows: npt.NDArray[np.int64]
    y: npt.NDArray[np.int8]
    typology: tuple[str | None, ...]
    tabular: dict[str, FloatArray]
    queue_size: int


def build_alert_sets(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    rate: float = ALERT_RATE,
    sample: int = EVAL_SAMPLE,
    seed: int = 42,
    min_alerts: int = MIN_ALERTS,
) -> dict[str, AlertSet]:
    """Select each split's alert queue and a seeded evaluation sample of it.

    Args:
        frame: Processed transactions (with ``tx_id`` and ``typology``).
        scores: System 1a scores (``tx_id``, ``split``, ``p_<variant>_cal``).
        rate: Fraction of each split forming the alert queue.
        sample: Alerts sampled per split for text triage.
        seed: Sampling seed.
        min_alerts: Minimum queue size (capped at the split size).

    Returns:
        ``{split: AlertSet}`` for validation and test.

    Raises:
        DataValidationError: If score tx_ids are missing from the frame.
    """
    position = pd.Index(frame[Col.TX_ID]).get_indexer(scores[Col.TX_ID])
    if (position < 0).any():
        raise DataValidationError("system1a scores reference unknown transactions")
    rng = np.random.default_rng(seed)
    y_all = frame[Col.IS_LAUNDERING].to_numpy().astype(np.int8)
    typ_all = frame[TYPOLOGY].astype("string").to_numpy()
    out = {}
    for split in EVAL_SPLITS:
        in_split = (scores["split"].astype("string") == split).to_numpy()
        idx = np.flatnonzero(in_split)
        p = scores[TABULAR["lgbm_all"]].to_numpy()[idx]
        k = max(round(rate * idx.size), min(min_alerts, idx.size), 1)
        queue = idx[np.argsort(-p, kind="stable")[:k]]
        chosen = np.sort(rng.choice(queue, size=min(sample, queue.size), replace=False))
        rows = position[chosen].astype(np.int64)
        order = np.argsort(rows, kind="stable")  # time order, needed by cross-fitting
        rows, chosen = rows[order], chosen[order]
        y = y_all[rows]
        labels = tuple(
            None
            if (lab := laya_label(bool(yi), t if isinstance(t, str) else None)) is None
            else lab.value
            for yi, t in zip(y, typ_all[rows], strict=True)
        )
        out[split] = AlertSet(
            rows=rows,
            y=y,
            typology=labels,
            tabular={
                name: scores[col].to_numpy(dtype=np.float64)[chosen]
                for name, col in TABULAR.items()
            },
            queue_size=int(queue.size),
        )
    return out


def score_metrics(y: npt.ArrayLike, p: npt.ArrayLike) -> dict[str, Any]:
    """Ranking and calibration metrics within the alert set.

    Args:
        y: Labels.
        p: Probabilities.

    Returns:
        Ranking summary, precision@k, quantile ECE and Brier score.
    """
    return {
        "ranking": ranking_summary(y, p),
        "precision_at_k": precision_at_k(y, p, TOP_K),
        "ece_quantile": expected_calibration_error(p, y, n_bins=10),
        "brier": brier_score(p, y),
    }


def typology_metrics(true: Sequence[str | None], pred: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 and confusion matrix on alerts with a known label.

    Args:
        true: True labels (``None`` = unknown, excluded).
        pred: Predicted labels.

    Returns:
        Metrics plus the number of alerts scored.
    """
    pairs = [(t, p) for t, p in zip(true, pred, strict=True) if t is not None]
    if not pairs:
        return {"n": 0, "accuracy": None, "macro_f1": None, "confusion": {}}
    labels = [lab for lab in TYPOLOGY_LABELS if any(t == lab for t, _ in pairs)]
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in pairs if t == lab and p == lab)
        fp = sum(1 for t, p in pairs if t != lab and p == lab)
        fn = sum(1 for t, p in pairs if t == lab and p != lab)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    confusion = {
        t: {p: sum(1 for a, b in pairs if a == t and b == p) for p in TYPOLOGY_LABELS}
        for t in labels
    }
    return {
        "n": len(pairs),
        "accuracy": sum(1 for t, p in pairs if t == p) / len(pairs),
        "macro_f1": float(np.mean(f1s)),
        "labels_present": labels,
        "confusion": confusion,
    }


def escalation_metrics(
    y_val: npt.ArrayLike,
    p_val: npt.ArrayLike,
    y_test: npt.ArrayLike,
    p_test: npt.ArrayLike,
    *,
    share: float = ESCALATE_SHARE,
) -> dict[str, Any]:
    """Escalate alerts above a threshold set on validation; measure it on test.

    Args:
        y_val: Validation labels.
        p_val: Validation scores.
        y_test: Test labels.
        p_test: Test scores.
        share: Target share of alerts escalated (on validation).

    Returns:
        Threshold, measured test escalation share and test metrics.
    """
    threshold = float(np.quantile(np.asarray(p_val, dtype=np.float64), 1 - share))
    escalate = np.asarray(p_test, dtype=np.float64) >= threshold
    return {
        "threshold": threshold,
        "target_share": share,
        "test_share": float(escalate.mean()),
        "test": binary_metrics(np.asarray(y_test), escalate).to_dict(),
    }


def run_text_model(
    model: Triage, frame_features: pd.DataFrame, alerts: AlertSet, *, is_synthetic: bool
) -> tuple[TriageOutput, float]:
    """Summarise alerts, run one text model and time it.

    Args:
        model: Text triage model.
        frame_features: Feature frame of the whole dataset.
        alerts: Alert set to score.
        is_synthetic: Provenance flag passed to the batch (Jev refuses ``False``).

    Returns:
        ``(output, milliseconds_per_alert)``.
    """
    feats = frame_features.iloc[alerts.rows].reset_index(drop=True)
    summaries = tuple(
        summarize({str(k): float(v) for k, v in feats.iloc[i].items()}) for i in range(len(feats))
    )
    batch = AlertBatch(features=feats, summaries=summaries, is_synthetic=is_synthetic)
    start = time.perf_counter()
    out = model.triage(batch)
    elapsed_ms = 1000 * (time.perf_counter() - start) / max(1, len(feats))
    return out, elapsed_ms


def fusion_inputs(tabular: FloatArray, text: TriageOutput | None) -> pd.DataFrame:
    """Named fusion inputs for one alert set.

    Args:
        tabular: LightGBM calibrated probabilities.
        text: Text-model output, or ``None`` for a tabular-only stack.

    Returns:
        Output of :func:`stack_inputs`.
    """
    cols: dict[str, Any] = {"p_tabular": tabular}
    if text is not None:
        extras = text.extras or {}
        cols["p_escalate"] = text.scores
        if "risk_expected" in extras:
            cols["risk_expected"] = extras["risk_expected"]
        if "p_benign" in extras:
            cols["p_benign"] = extras["p_benign"]
    return stack_inputs(cols)


def evaluate_system1b(
    alert_sets: Mapping[str, AlertSet],
    outputs: Mapping[str, Mapping[str, TriageOutput]],
    latency_ms: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, FloatArray]], dict[str, FusionModel]]:
    """Compute every System 1b metric and fit the fusion models.

    Args:
        alert_sets: ``{"val": ..., "test": ...}``.
        outputs: ``{text_model: {split: TriageOutput}}``.
        latency_ms: ``{text_model: ms per alert}``.

    Returns:
        ``(report, test_scores, fusion_models)``. ``test_scores[name]`` holds every
        score evaluated on the test alerts.
    """
    val, test = alert_sets["val"], alert_sets["test"]
    report: dict[str, Any] = {
        "alerts": {
            s: {"queue": a.queue_size, "sampled": int(a.rows.size), "positives": int(a.y.sum())}
            for s, a in alert_sets.items()
        },
        "models": {},
        "fusion": {},
    }
    test_scores: dict[str, dict[str, FloatArray]] = {}
    for name, p in test.tabular.items():
        report["models"][name] = {
            **score_metrics(test.y, p),
            "escalation": escalation_metrics(val.y, val.tabular[name], test.y, p),
        }
        test_scores[name] = {"p": p}
    for name, by_split in outputs.items():
        out_test = by_split["test"]
        report["models"][name] = {
            **score_metrics(test.y, out_test.scores),
            "escalation": escalation_metrics(
                val.y, by_split["val"].scores, test.y, out_test.scores
            ),
            "typology": typology_metrics(test.typology, out_test.typology or ()),
            "latency_ms_per_alert": latency_ms.get(name),
            "mean_risk_positive": _mean_or_none(out_test.risk, test.y == 1),
            "mean_risk_negative": _mean_or_none(out_test.risk, test.y == 0),
            "reliability_test": reliability_table(out_test.scores, test.y, n_bins=10),
        }
        test_scores[name] = {"p": out_test.scores}

    fusions: dict[str, FusionModel] = {}
    combos: list[tuple[str, str, str | None]] = [(t, t, None) for t in TABULAR]
    combos += [(f"{t}+{m}", t, m) for t in TABULAR for m in outputs]
    for combo, tab, text in combos:
        x_val = fusion_inputs(val.tabular[tab], outputs[text]["val"] if text else None)
        x_test = fusion_inputs(test.tabular[tab], outputs[text]["test"] if text else None)
        model = FusionModel.fit(x_val, val.y)
        p_val, p_test = model.predict(x_val), model.predict(x_test)
        report["fusion"][f"fused[{combo}]"] = {
            **score_metrics(test.y, p_test),
            "escalation": escalation_metrics(val.y, p_val, test.y, p_test),
            "coefficients": model.explain(),
            "calibrator": model.calibrator.method,
            "reliability_test": reliability_table(p_test, test.y, n_bins=10),
        }
        fusions[combo] = model
        test_scores[f"fused[{combo}]"] = {"p": p_test}
    report["significance"] = significance(test.y, test_scores, list(outputs))
    return report, test_scores, fusions


def significance(
    y: npt.ArrayLike, test_scores: Mapping[str, Mapping[str, FloatArray]], text_models: list[str]
) -> dict[str, dict[str, float]]:
    """Paired-bootstrap PR-AUC gains: each text-augmented fusion vs. tabular-only fusion.

    Args:
        y: Test-alert labels.
        test_scores: Scores keyed by model name.
        text_models: Text model names.

    Returns:
        ``{"<with> vs <without>": bootstrap_result}``.
    """
    out: dict[str, dict[str, float]] = {}
    for tab in TABULAR:
        base = test_scores[f"fused[{tab}]"]["p"]
        for m in text_models:
            name = f"fused[{tab}+{m}]"
            out[f"{name} vs fused[{tab}]"] = bootstrap_ap_difference(
                y, test_scores[name]["p"], base
            )
    return out


def _mean_or_none(
    values: npt.NDArray[np.int64] | None, mask: npt.NDArray[np.bool_]
) -> float | None:
    if values is None or not mask.any():
        return None
    return float(values[mask].mean())
