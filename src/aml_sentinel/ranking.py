"""Final ranking: fuse System 1 ``p_triage`` with System 2 graph evidence.

Inputs per alert:

* ``p_triage``: System 1b fused probability. Validation rows use the
  **out-of-fold** score, so this model is never fitted on in-sample fusion outputs.
* System 2 evidence from past-only cases (strict horizon): the finding count and the
  strongest finding per detector, log-transformed.

The stacker is the same interpretable logistic regression + calibrator as System 1b
(:class:`~aml_sentinel.models.fusion.FusionModel`). It is fitted on validation alerts
and evaluated once on test alerts. Its output is the system's final calibrated risk
score; raw model scores are never presented as probabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.evaluation.ranking import bootstrap_ap_difference
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.integrity import sha256_file
from aml_sentinel.models.calibration import FITTERS_WITH_SMOOTHING
from aml_sentinel.models.fusion import FusionModel, stack_inputs
from aml_sentinel.models.system1b import escalation_metrics, score_metrics

logger = logging.getLogger(__name__)

TRIAGE_PREFERENCE: Final = (
    "p__fused[lgbm_all+laya[full]]",
    "p__fused[lgbm_all+laya[head_only]]",
    "p__fused[lgbm_all]",
)
EVIDENCE: Final = (
    "n_findings",
    "max_scatter_gather",
    "max_fan_out",
    "max_gather_scatter",
    "max_circular",
    "max_fan_in",
    "max_structuring",
)


def final_inputs(p_triage: np.ndarray, evidence: pd.DataFrame) -> pd.DataFrame:
    """Named inputs of the final stacker.

    Args:
        p_triage: System 1 probabilities.
        evidence: Evidence columns (:data:`EVIDENCE`) aligned with ``p_triage``.

    Returns:
        Output of :func:`~aml_sentinel.models.fusion.stack_inputs`.
    """
    cols: dict[str, Any] = {"p_triage": p_triage}
    for name in EVIDENCE:
        cols[f"ev_{name}"] = np.log1p(evidence[name].to_numpy(dtype=np.float64))
    return stack_inputs(cols)


def join_inputs(s1b: pd.DataFrame, evidence: pd.DataFrame, triage_col: str) -> pd.DataFrame:
    """Align System 1b scores with System 2 evidence per (split, tx_id).

    Args:
        s1b: System 1b scores with ``split``, ``tx_id``, ``y`` and ``triage_col``.
        evidence: Strict-horizon evidence with ``split`` and ``tx_id``.
        triage_col: Column holding ``p_triage``.

    Returns:
        Joined rows, time order preserved within each split.

    Raises:
        DataValidationError: If any alert lacks evidence.
    """
    left = s1b[["split", "tx_id", "y", triage_col]].reset_index(names="_order")
    merged = left.merge(
        evidence[["split", "tx_id", *EVIDENCE]],
        on=["split", "tx_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[list(EVIDENCE)].isna().any().any():
        raise DataValidationError("some alerts have no System 2 evidence; re-run detect-patterns")
    return merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def run_final_ranking(
    s1b: pd.DataFrame, evidence: pd.DataFrame, *, seed: int = 42
) -> tuple[dict[str, Any], FusionModel, pd.DataFrame]:
    """Fit on validation alerts, evaluate on test alerts.

    Args:
        s1b: System 1b scores (val out-of-fold and test).
        evidence: System 2 strict evidence.
        seed: Bootstrap seed.

    Returns:
        ``(report, model, scores)``. ``scores`` holds test ``tx_id``, ``y``,
        ``p_triage`` and ``p_final``.

    Raises:
        DataValidationError: If no triage column or no validation rows are present.
    """
    triage_col = next((c for c in TRIAGE_PREFERENCE if c in s1b.columns), None)
    if triage_col is None or "split" not in s1b.columns:
        raise DataValidationError("System 1b scores lack a fused column or a split column")
    data = join_inputs(s1b, evidence, triage_col)
    val, test = data[data["split"] == "val"], data[data["split"] == "test"]
    if val.empty or test.empty:
        raise DataValidationError("final ranking needs validation and test alerts")
    x_val = final_inputs(val[triage_col].to_numpy(), val)
    x_test = final_inputs(test[triage_col].to_numpy(), test)
    # The final layer may also pick smoothed isotonic, which never outputs exactly 1.0
    # (ADR-022); the choice is made by validation cross-fitting.
    model, p_val = FusionModel.fit_oof(x_val, val["y"].to_numpy(), fitters=FITTERS_WITH_SMOOTHING)
    p_test = model.predict(x_test)
    y_test, y_val = test["y"].to_numpy(), val["y"].to_numpy()
    p_triage_test = test[triage_col].to_numpy()
    report = {
        "triage_input": triage_col,
        "alerts": {"val": len(val), "test": len(test), "test_positives": int(y_test.sum())},
        "system1_only": {
            **score_metrics(y_test, p_triage_test),
            "escalation": escalation_metrics(
                y_val, val[triage_col].to_numpy(), y_test, p_triage_test
            ),
        },
        "final": {
            **score_metrics(y_test, p_test),
            "escalation": escalation_metrics(y_val, p_val, y_test, p_test),
            "coefficients": model.explain(),
            "calibrator": model.calibrator.method,
            "calibrator_selection_val_ece": model.selection,
        },
        "graph_evidence_gain": bootstrap_ap_difference(y_test, p_test, p_triage_test, seed=seed),
    }
    scores = pd.DataFrame(
        {
            "tx_id": test["tx_id"].to_numpy(),
            "y": y_test,
            "p_triage": p_triage_test,
            "p_final": p_test,
        }
    )
    return report, model, scores


def rank_command(settings: Settings, dataset: DatasetName) -> Path:
    """Run the final ranking and persist model, scores and report.

    Args:
        settings: Settings.
        dataset: Dataset variant.

    Returns:
        Path of the JSON report.

    Raises:
        DataNotFoundError: If System 1b scores or System 2 evidence are missing.
    """
    s1b_path = settings.processed_dir / f"{dataset.value}.system1b_scores.parquet"
    ev_path = settings.processed_dir / f"{dataset.value}.system2_evidence.parquet"
    for path, cmd in ((s1b_path, "triage-text"), (ev_path, "detect-patterns")):
        if not path.is_file():
            raise DataNotFoundError(f"{path.name} missing; run {cmd} first")
    report, model, scores = run_final_ranking(
        pd.read_parquet(s1b_path), pd.read_parquet(ev_path), seed=settings.random_seed
    )
    model_path = settings.models_dir / dataset.value / "final_ranker.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model.to_dict()) + "\n", encoding="utf-8", newline="\n")
    report = {"dataset": dataset.value, **report, "model_sha256": sha256_file(model_path)}
    scores.to_parquet(settings.processed_dir / f"{dataset.value}.final_scores.parquet", index=False)
    out = settings.reports_dir / "final"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{dataset.value}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / f"{dataset.value}.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    logger.info("final ranking written", extra={"dataset": dataset.value})
    return json_path


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the final-ranking report.

    Args:
        report: Output of :func:`rank_command`.

    Returns:
        Markdown text.
    """
    a = report["alerts"]
    lines = [
        f"# Final ranking: {report['dataset']}",
        "",
        f"Validation-fitted stacker over `{report['triage_input']}` plus System 2 evidence; "
        f"{a['test']:,} test alerts ({a['test_positives']} laundering).",
        "",
        "| Score | PR-AUC | P@100 | ECE | Escalate 5%: precision | recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("system1_only", "final"):
        m = report[name]
        esc = m["escalation"]["test"]
        lines.append(
            f"| {name} | {m['ranking']['average_precision']:.3f} "
            f"| {_pct(m['precision_at_k']['100']['precision'])} | {m['ece_quantile']:.4f} "
            f"| {_pct(esc['precision'])} | {_pct(esc['recall'])} |"
        )
    g = report["graph_evidence_gain"]
    lines += [
        "",
        f"Graph-evidence gain (paired bootstrap, PR-AUC): {g['diff']:+.4f} "
        f"[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}].",
        "",
        "Coefficients (standardised): "
        + ", ".join(f"{k}={v:+.2f}" for k, v in report["final"]["coefficients"].items()),
    ]
    return "\n".join(lines) + "\n"
