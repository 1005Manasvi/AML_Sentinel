"""Persist System 1a artifacts and render its report and figures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")  # headless: figures are written to files, never shown
import matplotlib.pyplot as plt

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.integrity import sha256_file
from aml_sentinel.models.lgbm import LgbmParams
from aml_sentinel.models.system1a import (
    load_baseline_config,
    train_system1a,
)

logger = logging.getLogger(__name__)


def _pct(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def _num(v: float | None, digits: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(report: dict[str, Any], dataset: str) -> str:
    """Render the System 1a headline tables.

    Args:
        report: Report produced by :func:`system1a_command`.
        dataset: Dataset name.

    Returns:
        Markdown text.
    """
    base = report["baseline_test"]
    lines = [
        f"# System 1a (LightGBM): {dataset}",
        "",
        "Measured on the test split. The model is fitted on train, early-stopped on the "
        "last train day and calibrated on validation. The baseline is the "
        f"validation-selected rule config (`{report['baseline_name']}`).",
        "",
        "## Ranking and calibration",
        "",
        "| Model | PR-AUC | ROC-AUC | Calibrator | ECE raw | ECE calibrated | Brier calibrated |",
        "|---|---:|---:|---|---:|---:|---:|",
        f"| rules (baseline) | {_num(base['ranking']['average_precision'])} "
        f"| {_num(base['ranking']['roc_auc'])} | n/a | n/a | n/a | n/a |",
    ]
    for name, v in report["variants"].items():
        cal = v["calibration"]
        lines.append(
            f"| lightgbm[{name}] | {_num(v['ranking']['average_precision'])} "
            f"| {_num(v['ranking']['roc_auc'])} | {cal['selected']} "
            f"| {_num(cal['test_raw']['ece_quantile'], 5)} "
            f"| {_num(cal['test_calibrated']['ece_quantile'], 5)} "
            f"| {_num(cal['test_calibrated']['brier'], 6)} |"
        )
    lines += [
        "",
        f"Test base rate: {_pct(base['ranking']['base_rate'], 3)}.",
        "",
        "## Against the rule baseline at matched operating points",
        "",
        "| Model | Comparison | Alerts | Model precision | Model recall | Baseline precision "
        "| Baseline recall | FP reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in report["variants"].items():
        for kind in ("matched_volume", "matched_recall"):
            c = v["vs_baseline"][kind]
            lines.append(
                f"| lightgbm[{name}] | {kind.replace('_', ' ')} | {c['model']['alerts']:,} "
                f"| {_pct(c['model']['precision'])} | {_pct(c['model']['recall'])} "
                f"| {_pct(c['baseline']['precision'])} | {_pct(c['baseline']['recall'])} "
                f"| {_pct(c['fp_reduction'], 1)} |"
            )
    lines += [
        "",
        "## Analyst budgets (precision@k)",
        "",
        "| Model | k | Precision | Recall |",
        "|---|---:|---:|---:|",
    ]
    for name, v in report["variants"].items():
        for k, m in v["precision_at_k"].items():
            lines.append(
                f"| lightgbm[{name}] | {int(k):,} | {_pct(m['precision'])} | {_pct(m['recall'])} |"
            )
    lines += ["", "## Top features by gain", ""]
    for name, v in report["variants"].items():
        top = ", ".join(list(v["feature_importance_gain"])[:8])
        lines.append(f"- **{name}**: {top}")
    return "\n".join(lines) + "\n"


def plot_figures(report: dict[str, Any], out_dir: Path, dataset: str) -> list[Path]:
    """Write the reliability diagram and the PR curve as PNG files.

    Args:
        report: System 1a report.
        out_dir: Output directory.
        dataset: Dataset name for titles and file names.

    Returns:
        Paths of the written figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5.5, 5))
    omitted = 0
    for name, v in report["variants"].items():
        table = v["calibration"]["reliability_test_calibrated"]
        # Log axes cannot show zeros; clipping them would draw fake spikes, so such
        # bins are dropped and counted in the subtitle instead.
        shown = [r for r in table if r["mean_pred"] > 0 and r["frac_pos"] > 0]
        omitted += len(table) - len(shown)
        ax.loglog(
            [r["mean_pred"] for r in shown],
            [r["frac_pos"] for r in shown],
            "o-",
            label=f"lightgbm[{name}] + {v['calibration']['selected']}",
        )
    lo, hi = 1e-6, 1.0
    ax.loglog([lo, hi], [lo, hi], "k--", lw=1, label="perfect calibration")
    ax.set(
        xlabel="Mean predicted probability (test, equal-mass bins)",
        ylabel="Observed laundering rate",
        title=f"Reliability: {dataset}\n({omitted} bins with zero positives not drawn)",
    )
    ax.legend(fontsize=8)
    paths.append(out_dir / f"{dataset}_reliability.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, v in report["variants"].items():
        ax.plot(v["pr_curve"]["recall"], v["pr_curve"]["precision"], label=f"lightgbm[{name}]")
    first = next(iter(report["variants"].values()))
    base = first["vs_baseline"]["matched_volume"]["baseline"]
    ax.plot(
        [base["recall"]],
        [base["precision"]],
        "rX",
        ms=10,
        label=f"rule baseline ({report['baseline_name']})",
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        yscale="log",
        title=f"Precision-recall (test): {dataset}",
    )
    ax.legend(fontsize=8)
    paths.append(out_dir / f"{dataset}_pr_curve.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def system1a_command(
    settings: Settings, dataset: DatasetName, *, params: LgbmParams | None = None
) -> Path:
    """Train, calibrate, evaluate and persist System 1a for one dataset.

    Writes models and calibrators to ``models/<dataset>/``, scores for Phase 4 to
    ``data/processed/<dataset>.system1a_scores.parquet``, and the report and figures
    to ``reports/system1a/``.

    Args:
        settings: Validated settings.
        dataset: Dataset variant.
        params: LightGBM parameters (seeded from settings by default).

    Returns:
        Path of the JSON report.
    """
    params = params or LgbmParams(seed=settings.random_seed)
    frame = load_processed(settings.processed_dir, dataset)
    baseline_name, baseline = load_baseline_config(settings.reports_dir, dataset)
    report, models, scores = train_system1a(
        frame, baseline, train_end=settings.train_end, params=params
    )
    report = {"dataset": dataset.value, "baseline_name": baseline_name, **report}

    model_dir = settings.models_dir / dataset.value
    for variant, (model, calibrator) in models.items():
        model.save(model_dir / f"lightgbm_{variant}.txt", params=params)
        cal_path = model_dir / f"calibrator_{variant}.json"
        cal_path.write_text(json.dumps(calibrator) + "\n", encoding="utf-8", newline="\n")
        report["variants"][variant]["artifacts"] = {
            "model_sha256": sha256_file(model_dir / f"lightgbm_{variant}.txt"),
            "calibrator_sha256": sha256_file(cal_path),
        }
    scores_path = settings.processed_dir / f"{dataset.value}.system1a_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    report["scores_sha256"] = sha256_file(scores_path)

    out_dir = settings.reports_dir / "system1a"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset.value}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{dataset.value}.md").write_text(
        render_markdown(report, dataset.value), encoding="utf-8", newline="\n"
    )
    plot_figures(report, out_dir, dataset.value)
    logger.info("system1a written", extra={"dataset": dataset.value})
    return json_path
