"""``evaluate``: the Phase 8 evaluation report (``reports/evaluation/<dataset>.*``).

Sections:

1. **Full test split, rules vs LightGBM.** Read from the Phase 2/3 reports (not
   recomputed): PR-AUC, precision@k, false-positive reduction at matched recall and
   matched alert volume, and a PR-curve figure.
2. **Ablation on the alert queue.** Cumulative steps rules -> +LightGBM -> +Laya ->
   +Jev (benchmark) -> +graph evidence -> +GNN, each a validation-fitted stacker scored
   on the same test alerts, plus the two production scores as reference rows. Needs
   System 1b scores (``triage-text``).
3. **Calibration.** Candidates (Platt, isotonic, smoothed isotonic, recent-window
   refits) on System 1a's full split, selected by validation cross-fit only.
4. **Attempt overlap** (ADR-003): positives continuing attempts or accounts seen
   before the test period, and the model's recall on each group.
5. **Per-stage latency** on this machine, with each number's provenance.
6. **Reliability bands** of ``p_triage`` on test alerts, used by the API.

Everything is seeded and regenerated with ``aml-sentinel evaluate --dataset <name>``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl

mpl.use("Agg")  # headless: figures are written to files only

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.evaluation.study import (
    AblationStep,
    as_probability,
    attempt_overlap,
    calibration_study,
    evidence_columns,
    recent_mask,
    reference_row,
    reliability_bands,
    rule_scores,
    run_ablation,
)
from aml_sentinel.exceptions import DataNotFoundError
from aml_sentinel.features.builder import build_features
from aml_sentinel.models.lgbm import LightGBMTriage
from aml_sentinel.models.system1a import load_baseline_config
from aml_sentinel.ranking import EVIDENCE, TRIAGE_PREFERENCE, join_inputs
from aml_sentinel.rules.engine import apply_rules, trailing_counts
from aml_sentinel.sar.drafting import draft_sar
from aml_sentinel.sar.facts import build_case_facts
from aml_sentinel.system2.cases import CaseIndex
from aml_sentinel.system2.detectors import run_detectors

logger = logging.getLogger(__name__)

# Best Laya variant first; any other verified variant (e.g. a test double) after.
LAYA_PREFERENCE: Final = ("p__laya[full]", "p__laya[head_only]")
LGBM_COLUMN: Final = "p__lgbm_all"
# Reference palette (dataviz skill), categorical slots in fixed order; validated.
SERIES: Final = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
STYLES: Final = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


# ---------------------------------------------------------------- section 1
def full_split_section(settings: Settings, dataset: DatasetName) -> dict[str, Any]:
    """Rules vs LightGBM on every test transaction, from the Phase 2/3 reports.

    Args:
        settings: Settings.
        dataset: Dataset.

    Returns:
        The section.

    Raises:
        DataNotFoundError: If the System 1a report is missing.
    """
    path = settings.reports_dir / "system1a" / f"{dataset.value}.json"
    if not path.is_file():
        raise DataNotFoundError(f"{path.name} missing; run train-tabular first")
    s1a = json.loads(path.read_text(encoding="utf-8"))
    base_path = settings.reports_dir / "baseline" / f"{dataset.value}.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    chosen = base["variants"][base["selected"]]["test"]
    out: dict[str, Any] = {
        "source": "reports/system1a and reports/baseline (Phases 2-3)",
        "rows": s1a["rows"]["test"],
        "positives": s1a["positives"]["test"],
        "baseline": {
            "name": base["selected"],
            "ranking": s1a["baseline_test"]["ranking"],
            "alerts": chosen["overall"]["alerts"],
            "precision": chosen["overall"]["precision"],
            "recall": chosen["overall"]["recall"],
            "false_positive_rate": chosen["overall"]["false_positive_rate"],
            "alerts_per_day": chosen["alerts_per_day"],
        },
        "models": {},
    }
    for variant, v in s1a["variants"].items():
        mr, mv = v["vs_baseline"]["matched_recall"], v["vs_baseline"]["matched_volume"]
        out["models"][f"lgbm_{variant}"] = {
            "ranking": v["ranking"],
            "precision_at_k": {k: v["precision_at_k"][k]["precision"] for k in v["precision_at_k"]},
            "matched_recall": {
                "recall": mr["baseline"]["recall"],
                "model_alerts": mr["model"]["alerts"],
                "baseline_alerts": mr["baseline"]["alerts"],
                "model_fp": mr["model"]["fp"],
                "baseline_fp": mr["baseline"]["fp"],
                "fp_reduction": mr["fp_reduction"],
            },
            "matched_volume": {
                "alerts": mv["model"]["alerts"],
                "model_recall": mv["model"]["recall"],
                "baseline_recall": mv["baseline"]["recall"],
                "fp_reduction": mv["fp_reduction"],
            },
            "pr_curve": v["pr_curve"],
        }
    return out


# ---------------------------------------------------------------- section 2
def _gnn_scores(
    settings: Settings, dataset: DatasetName, frame: pd.DataFrame, alerts: pd.DataFrame
) -> tuple[dict[str, np.ndarray] | None, str | None, dict[str, float]]:
    report_path = settings.reports_dir / "system2" / f"{dataset.value}_gnn.json"
    weights = settings.models_dir / dataset.value / "gnn_sage.safetensors"
    if not report_path.is_file() or not weights.is_file():
        return None, "no GNN artifacts (run gnn-experiment)", {}
    try:
        from aml_sentinel.system2.gnn import load_trained_gnn
        from aml_sentinel.system2.gnn_data import build_split_graphs
        from aml_sentinel.system2.gnn_report import alert_pair_scores
    except ImportError:  # pragma: no cover - depends on the optional GNN stack
        return None, "GNN stack not installed (make gnn-setup)", {}
    sha = json.loads(report_path.read_text(encoding="utf-8"))["artifacts"]["sha256"]
    graphs, t_graph = _timed(lambda: build_split_graphs(frame))
    model = load_trained_gnn(weights, sha, graphs["train"])
    out: dict[str, np.ndarray] = {}
    t_pred = 0.0
    found_share: dict[str, float] = {}
    for split in ("val", "test"):
        start = time.perf_counter()
        pairs = model.predict(graphs[split])
        t_pred += time.perf_counter() - start
        ids = alerts.loc[alerts["split"] == split, "tx_id"].tolist()
        scores, found = alert_pair_scores(frame, graphs[split], pairs, ids)
        out[split] = scores
        found_share[split] = float(found.mean())
    timing = {"graph_build_s": t_graph, "predict_s": t_pred}
    return out, None, {**timing, **{f"pair_found_{k}": v for k, v in found_share.items()}}


def ablation_section(
    settings: Settings, dataset: DatasetName, frame: pd.DataFrame
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Cumulative ablation over the System 1b alert samples.

    Args:
        settings: Settings.
        dataset: Dataset.
        frame: Processed transactions.

    Returns:
        ``(section or None if System 1b scores are missing, extras for plots/latency)``.
    """
    s1b_path = settings.processed_dir / f"{dataset.value}.system1b_scores.parquet"
    ev_path = settings.processed_dir / f"{dataset.value}.system2_evidence.parquet"
    if not s1b_path.is_file() or not ev_path.is_file():
        return None, {}
    s1b = pd.read_parquet(s1b_path)
    triage_col = next((c for c in TRIAGE_PREFERENCE if c in s1b.columns), None)
    if triage_col is None:
        return None, {}
    data = join_inputs(s1b, pd.read_parquet(ev_path), triage_col)
    laya_col = next((c for c in LAYA_PREFERENCE if c in s1b.columns), None) or next(
        (c for c in s1b.columns if c.startswith("p__laya[") and "zero_shot" not in c), None
    )
    extra_cols = [c for c in (LGBM_COLUMN, laya_col) if c is not None and c in s1b.columns]
    data = data.merge(s1b[["split", "tx_id", *extra_cols]], on=["split", "tx_id"], how="left")
    pos = pd.Index(frame[Col.TX_ID]).get_indexer(data["tx_id"])
    _, rules = load_baseline_config(settings.reports_dir, dataset)
    data["rule_score"] = rule_scores(frame, rules, pos)
    gnn, gnn_skip, gnn_timing = _gnn_scores(settings, dataset, frame, data)
    if gnn is not None:
        data["p_gnn"] = np.nan
        for split, split_scores in gnn.items():
            data.loc[data["split"] == split, "p_gnn"] = split_scores

    val, test = data[data["split"] == "val"], data[data["split"] == "test"]

    def cols(names: dict[str, str], *, prob: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        conv = as_probability if prob else (lambda a: np.asarray(a, dtype=np.float64))
        return (
            {k: conv(val[c].to_numpy()) for k, c in names.items()},
            {k: conv(test[c].to_numpy()) for k, c in names.items()},
        )

    rv, rt = cols({"rule_score": "rule_score"}, prob=False)
    lv, lt = cols({"p_lgbm": LGBM_COLUMN}, prob=True)
    steps = [
        AblationStep("rules only", rv, rt, "validation-selected rule baseline (ADR-009)"),
        AblationStep("+ LightGBM", lv, lt, "calibrated lgbm_all"),
    ]
    if laya_col is not None:
        av, at = cols({"p_laya": laya_col}, prob=True)
        variant = laya_col.removeprefix("p__")
        steps.append(AblationStep("+ Laya", av, at, f"{variant}, escalate probability"))
    else:
        steps.append(AblationStep("+ Laya", {}, {}, skipped="no Laya scores"))
    jev = "not run: JEV_ENABLED=false or no key (benchmark-only, synthetic data only)"
    steps.append(AblationStep("+ Jev (benchmark)", {}, {}, skipped=jev))
    steps.append(
        AblationStep(
            "+ graph evidence",
            evidence_columns(val, EVIDENCE),
            evidence_columns(test, EVIDENCE),
            "past-only System 2 detector evidence (strict horizon)",
        )
    )
    if gnn is not None:
        gv, gt = cols({"p_gnn": "p_gnn"}, prob=True)
        steps.append(
            AblationStep(
                "+ GNN (full stack)",
                gv,
                gt,
                "LOOKAHEAD: end-of-period GraphSAGE sees the whole split, including "
                "activity after each alert; not deployable in real time (ADR-016)",
            )
        )
    else:
        steps.append(AblationStep("+ GNN (full stack)", {}, {}, skipped=gnn_skip))

    y_val, y_test = val["y"].to_numpy(), test["y"].to_numpy()
    rows, scores = run_ablation(y_val, y_test, steps, seed=settings.random_seed)
    references = [
        reference_row(
            "production: System 1 p_triage (queue ranking)",
            y_test,
            test[triage_col].to_numpy(),
            f"{triage_col.removeprefix('p__')}, ADR-020",
        )
    ]
    final_path = settings.processed_dir / f"{dataset.value}.final_scores.parquet"
    if final_path.is_file():
        final = pd.read_parquet(final_path, columns=["tx_id", "p_final"])
        p_final = test[["tx_id"]].merge(final, on="tx_id", how="left")["p_final"]
        if p_final.notna().all():
            references.append(
                reference_row(
                    "production: final score (System 1 + System 2)",
                    y_test,
                    p_final.to_numpy(),
                    "ADR-018",
                )
            )
    section = {
        "alerts": {
            "val": len(val),
            "test": len(test),
            "test_positives": int(y_test.sum()),
            "test_base_rate": float(y_test.mean()),
        },
        "steps": rows,
        "references": references,
        "p_triage_reliability_test": reliability_bands(test[triage_col].to_numpy(), y_test),
        "triage_column": triage_col,
    }
    extras = {"scores": scores, "y_test": y_test, "gnn_timing": gnn_timing, "test": test}
    return section, extras


# ---------------------------------------------------------------- sections 3-4
def calibration_section(
    settings: Settings, dataset: DatasetName, frame: pd.DataFrame
) -> dict[str, Any]:
    """System 1a calibrator candidates on the full split.

    Args:
        settings: Settings.
        dataset: Dataset.
        frame: Processed transactions.

    Returns:
        The section, plus the final ranker's recorded selection when available.
    """
    scores = pd.read_parquet(
        settings.processed_dir / f"{dataset.value}.system1a_scores.parquet",
        columns=["tx_id", "split", "p_all_raw"],
    )
    labelled = scores.merge(
        frame[[Col.TX_ID.value, Col.TIMESTAMP.value, Col.IS_LAUNDERING.value]],
        on=Col.TX_ID.value,
        how="left",
    )
    split = labelled["split"].astype("string")
    val, test = labelled[split == "val"], labelled[split == "test"]
    study = calibration_study(
        val["p_all_raw"].to_numpy(),
        val[Col.IS_LAUNDERING].to_numpy(),
        test["p_all_raw"].to_numpy(),
        test[Col.IS_LAUNDERING].to_numpy(),
        recent_val=recent_mask(val[Col.TIMESTAMP], settings.val_end, timedelta(days=1)),
    )
    out: dict[str, Any] = {"system1a_lgbm_all": study}
    final = settings.reports_dir / "final" / f"{dataset.value}.json"
    if final.is_file():
        rep = json.loads(final.read_text(encoding="utf-8"))["final"]
        out["final_ranker"] = {
            "selected": rep.get("calibrator"),
            "val_crossfit_ece": rep.get("calibrator_selection_val_ece"),
        }
    return out


# ---------------------------------------------------------------- section 5
def latency_section(
    settings: Settings,
    dataset: DatasetName,
    frame: pd.DataFrame,
    ablation_extras: dict[str, Any],
) -> dict[str, Any]:
    """Per-stage latency on this machine, with provenance for reused measurements.

    Args:
        settings: Settings.
        dataset: Dataset.
        frame: Processed transactions.
        ablation_extras: Extras from :func:`ablation_section` (test alerts, GNN timing).

    Returns:
        The section.
    """
    n = len(frame)
    stages: dict[str, Any] = {}
    feats, t_feat = _timed(lambda: build_features(frame))
    stages["features (all past-only features, batch)"] = {
        "us_per_tx": 1e6 * t_feat / n,
        "rows": n,
        "measured": "now",
    }
    _, rules = load_baseline_config(settings.reports_dir, dataset)
    test_mask = (frame["split"].astype("string") == "test").to_numpy()

    def rules_run() -> Any:
        velocity = trailing_counts(frame, window=rules.velocity_window)
        return apply_rules(frame.loc[test_mask], rules, velocity[test_mask])

    _, t_rules = _timed(rules_run)
    stages["rules (velocity over history + 3 rules on test)"] = {
        "us_per_tx": 1e6 * t_rules / int(test_mask.sum()),
        "measured": "now",
    }
    model = LightGBMTriage.load(settings.models_dir / dataset.value / "lightgbm_all.txt")
    test_feats = feats.loc[test_mask]
    _, t_lgbm = _timed(lambda: model.predict(test_feats))
    stages["LightGBM inference (batch)"] = {
        "us_per_tx": 1e6 * t_lgbm / len(test_feats),
        "measured": "now",
    }
    s1b_report = settings.reports_dir / "system1b" / f"{dataset.value}.json"
    if s1b_report.is_file():
        models = json.loads(s1b_report.read_text(encoding="utf-8"))["models"]
        for name in ("laya[zero_shot]", "laya[head_only]"):
            if name in models and models[name].get("latency_ms_per_alert") is not None:
                stages[f"Laya {name} (per alert, GPU)"] = {
                    "ms_per_alert": models[name]["latency_ms_per_alert"],
                    "measured": "Phase 4 run (reports/system1b), RTX 3050 Laptop GPU",
                }
    ev_path = settings.processed_dir / f"{dataset.value}.system2_evidence.parquet"
    if ev_path.is_file():
        cost = pd.read_parquet(ev_path, columns=["ms", "split"])
        ms = cost.loc[cost["split"] == "test", "ms"].to_numpy()
        stages["System 2 case build + detectors (per alert)"] = {
            "ms_median": float(np.median(ms)),
            "ms_p95": float(np.percentile(ms, 95)),
            "alerts": int(ms.size),
            "measured": "Phase 5 run (system2_evidence.parquet)",
        }
    test_alerts = ablation_extras.get("test")
    if test_alerts is not None and len(test_alerts):
        index = CaseIndex(frame)
        tx_ids = test_alerts["tx_id"].to_numpy()[:20]
        timings = []
        for tx in tx_ids:
            case = index.build(int(tx))
            facts = build_case_facts(case, run_detectors(case), risk_score=0.5)
            start = time.perf_counter()
            draft_sar(facts)
            timings.append(1000 * (time.perf_counter() - start))
        stages["SAR draft (local template + grounding, per case)"] = {
            "ms_median": float(np.median(timings)),
            "cases": len(timings),
            "measured": "now",
        }
    gnn_timing = ablation_extras.get("gnn_timing") or {}
    if gnn_timing:
        stages["GNN (end-of-period batch: build graphs + score all pairs)"] = {
            "seconds": gnn_timing["graph_build_s"] + gnn_timing["predict_s"],
            "measured": "now",
        }
    if n <= settings.api_score_max_history_rows:
        from aml_sentinel.api.scoring import build_scorer
        from aml_sentinel.data.schema import Transaction

        scorer, _ = build_scorer(settings, dataset, frame)
        if scorer is not None:
            row = frame.iloc[-1]
            tx = Transaction.model_validate(
                {
                    "timestamp": pd.Timestamp(row[Col.TIMESTAMP]).to_pydatetime(),
                    **{
                        c.value: str(row[c])
                        for c in (
                            Col.FROM_BANK,
                            Col.FROM_ACCOUNT,
                            Col.TO_BANK,
                            Col.TO_ACCOUNT,
                            Col.RECEIVING_CURRENCY,
                            Col.PAYMENT_CURRENCY,
                            Col.PAYMENT_FORMAT,
                        )
                    },
                    "amount_received": float(row[Col.AMOUNT_RECEIVED]),
                    "amount_paid": float(row[Col.AMOUNT_PAID]),
                }
            )
            runs = [scorer.score(tx).latency_ms for _ in range(10)]
            stages["API POST /score (one new transaction, past-only features)"] = {
                "ms_median": float(np.median(runs)),
                "history_rows": n,
                "measured": "now",
            }
    return {"machine": _machine(), "stages": stages}


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - optional stack
        pass
    return info


# ---------------------------------------------------------------- command
def evaluate_command(settings: Settings, dataset: DatasetName) -> Path:
    """Write ``reports/evaluation/<dataset>.json``, ``.md`` and figures.

    Args:
        settings: Settings.
        dataset: Dataset.

    Returns:
        The JSON report path.
    """
    frame = load_processed(settings.processed_dir, dataset)
    ablation, extras = ablation_section(settings, dataset, frame)
    s1a_scores = pd.read_parquet(
        settings.processed_dir / f"{dataset.value}.system1a_scores.parquet",
        columns=["tx_id", "split", "p_all_raw"],
    )
    test_s1a = s1a_scores.loc[s1a_scores["split"].astype("string") == "test"]
    report: dict[str, Any] = {
        "dataset": dataset.value,
        "seed": settings.random_seed,
        "full_split": full_split_section(settings, dataset),
        "ablation": ablation,
        "calibration": calibration_section(settings, dataset, frame),
        "attempt_overlap": attempt_overlap(
            frame, test_s1a["tx_id"].to_numpy(), test_s1a["p_all_raw"].to_numpy()
        ),
        "latency": latency_section(settings, dataset, frame, extras),
    }
    out = settings.reports_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    figures = plot_figures(report, extras, out, dataset.value)
    report["figures"] = [p.name for p in figures]
    path = out / f"{dataset.value}.json"
    path.write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8", newline="\n"
    )
    (out / f"{dataset.value}.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    logger.info("evaluation written", extra={"dataset": dataset.value})
    return path


# ---------------------------------------------------------------- figures
def _axes(title: str, xlabel: str, ylabel: str) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def _save(fig: Any, ax: Any, path: Path) -> Path:
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    if legend is not None:
        legend.set_title(None)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_figures(
    report: dict[str, Any], extras: dict[str, Any], out: Path, dataset: str
) -> list[Path]:
    """PR curves (full split and alert queue) and test reliability by calibrator.

    Args:
        report: The report being written.
        extras: Ablation scores and labels.
        out: Output directory.
        dataset: Dataset name.

    Returns:
        Written figure paths.
    """
    from sklearn.metrics import precision_recall_curve

    paths = []
    full = report["full_split"]
    fig, ax = _axes(f"{dataset}: every test transaction", "Recall", "Precision")
    for i, (name, m) in enumerate(full["models"].items()):
        ax.plot(
            m["pr_curve"]["recall"],
            m["pr_curve"]["precision"],
            color=SERIES[i],
            linestyle=STYLES[i],
            linewidth=2,
            label=f"{name} (PR-AUC {m['ranking']['average_precision']:.3f})",
        )
    b = full["baseline"]
    ax.plot(
        [b["recall"]],
        [b["precision"]],
        marker="o",
        markersize=8,
        color=SERIES[len(full["models"])],
        linestyle="none",
        label=f"rules ({b['name']}): one operating point",
    )
    ax.set_yscale("log")
    paths.append(_save(fig, ax, out / f"{dataset}_pr_full_split.png"))

    scores: dict[str, np.ndarray] = extras.get("scores") or {}
    if scores:
        y = extras["y_test"]
        fig, ax = _axes(f"{dataset}: ablation on {len(y):,} test alerts", "Recall", "Precision")
        ap = {r["name"]: r for r in report["ablation"]["steps"] if "ranking" in r}
        for i, (name, p) in enumerate(scores.items()):
            prec, rec, _ = precision_recall_curve(y, p)
            # Drop sklearn's artificial (recall 0, precision 1) end point and draw steps,
            # so coarse scores (the rules have four levels) are not drawn as slopes.
            ax.step(
                rec[:-1],
                prec[:-1],
                where="post",
                color=SERIES[i % len(SERIES)],
                linestyle=STYLES[i % len(STYLES)],
                linewidth=2,
                label=f"{name} ({ap[name]['ranking']['average_precision']:.3f})",
            )
        ax.axhline(float(np.mean(y)), color=INK_2, linewidth=1, linestyle=":")
        paths.append(_save(fig, ax, out / f"{dataset}_ablation_pr.png"))

    cal = report["calibration"]["system1a_lgbm_all"]["test"]
    fig, ax = _axes(
        f"{dataset}: test reliability by calibrator (50 equal-mass bins, log scale)",
        "Mean predicted probability",
        "Observed laundering rate",
    )
    picks = [
        k
        for k in ("raw (uncalibrated)", "isotonic", "isotonic_smoothed", "isotonic_recent_window")
        if k in cal
    ]
    lo, hi = 1.0, 0.0
    for i, name in enumerate(picks):
        # Log axes: at a 0.1% base rate almost every bin sits near zero on a linear
        # scale. Bins with no positives (or zero predictions) cannot be drawn on log.
        rel = [r for r in cal[name]["reliability"] if r["mean_pred"] > 0 and r["frac_pos"] > 0]
        xs = [r["mean_pred"] for r in rel]
        ys = [r["frac_pos"] for r in rel]
        lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
        ax.plot(
            xs,
            ys,
            color=SERIES[i],
            linestyle=STYLES[i],
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"{name} (ECE {cal[name]['ece_quantile']:.5f})",
        )
    ax.plot(
        [lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle=":", label="perfect calibration"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    paths.append(_save(fig, ax, out / f"{dataset}_calibration.png"))
    return paths


# ---------------------------------------------------------------- markdown
def _pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{d}f}%"


def _f(v: float | None, d: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the evaluation report.

    Args:
        report: Output of :func:`evaluate_command`.

    Returns:
        Markdown.
    """
    ds = report["dataset"]
    full = report["full_split"]
    b = full["baseline"]
    lines = [
        f"# Evaluation: {ds}",
        "",
        f"Regenerate with `aml-sentinel evaluate --dataset {ds}` (seed {report['seed']}). "
        "Synthetic data: none of this shows real-world performance.",
        "",
        "## 1. Every test transaction: rules vs LightGBM",
        "",
        f"{full['rows']:,} test transactions, {full['positives']:,} laundering. "
        f"Rule baseline `{b['name']}`: precision {_pct(b['precision'], 2)}, recall "
        f"{_pct(b['recall'])}, FPR {_pct(b['false_positive_rate'])}, "
        f"{b['alerts_per_day']:,.0f} alerts/day, "
        f"PR-AUC {_f(b['ranking']['average_precision'], 4)}.",
        "",
        "| Model | PR-AUC | P@100 | P@1,000 | P@10,000 | Alerts for the rules' recall "
        "(model vs rules) | FP reduction at matched recall | Recall at the rules' alert "
        "volume (model vs rules) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in full["models"].items():
        mr, mv, pk = m["matched_recall"], m["matched_volume"], m["precision_at_k"]
        lines.append(
            f"| {name} | {_f(m['ranking']['average_precision'], 4)} | {_pct(pk.get('100'))} "
            f"| {_pct(pk.get('1000'))} | {_pct(pk.get('10000'))} "
            f"| {mr['model_alerts']:,} vs {mr['baseline_alerts']:,} "
            f"| {_pct(mr['fp_reduction'])} | {_pct(mv['model_recall'])} vs "
            f"{_pct(mv['baseline_recall'])} |"
        )
    lines += [
        "",
        "The rules are a weak bar (ADR-009), so the false-positive reduction is large by "
        "construction; precision@k and PR-AUC describe the model itself.",
        "",
    ]
    ab = report["ablation"]
    if ab is None:
        lines += [
            "## 2. Ablation",
            "",
            "Not available: needs System 1b scores (`triage-text`).",
            "",
        ]
    else:
        a = ab["alerts"]
        lines += [
            "## 2. Ablation on the alert queue",
            "",
            f"Same {a['test']:,} sampled test alerts from the System 1a queue for every row "
            f"({a['test_positives']} laundering, base rate {_pct(a['test_base_rate'])}). "
            "Each step adds inputs to the previous one; every row is a logistic stacker "
            f"plus calibrator fitted on {a['val']:,} validation alerts. Δ is the paired-"
            "bootstrap PR-AUC change versus the previous row (row resampling, so the "
            "intervals are optimistic). FP@5% counts false positives among the top 5% of "
            "test alerts (the escalation budget).",
            "",
            "| Step | PR-AUC | ROC-AUC | P@100 | ECE | FP@5% | FP reduction vs rules | "
            "Δ PR-AUC [95% CI] | Note |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for r in ab["steps"]:
            if r.get("skipped"):
                lines.append(f"| {r['name']} | - | - | - | - | - | - | - | {r['skipped']} |")
                continue
            d = r["vs_previous"]
            delta = (
                "-" if d is None else f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
            )
            lines.append(
                f"| {r['name']} | {_f(r['ranking']['average_precision'])} "
                f"| {_f(r['ranking']['roc_auc'])} "
                f"| {_pct(r['precision_at_k']['100']['precision'])} | {_f(r['ece_quantile'], 4)} "
                f"| {r['top_5pct']['fp']} | {_pct(r['fp_reduction_vs_first'])} | {delta} "
                f"| {r['note']} |"
            )
        lines += ["", "Production scores on the same alerts (not refitted):", ""]
        lines += [
            "| Score | PR-AUC | P@100 | ECE | FP@5% | Source |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in ab["references"]:
            lines.append(
                f"| {r['name']} | {_f(r['ranking']['average_precision'])} "
                f"| {_pct(r['precision_at_k']['100']['precision'])} | {_f(r['ece_quantile'], 4)} "
                f"| {r['top_5pct']['fp']} | {r['note']} |"
            )
        lines += [
            "",
            "The ablation's '+ Laya' uses only Laya's escalate probability; production "
            "fusion also uses its expected risk and benign probability, which were not "
            "persisted, so the two rows differ slightly.",
            "",
            "### Reliability of p_triage on test alerts",
            "",
            "| p_triage band | Alerts | Mean p_triage | Observed laundering rate |",
            "|---|---:|---:|---:|",
        ]
        for band in ab["p_triage_reliability_test"]:
            lines.append(
                f"| {band['lo']:.3f}-{band['hi']:.3f} | {band['count']} "
                f"| {band['mean_pred']:.3f} | {_pct(band['frac_pos'])} |"
            )
        lines.append("")

    cal = report["calibration"]["system1a_lgbm_all"]
    lines += [
        "## 3. Calibration (System 1a LightGBM, every test transaction)",
        "",
        f"Chosen on validation cross-fit: `{cal['selected_on_validation']}` "
        "(held-out validation ECE: "
        + ", ".join(f"{k} {v:.5f}" for k, v in cal["val_crossfit_ece"].items())
        + f"). Recent-window candidates are fitted on the last validation day only "
        f"({cal['recent_window_rows']:,} rows).",
        "",
        "| Calibrator | Test ECE (quantile) | Brier | Mean predicted | Observed rate | Max "
        "predicted | Share at 1.0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in cal["test"].items():
        lines.append(
            f"| {name} | {m['ece_quantile']:.5f} | {m['brier']:.6f} | {_pct(m['mean_pred'], 3)} "
            f"| {_pct(m['observed_rate'], 3)} | {m['max_pred']:.3f} "
            f"| {_pct(m['share_at_one'], 3)} |"
        )
    fr = report["calibration"].get("final_ranker")
    if fr:
        lines += [
            "",
            f"Final ranker (alert queue): validation chose `{fr['selected']}`; held-out "
            "validation ECE "
            + ", ".join(f"{k} {v:.4f}" for k, v in (fr["val_crossfit_ece"] or {}).items())
            + ".",
        ]
    ov = report["attempt_overlap"]
    lines += [
        "",
        "## 4. Attempt overlap (ADR-003)",
        "",
        f"{ov['test_positives']} test positives; queue = top {ov['queue_size']:,} by "
        "LightGBM (top 1%). Each group is ranked against all test negatives.",
        "",
        "| Group | Positives | Share | Recall in queue | PR-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, g in ov.items():
        if isinstance(g, dict):
            lines.append(
                f"| {name} | {g['positives']} | {_pct(g['share_of_positives'])} "
                f"| {_pct(g['recall_in_queue'])} | {_f(g['pr_auc_vs_all_negatives'])} |"
            )
    lat = report["latency"]
    m = lat["machine"]
    lines += [
        "",
        "## 5. Per-stage latency",
        "",
        f"Machine: {m.get('processor') or m['platform']}, {m['cpu_count']} logical CPUs"
        + (f", GPU {m['gpu']}" if m.get("gpu") else "")
        + ". Single process.",
        "",
        "| Stage | Latency | Measured |",
        "|---|---:|---|",
    ]
    for name, s in lat["stages"].items():
        if "us_per_tx" in s:
            value = f"{s['us_per_tx']:.2f} µs/tx"
        elif "ms_per_alert" in s:
            value = f"{s['ms_per_alert']:.0f} ms/alert"
        elif "ms_p95" in s:
            value = f"{s['ms_median']:.0f} ms median, {s['ms_p95']:.0f} ms p95"
        elif "ms_median" in s:
            value = f"{s['ms_median']:.1f} ms median"
        else:
            value = f"{s['seconds']:.0f} s total"
        lines.append(f"| {name} | {value} | {s['measured']} |")
    lines += ["", "Figures: " + ", ".join(f"`{f}`" for f in report.get("figures", []))]
    return "\n".join(lines) + "\n"
