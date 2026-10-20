"""Distribution drift between time periods: PSI per feature and for the model score.

Phase 3 found that calibration fitted on validation made test calibration worse
(ADR-011). This report measures how far the inputs and the score moved between the
periods, so that finding has a quantified cause rather than a guess. It uses the
Population Stability Index (PSI), the standard monitoring statistic in credit and AML
model risk management.

PSI thresholds used for labels are the common industry rule of thumb (below 0.1
stable, 0.1 to 0.25 moderate shift, above 0.25 major shift), not a statistical test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")  # headless: reports are rendered to files
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.features.builder import build_features, feature_columns
from aml_sentinel.models.system1a import split_masks

logger = logging.getLogger(__name__)

DEFAULT_BINS: Final = 10
EPSILON: Final = 1e-6  # keeps empty bins from producing log(0)
STABLE: Final = 0.1
MAJOR: Final = 0.25
COMPARISONS: Final = (("train", "test"), ("val", "test"), ("train", "val"))


def psi(expected: npt.ArrayLike, actual: npt.ArrayLike, bins: int = DEFAULT_BINS) -> float:
    """Population Stability Index of ``actual`` against ``expected``.

    Bins are quantiles of ``expected``. Repeated edges are merged, so binary and
    low-cardinality features get one bin per distinct value instead of empty bins.

    Args:
        expected: Reference sample (the earlier period).
        actual: Comparison sample (the later period).
        bins: Target number of quantile bins.

    Returns:
        PSI (0 means identical binned distributions; it is never negative).

    Raises:
        ValueError: If either sample is empty after dropping NaNs, or ``bins < 2``.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if e.size == 0 or a.size == 0:
        raise ValueError("both samples must contain values")
    inner = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    # side="right": a value equal to an edge goes to the upper bin, for both samples.
    e_share = np.bincount(np.searchsorted(inner, e, side="right"), minlength=inner.size + 1)
    a_share = np.bincount(np.searchsorted(inner, a, side="right"), minlength=inner.size + 1)
    p = np.clip(e_share / e.size, EPSILON, None)
    q = np.clip(a_share / a.size, EPSILON, None)
    return float(np.sum((q - p) * np.log(q / p)))


def label(value: float) -> str:
    """Rule-of-thumb reading of a PSI value.

    Args:
        value: PSI.

    Returns:
        ``stable``, ``moderate`` or ``major``.
    """
    if value < STABLE:
        return "stable"
    return "moderate" if value < MAJOR else "major"


def drift_section(
    features: pd.DataFrame,
    split_of: dict[str, npt.NDArray[np.bool_]],
    columns: list[str],
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, dict[str, float]]:
    """PSI of every column for each period comparison.

    Args:
        features: Feature (or score) frame aligned with the masks.
        split_of: Boolean row masks named ``train``, ``val`` and ``test``.
        columns: Columns to compare.
        bins: Quantile bins.

    Returns:
        ``{"train->test": {column: psi}, ...}``, each sorted by PSI descending.
    """
    out: dict[str, dict[str, float]] = {}
    for ref, cmp in COMPARISONS:
        values = {
            c: psi(features.loc[split_of[ref], c], features.loc[split_of[cmp], c], bins)
            for c in columns
        }
        out[f"{ref}->{cmp}"] = dict(sorted(values.items(), key=lambda kv: -kv[1]))
    return out


def build_drift_report(
    frame: pd.DataFrame, settings: Settings, dataset: DatasetName, scores: pd.DataFrame | None
) -> dict[str, Any]:
    """Compute the drift report for one dataset.

    Args:
        frame: Processed transactions (sorted, with ``split``).
        settings: Settings (split boundaries, seed).
        dataset: Dataset.
        scores: System 1a scores (``tx_id``, ``split``, ``p_all_raw``) or ``None``.

    Returns:
        JSON-serialisable report.
    """
    masks = split_masks(frame, settings.train_end)
    # Trees were fitted on "fit"; "stop" is the early-stopping day. Training = both.
    split_of = {"train": masks["fit"] | masks["stop"], "val": masks["val"], "test": masks["test"]}
    columns = feature_columns(include_shortcuts=True)
    features = build_features(frame)[columns]
    y = frame[Col.IS_LAUNDERING].to_numpy()
    report: dict[str, Any] = {
        "dataset": dataset.value,
        "seed": settings.random_seed,
        "bins": DEFAULT_BINS,
        "thresholds": {"stable_below": STABLE, "major_from": MAJOR},
        "rows": {k: int(m.sum()) for k, m in split_of.items()},
        "base_rate": {k: float(y[m].mean()) for k, m in split_of.items()},
        "features": drift_section(features, split_of, columns),
    }
    if scores is not None:
        # The calibrator is fitted on val and applied on test, so val->test is the
        # comparison that matters for calibration.
        split = scores["split"].astype("string")
        raw = scores["p_all_raw"].to_numpy(dtype=np.float64)
        report["score_psi_val_to_test"] = psi(raw[split == "val"], raw[split == "test"])
        report["score_mean"] = {s: float(raw[split == s].mean()) for s in ("val", "test")}
    return report


def render_markdown(report: dict[str, Any], top: int = 10) -> str:
    """Human-readable drift report.

    Args:
        report: Output of :func:`build_drift_report`.
        top: Features listed per comparison.

    Returns:
        Markdown text.
    """
    lines = [
        f"# Drift report: {report['dataset']}",
        "",
        f"PSI with {report['bins']} quantile bins of the earlier period. Rule of thumb: "
        f"< {STABLE} stable, {STABLE}-{MAJOR} moderate, >= {MAJOR} major shift.",
        "",
        "| Period | Rows | Laundering rate |",
        "|---|---:|---:|",
    ]
    for k in ("train", "val", "test"):
        lines.append(f"| {k} | {report['rows'][k]:,} | {100 * report['base_rate'][k]:.3f}% |")
    if "score_psi_val_to_test" in report:
        v = report["score_psi_val_to_test"]
        m = report["score_mean"]
        lines += [
            "",
            f"**LightGBM score, validation → test:** PSI {v:.4f} ({label(v)}); mean raw "
            f"score {m['val']:.5f} → {m['test']:.5f}.",
        ]
    for comparison, values in report["features"].items():
        n_major = sum(1 for v in values.values() if v >= MAJOR)
        n_mod = sum(1 for v in values.values() if STABLE <= v < MAJOR)
        lines += [
            "",
            f"## {comparison}: {n_major} major, {n_mod} moderate of {len(values)} features",
            "",
            "| Feature | PSI | Reading |",
            "|---|---:|---|",
        ]
        lines += [f"| {f} | {v:.4f} | {label(v)} |" for f, v in list(values.items())[:top]]
    return "\n".join(lines) + "\n"


def plot_drift(report: dict[str, Any], path: Path, top: int = 12) -> Path:
    """Bar chart of the most-shifted features, train→test and val→test.

    Args:
        report: Output of :func:`build_drift_report`.
        path: PNG path.
        top: Features drawn.

    Returns:
        ``path``.
    """
    tt = report["features"]["train->test"]
    vt = report["features"]["val->test"]
    names = list(tt)[:top][::-1]
    yy = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.4))
    ax.barh(yy + 0.2, [tt[n] for n in names], height=0.4, label="train → test", color="#2471a3")
    ax.barh(yy - 0.2, [vt[n] for n in names], height=0.4, label="val → test", color="#e67e22")
    for x, style in ((STABLE, ":"), (MAJOR, "--")):
        ax.axvline(x, color="#7f8c8d", linestyle=style, linewidth=1)
    ax.set_yticks(yy, names)
    ax.set_xlabel("PSI (dotted 0.1 = moderate, dashed 0.25 = major)")
    ax.set_title(f"{report['dataset']}: most-shifted features")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def drift_command(settings: Settings, dataset: DatasetName) -> Path:
    """Write ``reports/drift/<dataset>.json``, ``.md`` and ``_psi.png``.

    Args:
        settings: Settings.
        dataset: Dataset.

    Returns:
        The JSON report path.
    """
    frame = load_processed(settings.processed_dir, dataset)
    scores_path = settings.processed_dir / f"{dataset.value}.system1a_scores.parquet"
    scores = (
        pd.read_parquet(scores_path, columns=["tx_id", "split", "p_all_raw"])
        if scores_path.is_file()
        else None
    )
    report = build_drift_report(frame, settings, dataset, scores)
    out_dir = settings.reports_dir / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset.value}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{dataset.value}.md").write_text(render_markdown(report), encoding="utf-8")
    plot_drift(report, out_dir / f"{dataset.value}_psi.png")
    logger.info("drift report written", extra={"dataset": dataset.value, "file": path.name})
    return path
