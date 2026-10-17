"""Fit, evaluate and report the rule-engine baseline.

Two variants are reported, both fitted on the training split only:

* ``static``: textbook thresholds (10k USD; velocity at the train 99.9th percentile).
* ``tuned``: thresholds and rule subset grid-searched for train F1.

The **baseline to beat** is whichever variant has the better *validation* F1
(``report["selected"]``), so later models are compared with the stronger one and
never with a strawman.

Each variant is evaluated on validation and test; the test split is touched once,
for reporting only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.patterns import TYPOLOGY
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.data.splits import SplitName
from aml_sentinel.evaluation.metrics import binary_metrics, recall_by_group
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.rules.engine import (
    ALERT,
    Rule,
    RuleConfig,
    apply_rules,
    fit_static,
    select_high_risk_banks,
    trailing_counts,
    tune,
)

logger = logging.getLogger(__name__)

SPLIT_COL = "split"
EVAL_SPLITS = (SplitName.VAL, SplitName.TEST)


def evaluate_config(
    frame: pd.DataFrame, config: RuleConfig, velocity: npt.NDArray[np.int64]
) -> dict[str, Any]:
    """Evaluate one rule config on one split.

    Args:
        frame: Transactions of a single split.
        config: Rule config.
        velocity: Trailing counts for the same rows.

    Returns:
        Overall metrics, per-rule metrics (each rule alone), alerts per day and
        recall by typology.
    """
    flags = apply_rules(frame, config, velocity)
    y = frame[Col.IS_LAUNDERING].to_numpy() == 1
    n_days = max(1, int(frame[Col.TIMESTAMP].dt.normalize().nunique()))
    overall = binary_metrics(y, flags[ALERT])
    return {
        "overall": overall.to_dict(),
        "alerts_per_day": overall.alerts / n_days,
        "per_rule": {r.value: binary_metrics(y, flags[r.value]).to_dict() for r in Rule},
        "recall_by_typology": (
            recall_by_group(y, flags[ALERT].to_numpy(), frame[TYPOLOGY])
            if TYPOLOGY in frame.columns
            else {}
        ),
    }


def run_baseline(frame: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    """Fit both baseline variants on train and evaluate them on val and test.

    Args:
        frame: Processed transactions with a ``split`` column, sorted by time.
        seed: Seed for the synthetic high-risk bank list.

    Returns:
        JSON-serialisable report.

    Raises:
        DataValidationError: If the frame has no ``split`` column or no train rows.
    """
    if SPLIT_COL not in frame.columns:
        raise DataValidationError("frame has no split column; re-run prepare-data")
    # Velocity is computed over the whole timeline: each row only counts its own
    # past, so test rows legitimately see late-validation history (as in production).
    velocity = trailing_counts(frame)
    masks = {s: (frame[SPLIT_COL].astype("string") == s.value).to_numpy() for s in SplitName}
    train = frame.loc[masks[SplitName.TRAIN]]
    if len(train) == 0:
        raise DataValidationError("no training rows")
    train_velocity = velocity[masks[SplitName.TRAIN]]

    banks = select_high_risk_banks(train, seed=seed)
    tuned = tune(train, train_velocity, banks)
    variants = {"static": fit_static(train, banks), "tuned": tuned.config}

    report: dict[str, Any] = {
        "seed": seed,
        "n_high_risk_banks": len(banks),
        "tuning": {"train_f1": tuned.train_f1, "n_candidates": tuned.n_candidates},
        "variants": {},
    }
    for name, cfg in variants.items():
        report["variants"][name] = {
            "config": cfg.to_dict(),
            **{
                s.value: evaluate_config(frame.loc[masks[s]], cfg, velocity[masks[s]])
                for s in EVAL_SPLITS
            },
        }
    report["selected"] = select_by_validation(report["variants"])
    return report


def select_by_validation(variants: dict[str, Any]) -> str:
    """Pick the baseline to beat: the variant with the best **validation** F1.

    Selecting on test would leak; selecting on train alone over-trusted the tuned
    variant on HI-Small, whose train-optimal rule did not generalise.

    Args:
        variants: ``report["variants"]`` from :func:`run_baseline`.

    Returns:
        Name of the selected variant (first one wins ties).
    """

    def val_f1(name: str) -> float:
        f1 = variants[name][SplitName.VAL.value]["overall"]["f1"]
        return -1.0 if f1 is None else float(f1)

    return max(variants, key=val_f1)


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def render_markdown(report: dict[str, Any], dataset: str) -> str:
    """Render the headline baseline table as Markdown.

    Args:
        report: Output of :func:`run_baseline`.
        dataset: Dataset name for the title.

    Returns:
        Markdown text.
    """
    lines = [
        f"# Rule baseline: {dataset}",
        "",
        "Generated by `aml-sentinel baseline`. Measured, transaction-level, on time-based "
        "splits. The jurisdiction rule uses a **synthetic** bank list (ADR-008).",
        "",
        "| Variant | Split | Alerts | Alerts/day | Precision | Recall | FPR | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, variant in report["variants"].items():
        for split in (s.value for s in EVAL_SPLITS):
            m = variant[split]["overall"]
            lines.append(
                f"| {name} | {split} | {m['alerts']:,} | {variant[split]['alerts_per_day']:,.0f} "
                f"| {_pct(m['precision'])} | {_pct(m['recall'])} "
                f"| {_pct(m['false_positive_rate'], 3)} | {_pct(m['f1'])} |"
            )
    lines += ["", f"**Baseline to beat (best validation F1): `{report['selected']}`**", ""]
    lines += ["## Configs", ""]
    for name, variant in report["variants"].items():
        cfg = dict(variant["config"])
        cfg["high_risk_banks"] = f"{len(cfg['high_risk_banks'])} banks (synthetic)"
        lines.append(f"- **{name}**: `{json.dumps(cfg)}`")
    lines += [
        "",
        "## Test recall by typology (selected)",
        "",
        "| Typology | Positives | Caught | Recall |",
        "|---|---:|---:|---:|",
    ]
    for typ, r in report["variants"][report["selected"]]["test"]["recall_by_typology"].items():
        lines.append(f"| {typ} | {r['positives']:,} | {r['caught']:,} | {_pct(r['recall'], 1)} |")
    return "\n".join(lines) + "\n"


def baseline_command(settings: Settings, dataset: DatasetName) -> Path:
    """Load a processed dataset, run the baseline and write the reports.

    Args:
        settings: Validated settings.
        dataset: Dataset variant.

    Returns:
        Path of the JSON report.
    """
    frame = load_processed(settings.processed_dir, dataset)
    report = {"dataset": dataset.value, **run_baseline(frame, seed=settings.random_seed)}
    out_dir = settings.reports_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset.value}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{dataset.value}.md").write_text(
        render_markdown(report, dataset.value), encoding="utf-8", newline="\n"
    )
    logger.info("baseline written", extra={"dataset": dataset.value, "file": json_path.name})
    return json_path
