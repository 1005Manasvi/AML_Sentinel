"""``detect-patterns``: run the System 2 detectors and write their evaluation report.

Alert sets:

* IBM datasets: the same seeded val/test samples of the System 1a alert queue that
  Phase 4 used. The "escalated" subset is the top 5% of the test sample by the best
  available System 1b fused score.
* The bundled sample: every laundering transaction plus a seeded sample of benign
  ones, because the queue there is tiny.

Two investigation horizons are reported. ``strict`` (0 h) is the headline and the
only one saved as evidence for Phase 6. ``next_day`` (24 h) is labelled as a
different scenario.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.patterns import TYPOLOGY
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import DataNotFoundError
from aml_sentinel.models.system1b import build_alert_sets
from aml_sentinel.system2.cases import CaseConfig, CaseIndex
from aml_sentinel.system2.evaluate import cost, run_cases, separation, typology_agreement

logger = logging.getLogger(__name__)

HORIZONS: Final = {"strict": timedelta(0), "next_day": timedelta(hours=24)}
FUSED_PREFERENCE: Final = (
    "p__fused[lgbm_all+laya[full]]",
    "p__fused[lgbm_all+laya[head_only]]",
    "p__fused[lgbm_all]",
)


def _alerts(settings: Settings, dataset: DatasetName, frame: pd.DataFrame) -> dict[str, Any]:
    if dataset is DatasetName.SAMPLE:
        rng = np.random.default_rng(settings.random_seed)
        y = frame[Col.IS_LAUNDERING].to_numpy() == 1
        benign = rng.choice(
            np.flatnonzero(~y), size=min(2 * int(y.sum()), int((~y).sum())), replace=False
        )
        rows = np.sort(np.concatenate([np.flatnonzero(y), benign]))
        return {"test": rows, "val": rows[:0]}
    scores_path = settings.processed_dir / f"{dataset.value}.system1a_scores.parquet"
    if not scores_path.is_file():
        raise DataNotFoundError(f"{scores_path.name} missing; run train-tabular first")
    sets = build_alert_sets(frame, pd.read_parquet(scores_path), seed=settings.random_seed)
    return {split: a.rows for split, a in sets.items()}


def _escalated(settings: Settings, dataset: DatasetName, tx_ids: np.ndarray) -> np.ndarray | None:
    path = settings.processed_dir / f"{dataset.value}.system1b_scores.parquet"
    if not path.is_file():
        return None
    s1b = pd.read_parquet(path)
    col = next((c for c in FUSED_PREFERENCE if c in s1b.columns), None)
    if col is None:
        return None
    k = max(1, round(0.05 * len(s1b)))
    top = set(s1b.nlargest(k, col)["tx_id"].astype(int))
    return np.array([t in top for t in tx_ids])


def _section(evidence: pd.DataFrame, y: np.ndarray, typ: list[str | None]) -> dict[str, Any]:
    return {
        "separation": separation(evidence, y),
        "typology_agreement": typology_agreement(evidence, typ),
        "cost": cost(evidence),
    }


def detect_command(
    settings: Settings, dataset: DatasetName, *, alert_limit: int | None = None
) -> Path:
    """Run detectors on the alert sets and write report plus Phase 6 evidence.

    Args:
        settings: Settings.
        dataset: Dataset variant.
        alert_limit: Use only the first N alerts per split (quick runs and tests).

    Returns:
        Path of the JSON report.
    """
    frame = load_processed(settings.processed_dir, dataset)
    index = CaseIndex(frame)
    alert_rows = _alerts(settings, dataset, frame)
    if alert_limit is not None:
        alert_rows = {k: v[:alert_limit] for k, v in alert_rows.items()}
    y_all = frame[Col.IS_LAUNDERING].to_numpy().astype(np.int8)
    typ_all = frame[TYPOLOGY].astype("string").to_numpy()
    tx_all = frame[Col.TX_ID].to_numpy()

    report: dict[str, Any] = {"dataset": dataset.value, "horizons": {}}
    saved = []
    for name, horizon in HORIZONS.items():
        cfg = CaseConfig(horizon=horizon)
        report["horizons"][name] = {"hours": horizon.total_seconds() / 3600}
        for split, rows in alert_rows.items():
            if rows.size == 0:
                continue
            ev = run_cases(index, tx_all[rows].tolist(), case_config=cfg)
            y = y_all[rows]
            typ = [t if isinstance(t, str) else None for t in typ_all[rows]]
            if split == "test":
                section = _section(ev, y, typ)
                esc = _escalated(settings, dataset, tx_all[rows])
                if esc is not None and esc.any():
                    section["escalated_subset"] = _section(
                        ev.loc[esc].reset_index(drop=True),
                        y[esc],
                        [t for t, e in zip(typ, esc, strict=True) if e],
                    )
                report["horizons"][name]["test"] = section
            if name == "strict":
                saved.append(ev.assign(split=split))
        logger.info("detectors evaluated", extra={"horizon": name})

    evidence_path = settings.processed_dir / f"{dataset.value}.system2_evidence.parquet"
    pd.concat(saved, ignore_index=True).to_parquet(evidence_path, index=False)

    out_dir = settings.reports_dir / "system2"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset.value}_detectors.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{dataset.value}_detectors.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    return json_path


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the detector evaluation.

    Args:
        report: Output of :func:`detect_command`.

    Returns:
        Markdown text.
    """
    lines = [
        f"# System 2 detectors: {report['dataset']}",
        "",
        "Detectors produce evidence, not verdicts. `strict` cases see nothing after "
        "the alert (headline). `next_day` cases may see 24 h past it, modelling a "
        "later investigation (a different scenario).",
        "",
    ]
    for name, h in report["horizons"].items():
        for label, sec in (
            ("all sampled test alerts", h.get("test")),
            ("escalated subset", (h.get("test") or {}).get("escalated_subset")),
        ):
            if not sec:
                continue
            sep = sec["separation"]
            lines += [
                f"## Horizon `{name}`: {label}",
                "",
                f"{sep['alerts']:,} alerts, {sep['positives']} laundering "
                f"(base rate {_pct(sep['base_rate'])}). Median {sec['cost']['ms_median']:.0f} ms "
                f"per case (p95 {sec['cost']['ms_p95']:.0f} ms); "
                f"{_pct(sec['cost']['truncated_share'])} of cases hit a hub cap.",
                "",
                "| Detector | Fires | On laundering | On benign | Precision | Lift |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for det, v in sep.items():
                if not isinstance(v, dict):
                    continue
                lift = "n/a" if v["lift"] is None else f"{v['lift']:.2f}x"
                lines.append(
                    f"| {det} | {v['fires']:,} | {_pct(v['rate_on_laundering'])} "
                    f"| {_pct(v['rate_on_benign'])} | {_pct(v['precision'])} | {lift} |"
                )
            lines += [
                "",
                "| Typology | Alerts | Matching detector fired | Any detector fired |",
                "|---|---:|---:|---:|",
            ]
            for typ, v in sec["typology_agreement"].items():
                lines.append(f"| {typ} | {v['n']} | {_pct(v['rate'])} | {_pct(v['any_rate'])} |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
