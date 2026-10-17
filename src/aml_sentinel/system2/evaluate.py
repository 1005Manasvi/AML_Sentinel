"""Measure the System 2 detectors on alert cases and produce per-alert graph evidence.

Detectors produce evidence, not verdicts, so the evaluation asks three questions:

* **Separation.** How often does each detector fire on laundering vs. benign alerts
  (precision, lift over the base rate)?
* **Agreement.** For laundering alerts with a documented typology, does the
  *matching* detector fire?
* **Cost.** How long do case extraction and detection take per alert?

The per-alert feature table (counts and strengths per detector) is persisted for
the final ranking in Phase 6. It is computed from past-only cases and never uses
labels.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.patterns import Typology
from aml_sentinel.system2.cases import CaseConfig, CaseIndex
from aml_sentinel.system2.detectors import DetectorConfig, Finding, FindingKind, run_detectors

TYPOLOGY_DETECTOR: Final[dict[str, FindingKind]] = {
    Typology.CYCLE.value: FindingKind.CIRCULAR,
    Typology.STRUCTURING.value: FindingKind.STRUCTURING,
    Typology.FAN_IN.value: FindingKind.FAN_IN,
    Typology.FAN_OUT.value: FindingKind.FAN_OUT,
    Typology.SCATTER_GATHER.value: FindingKind.SCATTER_GATHER,
    Typology.GATHER_SCATTER.value: FindingKind.GATHER_SCATTER,
}


def evidence_features(findings: Sequence[Finding]) -> dict[str, float]:
    """Summarise findings as numeric per-alert features.

    Args:
        findings: Findings for one case.

    Returns:
        ``n_<kind>`` and ``max_<kind>`` for every detector, plus ``n_findings``.
    """
    feats: dict[str, float] = {"n_findings": float(len(findings))}
    for kind in FindingKind:
        mine = [f.strength for f in findings if f.kind is kind]
        feats[f"n_{kind.value}"] = float(len(mine))
        feats[f"max_{kind.value}"] = float(max(mine, default=0.0))
    return feats


def run_cases(
    index: CaseIndex,
    tx_ids: Sequence[int],
    *,
    case_config: CaseConfig | None = None,
    detector_config: DetectorConfig | None = None,
) -> pd.DataFrame:
    """Build a case and run every detector for each alert.

    Args:
        index: Case index over the processed frame.
        tx_ids: Alert transaction ids.
        case_config: Case limits.
        detector_config: Detector thresholds.

    Returns:
        One row per alert: ``tx_id``, the evidence features, ``case_rows``,
        ``truncated`` and ``ms`` (time for case + detectors).
    """
    rows = []
    for tx in tx_ids:
        start = time.perf_counter()
        case = index.build(int(tx), case_config)
        findings = run_detectors(case, detector_config)
        ms = 1000 * (time.perf_counter() - start)
        rows.append(
            {
                "tx_id": int(tx),
                **evidence_features(findings),
                "case_rows": float(len(case.edges)),
                "truncated": float(case.truncated),
                "ms": ms,
            }
        )
    return pd.DataFrame(rows)


def separation(evidence: pd.DataFrame, y: npt.ArrayLike) -> dict[str, Any]:
    """Per-detector firing rates on laundering vs. benign alerts.

    Args:
        evidence: Output of :func:`run_cases`.
        y: Labels aligned with ``evidence``.

    Returns:
        Base rate plus, per detector and for "any", the firing rates, precision and lift.
    """
    t = np.asarray(y).astype(bool)
    base = float(t.mean()) if t.size else 0.0
    out: dict[str, Any] = {"alerts": int(t.size), "positives": int(t.sum()), "base_rate": base}
    columns = {"any": evidence["n_findings"] > 0}
    columns.update({k.value: evidence[f"n_{k.value}"] > 0 for k in FindingKind})
    for name, fired_series in columns.items():
        fired = fired_series.to_numpy(dtype=bool)
        n = int(fired.sum())
        precision = float(t[fired].mean()) if n else None
        out[name] = {
            "fires": n,
            "rate_on_laundering": float(fired[t].mean()) if t.any() else None,
            "rate_on_benign": float(fired[~t].mean()) if (~t).any() else None,
            "precision": precision,
            "lift": None if precision is None or base == 0 else precision / base,
        }
    return out


def typology_agreement(evidence: pd.DataFrame, typologies: Sequence[str | None]) -> dict[str, Any]:
    """Share of documented positives where the matching detector fired.

    Args:
        evidence: Output of :func:`run_cases`.
        typologies: IBM/generator typology per alert (``None`` for benign or undocumented).

    Returns:
        ``{typology: {"n", "matched", "rate", "any_rate"}}``. Typologies without a
        dedicated detector report only ``any_rate``.
    """
    out: dict[str, Any] = {}
    typ = pd.Series(list(typologies), dtype="object")
    for name in sorted({t for t in typologies if t}):
        mask = (typ == name).to_numpy()
        any_rate = float((evidence["n_findings"].to_numpy()[mask] > 0).mean())
        kind = TYPOLOGY_DETECTOR.get(name)
        if kind is None:
            out[name] = {"n": int(mask.sum()), "matched": None, "rate": None, "any_rate": any_rate}
            continue
        hit = evidence[f"n_{kind.value}"].to_numpy()[mask] > 0
        out[name] = {
            "n": int(mask.sum()),
            "matched": int(hit.sum()),
            "rate": float(hit.mean()),
            "any_rate": any_rate,
        }
    return out


def cost(evidence: pd.DataFrame) -> dict[str, float]:
    """Latency and case-size summary.

    Args:
        evidence: Output of :func:`run_cases`.

    Returns:
        Median and p95 milliseconds, median case rows and the truncated share.
    """
    return {
        "ms_median": float(evidence["ms"].median()),
        "ms_p95": float(evidence["ms"].quantile(0.95)),
        "case_rows_median": float(evidence["case_rows"].median()),
        "truncated_share": float(evidence["truncated"].mean()),
    }
