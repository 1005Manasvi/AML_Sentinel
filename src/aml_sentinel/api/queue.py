"""The analyst alert queue, ranked by the System 1 score ``p_triage``.

Owner's decision (ADR-020): the queue is ordered by System 1 ``p_triage`` because it
ranks better on test than the final combined score (PR-AUC 0.674 vs 0.651, ADR-018).
System 2 findings are attached as evidence, not used for ordering.

Where ``p_triage`` comes from, best first:

1. ``<dataset>.system1b_scores.parquet``: the fused System 1b probability (with Laya
   when a verified checkpoint was used). These are the seeded 4,000-alert samples
   per split that Laya scored, so on IBM data the queue is that sample.
2. ``<dataset>.system1a_scores.parquet``: calibrated LightGBM (``lgbm_all``) over the
   System 1a alert queue (top 1% of the split, at least 200 alerts).

Escalation reuses System 1b's rule: the threshold is the 95th percentile of the
validation alerts' ``p_triage``, applied to test alerts. Alerts below it are
auto-closed; System 2 evidence is attached only to escalated alerts, matching the
architecture (System 2 never runs on auto-closed alerts unless an analyst opens one).

Labels are never read: every score file is loaded with an explicit column list.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from aml_sentinel.config import Settings
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.models.system1b import ESCALATE_SHARE, queue_size
from aml_sentinel.ranking import TRIAGE_PREFERENCE
from aml_sentinel.system2.cases import CaseIndex
from aml_sentinel.system2.detectors import FindingKind
from aml_sentinel.system2.evaluate import run_cases

logger = logging.getLogger(__name__)

TABULAR_COLUMN: Final = "p_all_cal"
EVIDENCE_COLUMNS: Final = ("n_findings", *(f"n_{k.value}" for k in FindingKind))


def case_id_of(tx_id: int) -> str:
    """Case identifier of an alert.

    Args:
        tx_id: Alerted transaction id.

    Returns:
        ``CASE-<tx_id>``, the same id the SAR drafter and audit log use.
    """
    return f"CASE-{tx_id}"


@dataclass(frozen=True, slots=True)
class QueueEntry:
    """One test-period alert.

    Attributes:
        case_id: Case identifier.
        tx_id: Alerted transaction id.
        timestamp: Transaction time.
        p_triage: System 1 calibrated probability (the ranking score).
        escalated: True if ``p_triage`` reaches the validation-set threshold.
        evidence: System 2 finding counts (escalated alerts only).
        p_final: Final combined score (System 1 + System 2), informational only.
    """

    case_id: str
    tx_id: int
    timestamp: datetime
    p_triage: float
    escalated: bool
    evidence: dict[str, int] | None
    p_final: float | None


@dataclass(frozen=True, slots=True)
class ReliabilityBand:
    """How often test alerts with a ``p_triage`` in ``[lo, hi]`` were laundering.

    Attributes:
        lo: Lowest score in the band.
        hi: Highest score in the band.
        count: Test alerts in the band.
        frac_pos: Observed laundering rate.
    """

    lo: float
    hi: float
    count: int
    frac_pos: float


@dataclass(frozen=True, slots=True)
class AlertQueue:
    """Ranked alerts plus the provenance of their scores.

    Attributes:
        dataset: Dataset the alerts come from.
        triage_source: Model behind ``p_triage``.
        threshold: Escalation threshold on ``p_triage`` (fitted on validation).
        escalate_share: Target share of validation alerts escalated.
        entries: Alerts, highest ``p_triage`` first.
        reliability: Observed laundering rate per ``p_triage`` band on the held-out
            test period, from ``reports/evaluation`` (empty if not evaluated).
    """

    dataset: DatasetName
    triage_source: str
    threshold: float
    escalate_share: float
    entries: tuple[QueueEntry, ...]
    reliability: tuple[ReliabilityBand, ...] = ()
    _by_case: dict[str, tuple[int, QueueEntry]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index entries by case id."""
        index = {e.case_id: (rank, e) for rank, e in enumerate(self.entries, start=1)}
        object.__setattr__(self, "_by_case", index)

    def __iter__(self) -> Iterator[QueueEntry]:
        """Iterate in rank order."""
        return iter(self.entries)

    def __len__(self) -> int:
        """Number of alerts."""
        return len(self.entries)

    def band_of(self, p: float) -> ReliabilityBand | None:
        """The reliability band containing a score.

        Scores between two bands, or outside all of them, use the nearest band.

        Args:
            p: A ``p_triage`` value.

        Returns:
            The band, or ``None`` when no evaluation is available.
        """
        if not self.reliability:
            return None
        return min(
            self.reliability,
            key=lambda b: 0.0 if b.lo <= p <= b.hi else min(abs(p - b.lo), abs(p - b.hi)),
        )

    def get(self, case_id: str) -> QueueEntry | None:
        """Look up an alert.

        Args:
            case_id: Case identifier.

        Returns:
            The entry, or ``None`` if the case is not in the queue.
        """
        hit = self._by_case.get(case_id)
        return None if hit is None else hit[1]

    def rank_of(self, case_id: str) -> int | None:
        """1-based rank of a case.

        Args:
            case_id: Case identifier.

        Returns:
            The rank, or ``None`` if absent.
        """
        hit = self._by_case.get(case_id)
        return None if hit is None else hit[0]


def _triage_scores(settings: Settings, dataset: DatasetName) -> tuple[pd.DataFrame, str]:
    """``tx_id``, ``split``, ``p_triage`` for validation and test alerts."""
    s1b = settings.processed_dir / f"{dataset.value}.system1b_scores.parquet"
    if s1b.is_file():
        available = set(pq.read_schema(s1b).names)
        col = next((c for c in TRIAGE_PREFERENCE if c in available), None)
        if col is not None and "split" in available:
            table = pd.read_parquet(s1b, columns=["tx_id", "split", col])
            return table.rename(columns={col: "p_triage"}), col.removeprefix("p__")
    s1a = settings.processed_dir / f"{dataset.value}.system1a_scores.parquet"
    if not s1a.is_file():
        raise DataNotFoundError(f"no System 1 scores for {dataset.value}; run train-tabular first")
    table = pd.read_parquet(s1a, columns=["tx_id", "split", TABULAR_COLUMN])
    table["split"] = table["split"].astype("string")
    parts = []
    for split in ("val", "test"):
        rows = table.loc[table["split"] == split]
        k = queue_size(len(rows))
        parts.append(rows.nlargest(k, TABULAR_COLUMN, keep="first"))
    queue = pd.concat(parts, ignore_index=True).rename(columns={TABULAR_COLUMN: "p_triage"})
    return queue, "lgbm_all (calibrated)"


def _evidence(
    settings: Settings, dataset: DatasetName, index: CaseIndex, tx_ids: list[int]
) -> dict[int, dict[str, int]]:
    """Strict-horizon finding counts per escalated alert (precomputed where possible)."""
    out: dict[int, dict[str, int]] = {}
    path = settings.processed_dir / f"{dataset.value}.system2_evidence.parquet"
    if path.is_file():
        table = pd.read_parquet(path, columns=["tx_id", *EVIDENCE_COLUMNS])
        table = table.loc[table["tx_id"].isin(tx_ids)].drop_duplicates("tx_id")
        out.update(_counts(table))
    missing = [t for t in tx_ids if t not in out]
    if missing:
        out.update(_counts(run_cases(index, missing)))
    return out


def _counts(table: pd.DataFrame) -> dict[int, dict[str, int]]:
    values = table[["tx_id", *EVIDENCE_COLUMNS]].to_numpy(dtype=np.int64)
    return {int(v[0]): dict(zip(EVIDENCE_COLUMNS, map(int, v[1:]), strict=True)) for v in values}


def build_queue(
    settings: Settings, dataset: DatasetName, frame: pd.DataFrame, index: CaseIndex
) -> AlertQueue:
    """Assemble the ranked test-period queue from persisted pipeline outputs.

    Args:
        settings: Settings.
        dataset: Dataset variant.
        frame: Processed transactions (for timestamps).
        index: Case index over ``frame`` (for evidence not precomputed).

    Returns:
        The queue.

    Raises:
        DataNotFoundError: If no System 1 scores exist.
        DataValidationError: If there are no validation or test alerts.
    """
    scores, source = _triage_scores(settings, dataset)
    split = scores["split"].astype("string")
    val, test = scores.loc[split == "val"], scores.loc[split == "test"]
    if val.empty or test.empty:
        raise DataValidationError("the alert queue needs validation and test alerts")
    # Same rule as System 1b's escalation_metrics, so the API reproduces the reports.
    threshold = float(np.quantile(val["p_triage"].to_numpy(dtype=np.float64), 1 - ESCALATE_SHARE))
    test = test.sort_values(["p_triage", "tx_id"], ascending=[False, True], kind="stable")
    tx_ids = [int(t) for t in test["tx_id"]]
    p = test["p_triage"].to_numpy(dtype=np.float64)
    escalated = p >= threshold
    to_explain = [t for t, e in zip(tx_ids, escalated, strict=True) if e]
    evidence = _evidence(settings, dataset, index, to_explain)

    final_path = settings.processed_dir / f"{dataset.value}.final_scores.parquet"
    p_final: dict[int, float] = {}
    if final_path.is_file():
        final = pd.read_parquet(final_path, columns=["tx_id", "p_final"])
        p_final = {int(t): float(v) for t, v in zip(final["tx_id"], final["p_final"], strict=True)}

    ts = pd.Series(frame[Col.TIMESTAMP].to_numpy(), index=frame[Col.TX_ID].to_numpy())
    missing = [t for t in tx_ids if t not in ts.index]
    if missing:
        raise DataValidationError(f"{len(missing)} queued alerts are not in the processed data")
    entries = tuple(
        QueueEntry(
            case_id=case_id_of(tx),
            tx_id=tx,
            timestamp=pd.Timestamp(ts[tx]).to_pydatetime(),
            p_triage=float(prob),
            escalated=bool(esc),
            evidence=evidence.get(tx) if esc else None,
            p_final=p_final.get(tx),
        )
        for tx, prob, esc in zip(tx_ids, p, escalated, strict=True)
    )
    logger.info(
        "alert queue built",
        extra={
            "dataset": dataset.value,
            "alerts": len(entries),
            "escalated": int(escalated.sum()),
            "auto_closed": int((~escalated).sum()),
        },
    )
    bands = _reliability(settings, dataset, source)
    return AlertQueue(dataset, source, threshold, ESCALATE_SHARE, entries, bands)


def _reliability(
    settings: Settings, dataset: DatasetName, source: str
) -> tuple[ReliabilityBand, ...]:
    """Test-period reliability bands, only if measured for this very score."""
    path = settings.reports_dir / "evaluation" / f"{dataset.value}.json"
    if not path.is_file():
        return ()
    ablation = json.loads(path.read_text(encoding="utf-8")).get("ablation") or {}
    if str(ablation.get("triage_column", "")).removeprefix("p__") != source:
        return ()
    return tuple(
        ReliabilityBand(float(b["lo"]), float(b["hi"]), int(b["count"]), float(b["frac_pos"]))
        for b in ablation.get("p_triage_reliability_test", [])
    )
