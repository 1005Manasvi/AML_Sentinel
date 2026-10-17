"""The case fact sheet: the *only* information a SAR draft may use.

Every account, amount and date in a draft must trace back to this structure. It is
built from the case's own transactions and System 2 findings, plus the calibrated
risk score. Labels are never included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.system2.cases import Case
from aml_sentinel.system2.detectors import Finding


@dataclass(frozen=True, slots=True)
class TxFact:
    """One transaction as it may appear in a SAR.

    Attributes:
        tx_id: Transaction id.
        date: ISO date (``YYYY-MM-DD``).
        payer: Payer account id.
        payee: Payee account id.
        amount_usd: Amount in USD, rounded to cents.
        payment_format: Payment channel.
    """

    tx_id: int
    date: str
    payer: str
    payee: str
    amount_usd: float
    payment_format: str


@dataclass(frozen=True, slots=True)
class PatternFact:
    """One detector finding as it may appear in a SAR.

    Attributes:
        kind: Detector name.
        accounts: Involved accounts.
        tx_ids: Supporting transactions.
        total_usd: Sum over the supporting transactions, rounded to cents.
        first_date: Earliest supporting date.
        last_date: Latest supporting date.
        size: Detector strength (e.g. number of counterparties).
    """

    kind: str
    accounts: tuple[str, ...]
    tx_ids: tuple[int, ...]
    total_usd: float
    first_date: str
    last_date: str
    size: float


@dataclass(frozen=True, slots=True)
class CaseFacts:
    """Everything a SAR draft may state about one case.

    Attributes:
        case_id: Stable case identifier (``CASE-<alert tx id>``).
        alert: The alerted transaction.
        subjects: The alert's payer and payee.
        transactions: All case transactions, time-ordered.
        patterns: Detector findings.
        risk_score: Calibrated risk probability, or ``None`` if unavailable.
    """

    case_id: str
    alert: TxFact
    subjects: tuple[str, str]
    transactions: tuple[TxFact, ...]
    patterns: tuple[PatternFact, ...]
    risk_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accounts(self) -> frozenset[str]:
        """Every account id that may legitimately appear in a draft."""
        accts = {t.payer for t in self.transactions} | {t.payee for t in self.transactions}
        for p in self.patterns:
            accts.update(p.accounts)
        return frozenset(accts | set(self.subjects))

    @property
    def amounts(self) -> frozenset[float]:
        """Every USD amount (cents) that may appear: payments plus pattern totals."""
        return frozenset(
            {t.amount_usd for t in self.transactions}
            | {p.total_usd for p in self.patterns}
            | {round(sum(t.amount_usd for t in self.transactions), 2)}
        )

    @property
    def dates(self) -> frozenset[str]:
        """Every date that may appear."""
        return frozenset({t.date for t in self.transactions})


def _tx(row: Any) -> TxFact:
    return TxFact(
        tx_id=int(row[Col.TX_ID.value]),
        date=pd.Timestamp(row[Col.TIMESTAMP.value]).strftime("%Y-%m-%d"),
        payer=str(row[Col.FROM_ID.value]),
        payee=str(row[Col.TO_ID.value]),
        amount_usd=round(float(row[Col.AMOUNT_PAID_USD.value]), 2),
        payment_format=str(row[Col.PAYMENT_FORMAT.value]),
    )


def build_case_facts(
    case: Case, findings: Sequence[Finding], *, risk_score: float | None = None
) -> CaseFacts:
    """Assemble the fact sheet for a case.

    Args:
        case: Past-only case from :class:`~aml_sentinel.system2.cases.CaseIndex`.
        findings: Detector findings for that case.
        risk_score: Final calibrated risk probability, if known.

    Returns:
        The case facts.
    """
    txs = tuple(_tx(r) for _, r in case.edges.iterrows())
    by_id = {t.tx_id: t for t in txs}
    alert = by_id[case.alert_tx_id]
    patterns = []
    for f in findings:
        rows = [by_id[i] for i in f.tx_ids if i in by_id]
        if not rows:
            continue
        dates = sorted(r.date for r in rows)
        patterns.append(
            PatternFact(
                kind=f.kind.value,
                accounts=f.accounts,
                tx_ids=tuple(r.tx_id for r in rows),
                total_usd=round(sum(r.amount_usd for r in rows), 2),
                first_date=dates[0],
                last_date=dates[-1],
                size=float(f.strength),
            )
        )
    return CaseFacts(
        case_id=f"CASE-{case.alert_tx_id}",
        alert=alert,
        subjects=case.focal,
        transactions=txs,
        patterns=tuple(patterns),
        risk_score=risk_score,
    )
