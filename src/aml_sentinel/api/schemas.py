"""Request and response models of the API.

Requests forbid unknown fields and bound every string. Responses never contain
ground-truth labels: an analyst in production would not have them, and showing them
would make the demo misleading.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aml_sentinel.data.schema import Transaction

CASE_ID_PATTERN: Final = r"^CASE-\d{1,12}$"
MAX_NOTE: Final = 2_000
MAX_SAR_TEXT: Final = 20_000

CaseStatus = Literal["open", "auto_closed", "sar_approved", "dismissed", "escalated"]
ActionName = Literal["approve", "dismiss", "escalate"]


class _Out(BaseModel):
    model_config = ConfigDict(frozen=True)


class Health(_Out):
    """Liveness response (the only unauthenticated endpoint)."""

    status: Literal["ok"] = "ok"


class Me(_Out):
    """The authenticated caller."""

    user: str
    role: str


class ScoreRequest(Transaction):
    """A new, unlabeled transaction to score.

    The timestamp must be naive (dataset-local time), like the training data. A
    timezone-aware value is rejected rather than silently converted.
    """

    @field_validator("timestamp")
    @classmethod
    def _naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("timestamp must not carry a timezone (dataset-local time)")
        return value


class ScoreResponse(_Out):
    """System 1a score for one transaction. Nothing is stored."""

    p_tabular: float = Field(description="Validation-calibrated LightGBM probability")
    model: str
    alert: bool = Field(description="Reaches the System 1a alert-queue threshold")
    alert_threshold: float
    rules_fired: list[str] | None
    history_rows: int
    latency_ms: float
    persisted: Literal[False] = False


class AlertItem(_Out):
    """One row of the ranked queue."""

    rank: int
    case_id: str
    tx_id: int
    timestamp: datetime
    p_triage: float
    escalated: bool
    status: CaseStatus
    n_findings: int | None = Field(description="System 2 findings (escalated alerts only)")
    patterns: list[str] = Field(description="Detectors that fired")


class AlertList(_Out):
    """A page of the ranked queue plus score provenance."""

    dataset: str
    ranking: Literal["p_triage"] = "p_triage"
    triage_source: str
    escalation_threshold: float
    escalate_share: float
    total: int
    items: list[AlertItem]


class TxOut(_Out):
    """A payment in a case."""

    tx_id: int
    date: str
    payer: str
    payee: str
    amount_usd: float
    payment_format: str


class FindingOut(_Out):
    """A System 2 detector finding."""

    kind: str
    accounts: list[str]
    tx_ids: list[int]
    total_usd: float
    first_date: str
    last_date: str
    size: float


class EdgeOut(_Out):
    """Payments between one ordered account pair, aggregated for the graph view."""

    payer: str
    payee: str
    count: int
    total_usd: float


class ActionOut(_Out):
    """A recorded analyst decision."""

    actor: str
    action: str
    note: str
    created_at: str


class GroundingIssueOut(_Out):
    """A reference in a SAR draft that the case facts do not support."""

    kind: str
    value: str


class SarOut(_Out):
    """A stored SAR draft version."""

    case_id: str
    version: int
    text: str
    source: str
    grounded: bool
    issues: list[GroundingIssueOut]
    notes: list[str]
    author: str
    created_at: str


class ReliabilityOut(_Out):
    """Observed laundering rate of test alerts in the same ``p_triage`` band."""

    lo: float
    hi: float
    alerts: int
    observed_rate: float
    measured_on: Literal["held-out test period (reports/evaluation)"] = (
        "held-out test period (reports/evaluation)"
    )


class CaseDetail(_Out):
    """Everything an analyst sees for one case."""

    case_id: str
    rank: int | None
    alert: TxOut
    subjects: list[str]
    p_triage: float | None
    triage_source: str
    p_final: float | None = Field(
        description="Final combined score; informational, not used for ranking (ADR-020)"
    )
    reliability: ReliabilityOut | None = Field(
        description="How often alerts with a similar p_triage were laundering on test"
    )
    escalated: bool
    status: CaseStatus
    case_rows: int
    truncated: bool
    findings: list[FindingOut]
    transactions: list[TxOut] = Field(description="Most recent case payments (capped)")
    edges: list[EdgeOut]
    actions: list[ActionOut]
    sar: SarOut | None


class SarEdit(BaseModel):
    """An analyst's edited SAR text."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_SAR_TEXT)


class ActionRequest(BaseModel):
    """An analyst decision on a case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: ActionName
    note: str = Field(min_length=3, max_length=MAX_NOTE)


class AuditRecordOut(_Out):
    """One audit-log entry."""

    ts: str
    actor: str
    action: str
    case_id: str
    details: dict[str, str | int | float | bool | None]
    hash: str


class AuditView(_Out):
    """Audit log page plus a full chain verification."""

    chain_ok: bool
    chain_records: int
    chain_error: str | None
    records: list[AuditRecordOut]
