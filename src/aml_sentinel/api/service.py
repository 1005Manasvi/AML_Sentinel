"""Case-management logic behind the API, independent of HTTP.

Every state change and every case view is written to the hash-chained audit log
(who, what, when, case id, non-sensitive metadata only). Writes are serialised with
an instance lock: the audit log links each record to the previous one, so two
concurrent appends must not read the same chain head.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Final

import pandas as pd

from aml_sentinel.api.queue import AlertQueue, QueueEntry, build_queue
from aml_sentinel.api.schemas import CaseStatus
from aml_sentinel.api.scoring import TransactionScorer, build_scorer
from aml_sentinel.api.store import ActionRecord, CaseStore, SarRecord
from aml_sentinel.audit import AuditLog, AuditRecord
from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import AmlSentinelError, DataValidationError
from aml_sentinel.models.explain import Explanation
from aml_sentinel.sar.drafting import draft_sar
from aml_sentinel.sar.facts import CaseFacts, build_case_facts
from aml_sentinel.sar.llm import AnthropicClient, LlmClient
from aml_sentinel.sar.review import check_reviewed_text, restamp
from aml_sentinel.system2.cases import Case, CaseIndex
from aml_sentinel.system2.detectors import run_detectors

logger = logging.getLogger(__name__)

MAX_CASE_TRANSACTIONS: Final = 500
_STATUS_OF_ACTION: Final[dict[str, CaseStatus]] = {
    "approve": "sar_approved",
    "dismiss": "dismissed",
    "escalate": "escalated",
}
_AUDIT_ACTION: Final = {
    "approve": "case.approved",
    "dismiss": "case.dismissed",
    "escalate": "case.escalated",
}


class CaseNotFoundError(AmlSentinelError):
    """The case id does not refer to a queued alert."""


class CaseConflictError(AmlSentinelError):
    """The requested action is not allowed in the case's current state."""


class ScoringUnavailableError(AmlSentinelError):
    """New-transaction scoring is disabled for the served dataset."""


@dataclass(frozen=True, slots=True)
class CaseView:
    """A case with its live System 2 analysis.

    Attributes:
        entry: Queue entry.
        rank: 1-based queue rank.
        case: Past-only case subgraph.
        facts: SAR fact sheet (also the data shown to the analyst).
        status: Current status.
        explanation: Why LightGBM scored the alert as it did, or ``None``.
        explanation_unavailable: Why there is no explanation, when there is none.
    """

    entry: QueueEntry
    rank: int
    case: Case
    facts: CaseFacts
    status: CaseStatus
    explanation: Explanation | None = None
    explanation_unavailable: str | None = None


class CaseService:
    """Queue, case detail, SAR drafting and analyst decisions.

    Args:
        queue: Ranked alert queue.
        index: Case index over the served dataset.
        store: Case history store.
        audit: Audit log.
        scorer: New-transaction scorer, or ``None`` if disabled.
        scoring_disabled_reason: Why ``scorer`` is ``None``.
        llm_client: External SAR drafter; ``None`` keeps drafting local.
    """

    def __init__(
        self,
        *,
        queue: AlertQueue,
        index: CaseIndex,
        store: CaseStore,
        audit: AuditLog,
        scorer: TransactionScorer | None = None,
        scoring_disabled_reason: str | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        self.queue = queue
        self._index = index
        self._store = store
        self._audit = audit
        self._scorer = scorer
        self._scoring_reason = scoring_disabled_reason or "scoring is not configured"
        self._llm = llm_client
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ queue
    def status_of(self, entry: QueueEntry, latest: dict[str, str]) -> CaseStatus:
        """Current case status.

        Args:
            entry: Queue entry.
            latest: Output of :meth:`CaseStore.latest_actions`.

        Returns:
            The latest analyst decision, else ``open`` or ``auto_closed``.
        """
        action = latest.get(entry.case_id)
        if action is not None:
            return _STATUS_OF_ACTION[action]
        return "open" if entry.escalated else "auto_closed"

    def alerts(self) -> list[tuple[int, QueueEntry, CaseStatus]]:
        """Every alert with its rank and status.

        Returns:
            ``(rank, entry, status)`` in rank order.
        """
        latest = self._store.latest_actions()
        return [(r, e, self.status_of(e, latest)) for r, e in enumerate(self.queue, start=1)]

    # ------------------------------------------------------------------ cases
    def _entry(self, case_id: str) -> tuple[int, QueueEntry]:
        entry = self.queue.get(case_id)
        rank = self.queue.rank_of(case_id)
        if entry is None or rank is None:
            raise CaseNotFoundError("case not found")
        return rank, entry

    def _view(self, case_id: str) -> CaseView:
        rank, entry = self._entry(case_id)
        case = self._index.build(entry.tx_id)
        facts = build_case_facts(case, run_detectors(case), risk_score=entry.p_triage)
        status = self.status_of(entry, self._store.latest_actions())
        explanation, unavailable = self._explain(entry.tx_id)
        return CaseView(entry, rank, case, facts, status, explanation, unavailable)

    def _explain(self, tx_id: int) -> tuple[Explanation | None, str | None]:
        # Explanations reuse the scorer's past-only feature path, so they share its
        # history-size cap: without an online feature store, large datasets get none.
        if self._scorer is None:
            return None, self._scoring_reason
        try:
            return self._scorer.explain(tx_id), None
        except DataValidationError as exc:
            return None, str(exc)

    def case(self, case_id: str, actor: str) -> CaseView:
        """Load a case and record that ``actor`` viewed it.

        Args:
            case_id: Case identifier.
            actor: User id.

        Returns:
            The case view.

        Raises:
            CaseNotFoundError: If the case is not queued.
        """
        view = self._view(case_id)
        with self._lock:
            self._audit.append(
                actor=actor,
                action="case.viewed",
                case_id=case_id,
                details={"n_findings": len(view.facts.patterns), "status": view.status},
            )
        return view

    def actions(self, case_id: str) -> list[ActionRecord]:
        """Decision history of a case.

        Args:
            case_id: Case identifier.

        Returns:
            Records, oldest first.
        """
        return self._store.actions(case_id)

    def latest_sar(self, case_id: str) -> SarRecord | None:
        """Newest SAR draft of a case.

        Args:
            case_id: Case identifier.

        Returns:
            The record or ``None``.
        """
        return self._store.latest_sar(case_id)

    # ------------------------------------------------------------------ SAR
    def generate_sar(self, case_id: str, actor: str) -> SarRecord:
        """Draft a new SAR version from the case facts.

        Args:
            case_id: Case identifier.
            actor: User id.

        Returns:
            The stored draft.

        Raises:
            CaseNotFoundError: If the case is not queued.
        """
        view = self._view(case_id)
        draft = draft_sar(view.facts, self._llm)
        issues = tuple((i.kind, i.value) for i in draft.grounding_issues)
        with self._lock:
            record = self._store.add_sar(
                case_id,
                text=draft.text,
                source=draft.source,
                issues=issues,
                notes=draft.notes,
                author=actor,
            )
            self._audit.append(
                actor=actor,
                action="sar.generated",
                case_id=case_id,
                details={
                    "source": draft.source[:64],
                    "grounded": draft.is_grounded,
                    "n_issues": len(issues),
                },
            )
        return record

    def save_sar(self, case_id: str, text: str, actor: str) -> SarRecord:
        """Store an analyst-edited draft as a new version.

        The text is re-stamped and grounding-checked; issues are recorded, not hidden.

        Args:
            case_id: Case identifier.
            text: Edited text.
            actor: User id.

        Returns:
            The stored draft.

        Raises:
            CaseNotFoundError: If the case is not queued.
            CaseConflictError: If there is no generated draft to edit.
        """
        view = self._view(case_id)
        if self._store.latest_sar(case_id) is None:
            raise CaseConflictError("generate a draft before editing it")
        stamped = restamp(text)
        issues = tuple((i.kind, i.value) for i in check_reviewed_text(stamped, view.facts))
        with self._lock:
            record = self._store.add_sar(
                case_id, text=stamped, source="analyst", issues=issues, notes=(), author=actor
            )
            self._audit.append(
                actor=actor,
                action="sar.edited",
                case_id=case_id,
                details={"grounded": not issues, "n_issues": len(issues)},
            )
        return record

    # ------------------------------------------------------------------ actions
    def act(self, case_id: str, action: str, note: str, actor: str) -> ActionRecord:
        """Record an analyst decision.

        Approval needs a grounded draft: approving a SAR whose amounts, dates or
        accounts could not be verified is refused.

        Args:
            case_id: Case identifier.
            action: ``approve``, ``dismiss`` or ``escalate``.
            note: Analyst note (stored, never logged).
            actor: User id.

        Returns:
            The stored decision.

        Raises:
            CaseNotFoundError: If the case is not queued.
            CaseConflictError: If approving without a grounded draft.
        """
        self._entry(case_id)
        if action == "approve":
            sar = self._store.latest_sar(case_id)
            if sar is None:
                raise CaseConflictError("approve needs a SAR draft")
            if not sar.is_grounded:
                raise CaseConflictError("the latest SAR draft has grounding issues; fix them first")
        with self._lock:
            record = self._store.add_action(case_id, actor, action, note)
            self._audit.append(
                actor=actor,
                action=_AUDIT_ACTION[action],
                case_id=case_id,
                details={"decision": action, "note_length": len(note)},
            )
        return record

    # ------------------------------------------------------------------ scoring
    @property
    def scorer(self) -> TransactionScorer:
        """The new-transaction scorer.

        Raises:
            ScoringUnavailableError: If scoring is disabled for this dataset.
        """
        if self._scorer is None:
            raise ScoringUnavailableError(self._scoring_reason)
        return self._scorer

    # ------------------------------------------------------------------ audit
    def audit_records(self) -> list[AuditRecord]:
        """Every audit record, oldest first.

        Returns:
            The records.
        """
        return self._audit.records()

    def verify_audit(self) -> int:
        """Verify the full audit chain.

        Returns:
            Number of verified records.

        Raises:
            IntegrityError: At the first broken record.
        """
        return self._audit.verify()


def aggregate_edges(case: Case) -> pd.DataFrame:
    """Payments per ordered account pair, for the graph view.

    Args:
        case: Case subgraph.

    Returns:
        ``payer``, ``payee``, ``count``, ``total_usd`` sorted by total descending.
    """
    edges = case.edges
    grouped = (
        edges.groupby([Col.FROM_ID.value, Col.TO_ID.value], observed=True)[Col.AMOUNT_PAID_USD]
        .agg(["count", "sum"])
        .reset_index()
    )
    grouped.columns = pd.Index(["payer", "payee", "count", "total_usd"])
    return grouped.sort_values("total_usd", ascending=False, kind="stable").reset_index(drop=True)


def pattern_names(entry: QueueEntry) -> list[str]:
    """Detectors that fired for an escalated alert.

    Args:
        entry: Queue entry.

    Returns:
        Detector names, empty when none fired or the alert was not escalated.
    """
    if entry.evidence is None:
        return []
    return sorted(
        k.removeprefix("n_") for k, v in entry.evidence.items() if v > 0 and k != "n_findings"
    )


def build_service(settings: Settings, *, llm_client: LlmClient | None = None) -> CaseService:
    """Load the served dataset and pipeline outputs into a :class:`CaseService`.

    Args:
        settings: Settings (``API_DATASET`` selects the dataset).
        llm_client: External drafter override; by default built from settings.

    Returns:
        The service.
    """
    dataset: DatasetName = settings.api_dataset
    frame = load_processed(settings.processed_dir, dataset)
    index = CaseIndex(frame)
    queue = build_queue(settings, dataset, frame, index)
    scorer, reason = build_scorer(settings, dataset, frame)
    if llm_client is None and settings.llm_provider == "external":
        llm_client = AnthropicClient.from_settings(settings)
    return CaseService(
        queue=queue,
        index=index,
        store=CaseStore(settings.cases_db),
        audit=AuditLog(settings.audit_path),
        scorer=scorer,
        scoring_disabled_reason=reason,
        llm_client=llm_client,
    )
