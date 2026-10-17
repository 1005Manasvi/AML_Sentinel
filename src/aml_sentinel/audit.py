"""Append-only, hash-chained audit log of analyst actions and SAR generations.

Each JSON line records **who, what, when and which case**, plus ``prev_hash`` and
its own ``hash``. The hash is SHA-256 over the canonical record including
``prev_hash``. Rewriting, deleting or reordering any earlier line breaks the chain,
and :meth:`AuditLog.verify` reports the first broken line.

Security:
    * The file is only ever opened in append mode by this class.
    * ``details`` is restricted to an allowlist of non-sensitive keys holding short
      scalar values. Account ids, amounts, names and narrative text cannot be
      logged here even by mistake.
    * Tamper-*evidence*, not tamper-*proofing*: someone with write access could
      rebuild the whole chain. Production would anchor the head hash externally
      (WORM storage or a signed timestamp). This is recorded as a known limitation
      in SECURITY.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from aml_sentinel.exceptions import DataValidationError, IntegrityError

GENESIS: Final = "0" * 64
ALLOWED_DETAIL_KEYS: Final = frozenset(
    {
        "source",
        "grounded",
        "n_issues",
        "decision",
        "note_length",
        "status",
        "model",
        "role",
        "reason_code",
        "n_findings",
        "risk_bucket",
        "duration_ms",
    }
)
_ACTOR_RE: Final = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
_ACTION_RE: Final = re.compile(r"^[a-z][a-z0-9_.]{1,48}$")
_CASE_RE: Final = re.compile(r"^CASE-\d{1,12}$")


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One audit entry.

    Attributes:
        ts: UTC ISO timestamp.
        actor: Authenticated user or service id.
        action: Dotted action name, e.g. ``sar.generated`` or ``case.dismissed``.
        case_id: ``CASE-<n>``.
        details: Allowlisted, non-sensitive metadata.
        prev_hash: Hash of the previous record (genesis for the first).
        hash: Hash of this record.
    """

    ts: str
    actor: str
    action: str
    case_id: str
    details: dict[str, Any]
    prev_hash: str
    hash: str


def _digest(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_details(details: dict[str, Any]) -> dict[str, Any]:
    bad = sorted(set(details) - ALLOWED_DETAIL_KEYS)
    if bad:
        raise DataValidationError(f"audit details contain non-allowlisted keys: {bad}")
    for key, value in details.items():
        if isinstance(value, str) and len(value) > 64:
            raise DataValidationError(f"audit detail {key} is too long")
        if not isinstance(value, str | int | float | bool) and value is not None:
            raise DataValidationError(f"audit detail {key} must be a scalar")
    return dict(details)


class AuditLog:
    """Hash-chained JSONL audit log.

    Args:
        path: Log file (created on first write).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _last_hash(self) -> str:
        if not self.path.is_file():
            return GENESIS
        last = GENESIS
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = str(json.loads(line)["hash"])
        return last

    def append(
        self, *, actor: str, action: str, case_id: str, details: dict[str, Any] | None = None
    ) -> AuditRecord:
        """Append one record.

        Args:
            actor: Who acted (validated identifier).
            action: What happened (validated dotted name).
            case_id: Which case.
            details: Allowlisted non-sensitive metadata.

        Returns:
            The written record.

        Raises:
            DataValidationError: On invalid identifiers or non-allowlisted details.
        """
        if (
            not _ACTOR_RE.match(actor)
            or not _ACTION_RE.match(action)
            or not _CASE_RE.match(case_id)
        ):
            raise DataValidationError("invalid audit actor, action or case id")
        ts, clean, prev = (
            datetime.now(UTC).isoformat(),
            _check_details(details or {}),
            self._last_hash(),
        )
        body: dict[str, Any] = {
            "ts": ts,
            "actor": actor,
            "action": action,
            "case_id": case_id,
            "details": clean,
            "prev_hash": prev,
        }
        record = AuditRecord(
            ts=ts,
            actor=actor,
            action=action,
            case_id=case_id,
            details=clean,
            prev_hash=prev,
            hash=_digest(body),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({**body, "hash": record.hash}, sort_keys=True) + "\n")
        return record

    def records(self) -> list[AuditRecord]:
        """Read all records (oldest first).

        Returns:
            The records.
        """
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [AuditRecord(**json.loads(line)) for line in fh if line.strip()]

    def verify(self) -> int:
        """Check the whole hash chain.

        Returns:
            Number of verified records.

        Raises:
            IntegrityError: At the first line whose hash or link does not match.
        """
        prev = GENESIS
        for n, rec in enumerate(self.records(), start=1):
            body = {
                "ts": rec.ts,
                "actor": rec.actor,
                "action": rec.action,
                "case_id": rec.case_id,
                "details": rec.details,
                "prev_hash": rec.prev_hash,
            }
            if rec.prev_hash != prev or _digest(body) != rec.hash:
                raise IntegrityError(f"audit log chain broken at record {n}")
            prev = rec.hash
        return len(self.records())
