"""Append-only SQLite store for analyst case actions and SAR draft versions.

Every statement is parameterised; no SQL is ever built from input. Rows are only
inserted: triggers make ``UPDATE`` and ``DELETE`` fail on both tables, so an edited
SAR becomes a new version and a case's history cannot be rewritten through this
code path. (Someone with file access could still edit the database; the hash-chained
audit log is the tamper-evident record of who did what.)

Case notes and SAR text are untrusted free text. They are stored here, never logged,
and the audit log records only their length.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS case_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('approve', 'dismiss', 'escalate')),
    note TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_case_actions_case ON case_actions (case_id, id);
CREATE TABLE IF NOT EXISTS sar_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    issues TEXT NOT NULL,
    notes TEXT NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (case_id, version)
);
CREATE TRIGGER IF NOT EXISTS case_actions_no_update BEFORE UPDATE ON case_actions
BEGIN SELECT RAISE(ABORT, 'case_actions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS case_actions_no_delete BEFORE DELETE ON case_actions
BEGIN SELECT RAISE(ABORT, 'case_actions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS sar_drafts_no_update BEFORE UPDATE ON sar_drafts
BEGIN SELECT RAISE(ABORT, 'sar_drafts is append-only'); END;
CREATE TRIGGER IF NOT EXISTS sar_drafts_no_delete BEFORE DELETE ON sar_drafts
BEGIN SELECT RAISE(ABORT, 'sar_drafts is append-only'); END;
"""


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One analyst decision on a case.

    Attributes:
        case_id: Case identifier.
        actor: User id.
        action: ``approve``, ``dismiss`` or ``escalate``.
        note: Analyst note (untrusted free text).
        created_at: UTC ISO timestamp.
    """

    case_id: str
    actor: str
    action: str
    note: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SarRecord:
    """One stored SAR draft version.

    Attributes:
        case_id: Case identifier.
        version: 1 for the first draft, incremented on every save.
        text: Stamped draft text.
        source: ``template``, an LLM tag, or ``analyst`` for edits.
        issues: Grounding issues as ``(kind, value)`` pairs.
        notes: Processing notes (for example a fallback reason).
        author: User id who generated or edited it.
        created_at: UTC ISO timestamp.
    """

    case_id: str
    version: int
    text: str
    source: str
    issues: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]
    author: str
    created_at: str

    @property
    def is_grounded(self) -> bool:
        """True when no grounding issue was found."""
        return not self.issues


class CaseStore:
    """SQLite-backed case history.

    Args:
        path: Database file (created with its parent directory if missing).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: safe across FastAPI's worker
        # threads without sharing a connection object.
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add_action(self, case_id: str, actor: str, action: str, note: str) -> ActionRecord:
        """Record a decision.

        Args:
            case_id: Case identifier.
            actor: User id.
            action: Decision name.
            note: Analyst note.

        Returns:
            The stored record.
        """
        rec = ActionRecord(case_id, actor, action, note, datetime.now(UTC).isoformat())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO case_actions (case_id, actor, action, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (rec.case_id, rec.actor, rec.action, rec.note, rec.created_at),
            )
        return rec

    def actions(self, case_id: str) -> list[ActionRecord]:
        """Decisions on a case, oldest first.

        Args:
            case_id: Case identifier.

        Returns:
            The records.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, actor, action, note, created_at FROM case_actions "
                "WHERE case_id = ? ORDER BY id",
                (case_id,),
            ).fetchall()
        return [ActionRecord(*row) for row in rows]

    def latest_actions(self) -> dict[str, str]:
        """Most recent decision per case.

        Returns:
            ``{case_id: action}``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, action FROM case_actions WHERE id IN "
                "(SELECT MAX(id) FROM case_actions GROUP BY case_id)"
            ).fetchall()
        return {str(c): str(a) for c, a in rows}

    def add_sar(
        self,
        case_id: str,
        *,
        text: str,
        source: str,
        issues: tuple[tuple[str, str], ...],
        notes: tuple[str, ...],
        author: str,
    ) -> SarRecord:
        """Store a new draft version.

        Args:
            case_id: Case identifier.
            text: Stamped draft text.
            source: Who or what produced the text.
            issues: Grounding issues.
            notes: Processing notes.
            author: User id.

        Returns:
            The stored record with its version number.
        """
        created = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            # IMMEDIATE takes the write lock before reading MAX(version), so two
            # concurrent saves cannot claim the same version.
            conn.execute("BEGIN IMMEDIATE")
            (current,) = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM sar_drafts WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            version = int(current) + 1
            conn.execute(
                "INSERT INTO sar_drafts (case_id, version, text, source, issues, notes, "
                "author, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    version,
                    text,
                    source,
                    json.dumps([list(i) for i in issues]),
                    json.dumps(list(notes)),
                    author,
                    created,
                ),
            )
        return SarRecord(case_id, version, text, source, issues, notes, author, created)

    def latest_sar(self, case_id: str) -> SarRecord | None:
        """Newest draft of a case.

        Args:
            case_id: Case identifier.

        Returns:
            The record, or ``None`` if no draft exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT case_id, version, text, source, issues, notes, author, created_at "
                "FROM sar_drafts WHERE case_id = ? ORDER BY version DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        issues = tuple((str(k), str(v)) for k, v in json.loads(row[4]))
        notes = tuple(str(n) for n in json.loads(row[5]))
        return SarRecord(
            str(row[0]), int(row[1]), str(row[2]), str(row[3]), issues, notes, *row[6:]
        )
