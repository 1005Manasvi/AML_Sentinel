"""Structured logging setup.

Logs are emitted as one JSON object per line so they can be shipped to a log store
and queried. Context goes in ``extra={...}``; the formatter copies those fields into
the JSON document.

Security note:
    Never pass sensitive values (account identifiers, amounts, names, free text,
    secrets) as log fields. Log counts, durations, case IDs and file names instead.
    As a backstop, fields whose names look secret-bearing are masked.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

# Attributes present on every LogRecord; anything else came from ``extra=``.
_STANDARD_ATTRS: Final = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys() | {"message", "asctime"}
)
_SECRET_HINTS: Final = ("key", "secret", "token", "password", "authorization")
_MASK: Final = "***"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON.

        Args:
            record: The log record.

        Returns:
            A single-line JSON string.
        """
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for name, value in vars(record).items():
            if name in _STANDARD_ATTRS or name.startswith("_"):
                continue
            doc[name] = _MASK if _looks_secret(name) else value
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)


def _looks_secret(field_name: str) -> bool:
    lowered = field_name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger once at program start.

    Args:
        level: Minimum level name, e.g. ``"INFO"``.
        json_output: Use :class:`JsonFormatter` when true, plain text otherwise.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
