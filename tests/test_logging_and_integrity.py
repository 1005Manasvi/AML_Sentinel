from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from aml_sentinel.exceptions import DataNotFoundError, IntegrityError
from aml_sentinel.integrity import sha256_file, verify_checksum
from aml_sentinel.logging_utils import JsonFormatter, configure_logging


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_includes_extra_fields() -> None:
    doc = json.loads(JsonFormatter().format(_record(rows=5)))
    assert doc["msg"] == "hello x"
    assert doc["level"] == "INFO"
    assert doc["rows"] == 5


@pytest.mark.parametrize("field", ["api_key", "jev_secret", "auth_token", "Authorization"])
def test_json_formatter_masks_secret_like_fields(field: str) -> None:
    doc = json.loads(JsonFormatter().format(_record(**{field: "hunter2"})))
    assert doc[field] == "***"


def test_json_formatter_renders_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "x", None, sys.exc_info())
    assert "ValueError" in json.loads(JsonFormatter().format(record))["exc"]


@pytest.mark.parametrize("json_output", [True, False])
def test_configure_logging_replaces_handlers(json_output: bool) -> None:
    configure_logging("WARNING", json_output=json_output)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING
    assert isinstance(root.handlers[0].formatter, JsonFormatter) is json_output


def test_checksum_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "model.bin"
    f.write_bytes(b"weights")
    expected = hashlib.sha256(b"weights").hexdigest()
    assert sha256_file(f) == expected
    verify_checksum(f, expected.upper())


def test_checksum_mismatch_and_missing(tmp_path: Path) -> None:
    f = tmp_path / "model.bin"
    f.write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        verify_checksum(f, "0" * 64)
    with pytest.raises(DataNotFoundError):
        sha256_file(tmp_path / "absent.bin")
