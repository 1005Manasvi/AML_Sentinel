"""File integrity helpers (SHA-256 checksums).

Used to verify raw data files and, from Phase 3 on, every model artifact before it is
loaded.

Security note:
    Model files are only loaded from trusted local paths *after* their checksum
    matches a value recorded at training time. Comparison is constant-time.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from aml_sentinel.exceptions import DataNotFoundError, IntegrityError

_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """Compute the hex SHA-256 digest of a file, streaming it in chunks.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.

    Raises:
        DataNotFoundError: If ``path`` is not an existing file.
    """
    if not path.is_file():
        raise DataNotFoundError(f"file not found: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> None:
    """Raise unless ``path`` hashes to ``expected_sha256``.

    Args:
        path: File to verify.
        expected_sha256: Expected hex digest (case-insensitive).

    Raises:
        DataNotFoundError: If the file does not exist.
        IntegrityError: If the digest does not match.
    """
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected_sha256.strip().lower()):
        raise IntegrityError(f"checksum mismatch for {path.name}")
