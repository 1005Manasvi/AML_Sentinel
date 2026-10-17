"""Guards the .gitignore against two real mistakes made during development.

1. An unanchored ``models/`` pattern silently ignored ``src/aml_sentinel/models``.
2. A trailing ``# comment`` on a pattern line made it match nothing, exposing
   trained model files.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - fixed git invocation, no user input
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    GIT is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def _ignored(paths: list[str]) -> set[str]:
    assert GIT is not None
    result = subprocess.run(  # noqa: S603 - fixed argv, paths are repo-relative literals
        [GIT, "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return set(result.stdout.split())


def test_no_source_or_test_file_is_ignored() -> None:
    files = [
        p.relative_to(ROOT).as_posix()
        for d in ("src", "tests")
        for p in (ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    assert files
    assert _ignored(files) == set()


@pytest.mark.parametrize(
    "path",
    [
        "models/HI-Small/lightgbm_all.txt",
        "data/raw/HI-Small_Trans.csv",
        "data/processed/HI-Small.parquet",
        ".env",
        "weights.pkl",
        "data/audit/audit.jsonl",
    ],
)
def test_artifacts_and_secrets_are_ignored(path: str) -> None:
    assert path in _ignored([path])


def test_bundled_sample_is_not_ignored() -> None:
    assert _ignored(["data/sample/Sample_Trans.csv"]) == set()


def test_gitignore_has_no_trailing_comments() -> None:
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert " #" not in stripped, f"trailing comment breaks pattern: {line!r}"
