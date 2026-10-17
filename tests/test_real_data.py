"""Smoke checks against the real IBM AMLworld files; skipped when they are absent (CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aml_sentinel.data.loader import patterns_path, read_header, transactions_path
from aml_sentinel.data.patterns import load_patterns
from aml_sentinel.data.schema import RAW_HEADER, DatasetName

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

pytestmark = pytest.mark.real_data


@pytest.mark.parametrize("dataset", [DatasetName.HI_SMALL, DatasetName.LI_SMALL])
def test_real_files_match_schema(dataset: DatasetName) -> None:
    trans = transactions_path(RAW, dataset)
    if not trans.exists():
        pytest.skip(f"{trans.name} not downloaded")
    assert read_header(trans) == RAW_HEADER
    pats = load_patterns(patterns_path(RAW, dataset))
    assert (pats["is_laundering"] == 1).all()
    assert pats["attempt_id"].nunique() > 100
