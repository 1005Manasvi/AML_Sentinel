from __future__ import annotations

from pathlib import Path

import pytest

from aml_sentinel.data.fx import add_usd_amounts, estimate_usd_rates
from aml_sentinel.data.loader import load_transactions
from aml_sentinel.data.patterns import ATTEMPT_ID, TYPOLOGY, attach_typology, load_patterns
from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import (
    DataNotFoundError,
    DataValidationError,
    FxEstimationError,
)
from tests.conftest import CYCLE_ROWS, EUR_USD, RawDataset


def test_fx_rates_estimated_from_both_directions(raw_dataset: RawDataset) -> None:
    df = load_transactions(raw_dataset.trans_file)
    train = df[df[Col.TIMESTAMP] < "2022-09-07"]
    rates = estimate_usd_rates(train)
    assert rates == pytest.approx({"US Dollar": 1.0, "Euro": EUR_USD})


def test_fx_fails_with_too_few_samples(raw_dataset: RawDataset) -> None:
    df = load_transactions(raw_dataset.trans_file)
    with pytest.raises(FxEstimationError, match="Euro"):
        estimate_usd_rates(df, min_samples=10_000)


def test_add_usd_amounts(raw_dataset: RawDataset) -> None:
    df = load_transactions(raw_dataset.trans_file)
    out = add_usd_amounts(df, {"US Dollar": 1.0, "Euro": 2.0})
    eur_paid = out[Col.PAYMENT_CURRENCY] == "Euro"
    assert (out.loc[eur_paid, Col.AMOUNT_PAID_USD] == out.loc[eur_paid, Col.AMOUNT_PAID] * 2).all()
    assert Col.AMOUNT_PAID_USD not in df.columns  # input untouched
    with pytest.raises(DataValidationError, match="no FX rate"):
        add_usd_amounts(df, {"US Dollar": 1.0})


def test_patterns_parsed(raw_dataset: RawDataset) -> None:
    pats = load_patterns(raw_dataset.patterns_file)
    assert len(pats) == len(CYCLE_ROWS) + 1
    assert list(pats.groupby(ATTEMPT_ID)[TYPOLOGY].first().astype(str)) == ["CYCLE", "FAN-OUT"]
    assert pats[Col.TIMESTAMP].is_monotonic_increasing


def test_attach_typology(raw_dataset: RawDataset) -> None:
    df = load_transactions(raw_dataset.trans_file)
    out = attach_typology(df, load_patterns(raw_dataset.patterns_file))
    labelled = out[out[TYPOLOGY].notna()]
    assert len(labelled) == 4
    assert (labelled[Col.IS_LAUNDERING] == 1).all()
    assert set(labelled[TYPOLOGY].astype(str)) == {"CYCLE", "FAN-OUT"}
    assert out[ATTEMPT_ID].isna().sum() == len(df) - 4
    assert len(out) == len(df)


ROW = CYCLE_ROWS[0]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("BEGIN LAUNDERING ATTEMPT - CYCLE\nBEGIN LAUNDERING ATTEMPT - CYCLE\n", "nested"),
        (f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW}\nEND LAUNDERING ATTEMPT - STACK\n", "unmatched"),
        (f"{ROW}\n", "outside"),
        (f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW}\n", "unterminated"),
        ("BEGIN LAUNDERING ATTEMPT - TELEPORT\n", "unknown typology"),
        (
            f"BEGIN LAUNDERING ATTEMPT - CYCLE\n{ROW},extra\nEND LAUNDERING ATTEMPT - CYCLE\n",
            "field count",
        ),
    ],
)
def test_malformed_patterns(tmp_path: Path, text: str, message: str) -> None:
    f = tmp_path / "p.txt"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(DataValidationError, match=message):
        load_patterns(f)


def test_patterns_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError):
        load_patterns(tmp_path / "absent.txt")
