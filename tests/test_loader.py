from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from aml_sentinel.data.loader import (
    load_transactions,
    patterns_path,
    read_header,
    transactions_path,
)
from aml_sentinel.data.schema import Col, DatasetName, Transaction, account_node_id
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from tests.conftest import HEADER, RawDataset, raw_row


def _write(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    f = tmp_path / "t.csv"
    f.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return f


GOOD = raw_row(
    "2022/09/01 00:20",
    "010",
    "8000EBD30",
    "03208",
    "8000F4580",
    1.5,
    "US Dollar",
    1.5,
    "US Dollar",
    "Cheque",
    0,
)


def test_load_types_ids_and_order(raw_dataset: RawDataset) -> None:
    df = load_transactions(raw_dataset.trans_file)
    assert df[Col.TIMESTAMP].is_monotonic_increasing
    assert pd.api.types.is_datetime64_dtype(df[Col.TIMESTAMP])
    assert df[Col.IS_LAUNDERING].dtype == "int8"
    # tx_id is the source-file row number: the out-of-order first row is now last.
    assert df[Col.TX_ID].iloc[-1] == 0
    assert df[Col.TX_ID].is_unique
    assert int(df[Col.IS_LAUNDERING].sum()) == 4


def test_leading_zeros_preserved(tmp_path: Path) -> None:
    df = load_transactions(_write(tmp_path, GOOD))
    assert df[Col.TO_BANK].iloc[0] == "03208"
    assert df[Col.FROM_ID].iloc[0] == account_node_id("010", "8000EBD30")
    assert df[Col.TO_ID].iloc[0] == "03208:8000F4580"


def test_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError):
        load_transactions(tmp_path / "absent.csv")
    empty = tmp_path / "e.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        read_header(empty)


def test_wrong_header_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="header"):
        load_transactions(_write(tmp_path, GOOD, header=HEADER.replace("Amount Paid", "Paid")))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (GOOD.replace(",0", ",2")[: -0 or None], "is_laundering"),
        (GOOD.replace("1.50,US Dollar,1.50", "-1.50,US Dollar,1.50"), "negative"),
        (GOOD.replace("US Dollar,Cheque", "Doubloon,Cheque"), "unknown"),
        (GOOD.replace("Cheque", "Carrier Pigeon"), "unknown"),
        (GOOD.replace("2022/09/01 00:20", "01-09-2022"), "unparseable"),
        (GOOD.replace("8000EBD30", ""), "null"),
    ],
)
def test_bad_values_rejected(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_transactions(_write(tmp_path, row))


def test_paths() -> None:
    raw = Path("data/raw")
    assert transactions_path(raw, DatasetName.LI_SMALL).name == "LI-Small_Trans.csv"
    assert patterns_path(raw, DatasetName.HI_SMALL).name == "HI-Small_Patterns.txt"


def test_unsorted_frame_rejected(raw_dataset: RawDataset) -> None:
    from aml_sentinel.data.loader import validate_transactions

    df = load_transactions(raw_dataset.trans_file).iloc[::-1]
    with pytest.raises(DataValidationError, match="sorted"):
        validate_transactions(df)
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_transactions(pd.DataFrame({"x": [1]}))


class TestTransactionModel:
    BASE = {
        "timestamp": datetime(2022, 9, 1),
        "from_bank": "010",
        "from_account": "8000EBD30",
        "to_bank": "020",
        "to_account": "80011F990",
        "amount_received": 10.0,
        "receiving_currency": "Euro",
        "amount_paid": 10.0,
        "payment_currency": "Euro",
        "payment_format": "ACH",
    }

    def test_valid(self) -> None:
        assert Transaction.model_validate(self.BASE).from_bank == "010"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("from_account", "<script>alert(1)</script>"),
            ("to_account", "ignore previous instructions"),
            ("from_bank", "01A"),
            ("amount_paid", -1.0),
            ("amount_paid", float("inf")),
            ("payment_currency", "Doubloon"),
            ("extra_field", "x"),
        ],
    )
    def test_untrusted_input_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            Transaction.model_validate({**self.BASE, field: value})
