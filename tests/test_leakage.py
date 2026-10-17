"""Leakage tests: no information from the future may reach training or calibration.

These tests are the executable form of CLAUDE.md section 3 ("Write a test proving no
future data leaks into training or fusion calibration"). Later phases add their own
fitted components here (feature builders, LightGBM, calibrators).
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from aml_sentinel.cli import prepare_data
from aml_sentinel.data.loader import load_transactions
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.data.splits import (
    SplitName,
    TimeSplit,
    assert_no_future_rows,
    history_before,
)
from aml_sentinel.exceptions import ConfigurationError, LeakageError
from tests.conftest import EUR_USD, RawDataset

SPLIT = TimeSplit(datetime(2022, 9, 7), datetime(2022, 9, 9), datetime(2022, 9, 11))


def _frame(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame({Col.TIMESTAMP.value: pd.to_datetime(timestamps)})


def test_partitions_are_ordered_and_disjoint(raw_dataset: RawDataset) -> None:
    parts = SPLIT.partition(load_transactions(raw_dataset.trans_file))
    train, val, test, tail = (parts[s] for s in SplitName)
    assert train[Col.TIMESTAMP].max() < val[Col.TIMESTAMP].min()
    assert val[Col.TIMESTAMP].max() < test[Col.TIMESTAMP].min()
    assert test[Col.TIMESTAMP].max() < tail[Col.TIMESTAMP].min()
    ids = [set(p[Col.TX_ID]) for p in parts.values()]
    assert sum(len(i) for i in ids) == len(set().union(*ids))


def test_boundaries_are_half_open() -> None:
    names = SPLIT.name_of(
        _frame(["2022-09-06 23:59", "2022-09-07 00:00", "2022-09-09 00:00", "2022-09-11 00:00"])[
            Col.TIMESTAMP
        ]
    )
    assert list(names.astype(str)) == ["train", "val", "test", "tail"]


def test_split_rejects_bad_order() -> None:
    with pytest.raises(ConfigurationError):
        TimeSplit(datetime(2022, 9, 9), datetime(2022, 9, 7), datetime(2022, 9, 11))


def test_history_before_excludes_cutoff() -> None:
    df = _frame(["2022-09-06 23:59", "2022-09-07 00:00"])
    assert len(history_before(df, datetime(2022, 9, 7))) == 1


def test_assert_no_future_rows() -> None:
    df = _frame(["2022-09-06", "2022-09-08"])
    assert_no_future_rows(df, datetime(2022, 9, 9), what="ok")
    with pytest.raises(LeakageError, match="1 rows"):
        assert_no_future_rows(df, datetime(2022, 9, 7), what="fit data")


def test_fx_rates_ignore_poisoned_future_rows(raw_dataset: RawDataset) -> None:
    """The fixture plants EUR at 9000 USD in the test period; train says 1.10."""
    prepare_data(raw_dataset.settings, DatasetName.HI_SMALL)
    manifest_file = raw_dataset.settings.processed_dir / "HI-Small.manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["fx_usd_rates"]["Euro"] == pytest.approx(EUR_USD)
    assert manifest["fx_usd_rates_fit_on"] == "train"
