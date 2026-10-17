from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aml_sentinel.cli import main, prepare_data
from aml_sentinel.data.loader import transactions_path
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.integrity import verify_checksum
from tests.conftest import RawDataset


def test_prepare_data_writes_parquet_and_manifest(raw_dataset: RawDataset) -> None:
    out = prepare_data(raw_dataset.settings, DatasetName.HI_SMALL)
    df = pd.read_parquet(out)
    assert {Col.AMOUNT_PAID_USD.value, "typology", "split"} <= set(df.columns)
    manifest = json.loads(out.with_name("HI-Small.manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == len(df)
    assert manifest["positives"] == 4
    assert manifest["rows_with_typology"] == 4
    assert manifest["split_rows"]["tail"] == 1
    verify_checksum(out, manifest["sha256"][out.name])


def test_main_success(raw_dataset: RawDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(raw_dataset.settings.data_dir))
    monkeypatch.chdir(raw_dataset.settings.data_dir.parent)  # no stray .env
    assert main(["prepare-data", "--dataset", "HI-Small"]) == 0


def test_main_reports_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["prepare-data"]) == 2


def test_main_reports_bad_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JEV_ENABLED", "true")
    assert main(["prepare-data"]) == 2
    assert not transactions_path(tmp_path, DatasetName.HI_SMALL).exists()
