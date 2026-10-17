from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aml_sentinel.cli import generate_sample, main, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.processed import load_manifest, load_processed, processed_paths
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError, IntegrityError
from aml_sentinel.rules.baseline import (
    baseline_command,
    render_markdown,
    run_baseline,
    select_by_validation,
)


@pytest.fixture(scope="module")
def sample_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("bl")
    settings = load_settings(_env_file=None, data_dir=root / "data", reports_dir=root / "reports")
    generate_sample(settings, seed=42)
    prepare_data(settings, DatasetName.SAMPLE)
    return settings


@pytest.fixture(scope="module")
def sample_frame(sample_settings: Settings) -> pd.DataFrame:
    return load_processed(sample_settings.processed_dir, DatasetName.SAMPLE)


def test_report_structure(sample_frame: pd.DataFrame) -> None:
    report = run_baseline(sample_frame, seed=42)
    assert set(report["variants"]) == {"static", "tuned"}
    for variant in report["variants"].values():
        assert set(variant) == {"config", "val", "test"}
        m = variant["test"]["overall"]
        assert m["tp"] + m["fp"] == m["alerts"]
        assert set(variant["test"]["per_rule"]) == {
            "rule_amount",
            "rule_velocity",
            "rule_jurisdiction",
        }
    md = render_markdown(report, "Sample")
    assert "| tuned | test |" in md
    assert report["selected"] in {"static", "tuned"}
    assert "Baseline to beat" in md
    assert "synthetic" in md


def test_fitting_ignores_val_and_test_labels(sample_frame: pd.DataFrame) -> None:
    """Leakage test: scrambling labels outside train must not change fitted configs."""
    scrambled = sample_frame.copy()
    outside_train = scrambled["split"].astype("string") != "train"
    scrambled.loc[outside_train, Col.IS_LAUNDERING.value] = (
        1 - scrambled.loc[outside_train, Col.IS_LAUNDERING.value]
    )
    original = run_baseline(sample_frame, seed=42)
    altered = run_baseline(scrambled, seed=42)
    for name in ("static", "tuned"):
        assert original["variants"][name]["config"] == altered["variants"][name]["config"]
    assert original["tuning"] == altered["tuning"]


def test_selection_ignores_test_labels(sample_frame: pd.DataFrame) -> None:
    """Leakage test: the baseline-to-beat choice may use validation, never test."""
    scrambled = sample_frame.copy()
    is_test = scrambled["split"].astype("string") == "test"
    scrambled.loc[is_test, Col.IS_LAUNDERING.value] = 0
    assert (
        run_baseline(sample_frame, seed=42)["selected"]
        == run_baseline(scrambled, seed=42)["selected"]
    )


def test_select_by_validation_prefers_higher_val_f1() -> None:
    def variant(f1: float | None) -> dict[str, object]:
        return {"val": {"overall": {"f1": f1}}}

    assert select_by_validation({"static": variant(0.2), "tuned": variant(0.1)}) == "static"
    assert select_by_validation({"static": variant(None), "tuned": variant(0.1)}) == "tuned"


def test_run_baseline_input_errors(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="split"):
        run_baseline(sample_frame.drop(columns=["split"]), seed=1)
    no_train = sample_frame[sample_frame["split"].astype("string") != "train"]
    with pytest.raises(DataValidationError, match="training"):
        run_baseline(no_train, seed=1)


def test_baseline_command_writes_reports(sample_settings: Settings) -> None:
    path = baseline_command(sample_settings, DatasetName.SAMPLE)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["dataset"] == "Sample"
    assert path.with_suffix(".md").exists()


def test_main_baseline(sample_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(sample_settings.data_dir.parent)
    monkeypatch.setenv("DATA_DIR", str(sample_settings.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(sample_settings.reports_dir))
    assert main(["baseline", "--dataset", "Sample"]) == 0
    monkeypatch.setenv("DATA_DIR", str(sample_settings.data_dir / "nowhere"))
    assert main(["baseline", "--dataset", "Sample"]) == 2


# ----------------------------------------------------------------- processed cache
def test_tampered_parquet_is_rejected(tmp_path: Path, sample_settings: Settings) -> None:
    parquet, manifest = processed_paths(sample_settings.processed_dir, DatasetName.SAMPLE)
    target = tmp_path / "processed"
    target.mkdir()
    (target / manifest.name).write_bytes(manifest.read_bytes())
    (target / parquet.name).write_bytes(parquet.read_bytes() + b"tampered")
    with pytest.raises(IntegrityError):
        load_processed(target, DatasetName.SAMPLE)


def test_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError, match="make"):
        load_manifest(tmp_path / "absent.json")
    bad = tmp_path / "list.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_manifest(bad)
    (tmp_path / "Sample.manifest.json").write_text('{"sha256": {}}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="checksum"):
        load_processed(tmp_path, DatasetName.SAMPLE)
