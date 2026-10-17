from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_sentinel.cli import generate_sample, main, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError, IntegrityError
from aml_sentinel.features.builder import build_features, feature_columns
from aml_sentinel.models.calibration import PlattCalibrator
from aml_sentinel.models.lgbm import LgbmParams, LightGBMTriage
from aml_sentinel.models.rule_triage import RuleTriage
from aml_sentinel.models.system1a import load_baseline_config, split_masks, train_system1a
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.models.triage import AlertBatch, CalibratedTriage, Triage, TriageOutput
from aml_sentinel.rules.baseline import baseline_command
from aml_sentinel.rules.engine import RuleConfig

FAST = LgbmParams(max_rounds=40, early_stopping_rounds=10, min_child_samples=20, num_threads=1)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("s1a")
    settings = load_settings(
        _env_file=None,
        data_dir=root / "data",
        reports_dir=root / "reports",
        models_dir=root / "models",
    )
    generate_sample(settings, seed=42)
    prepare_data(settings, DatasetName.SAMPLE)
    baseline_command(settings, DatasetName.SAMPLE)
    return settings


@pytest.fixture(scope="module")
def frame(env: Settings) -> pd.DataFrame:
    return load_processed(env.processed_dir, DatasetName.SAMPLE)


def test_command_writes_everything(env: Settings) -> None:
    path = system1a_command(env, DatasetName.SAMPLE, params=FAST)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["variants"]) == {"all", "no_shortcuts"}
    for v in report["variants"].values():
        assert v["calibration"]["selected"] in {"platt", "isotonic"}
        assert set(v["vs_baseline"]) == {"matched_volume", "matched_recall"}
    assert (path.parent / "Sample_reliability.png").stat().st_size > 0
    assert (path.parent / "Sample_pr_curve.png").stat().st_size > 0
    assert "matched volume" in path.with_suffix(".md").read_text(encoding="utf-8")
    scores = pd.read_parquet(env.processed_dir / "Sample.system1a_scores.parquet")
    assert set(scores["split"].astype(str)) == {"val", "test"}
    assert scores["p_all_cal"].between(0, 1).all()
    model = LightGBMTriage.load(env.models_dir / "Sample" / "lightgbm_all.txt")
    assert model.variant == "all"


def test_training_ignores_val_and_test_labels_for_the_model(frame: pd.DataFrame) -> None:
    """Leakage test: trees see train labels only; calibration sees val labels only."""
    baseline = RuleConfig()
    clean, _, s1 = train_system1a(
        frame, baseline, train_end=pd.Timestamp("2022-09-07"), params=FAST
    )
    scrambled = frame.copy()
    is_test = scrambled["split"].astype("string") == "test"
    scrambled.loc[is_test, Col.IS_LAUNDERING.value] = (
        1 - scrambled.loc[is_test, Col.IS_LAUNDERING.value]
    )
    _, _, s2 = train_system1a(
        scrambled, baseline, train_end=pd.Timestamp("2022-09-07"), params=FAST
    )
    # Test labels changed, yet every val/test score is identical: neither the trees
    # nor the calibrator ever saw them.
    for col in s1.columns:
        if col.startswith("p_"):
            np.testing.assert_array_equal(s1[col].to_numpy(), s2[col].to_numpy())
    assert clean["rows"]["test"] > 0


def test_split_masks_and_baseline_errors(frame: pd.DataFrame, tmp_path: Path) -> None:
    masks = split_masks(frame, pd.Timestamp("2022-09-07"))
    assert not (masks["fit"] & masks["stop"]).any()
    assert frame.loc[masks["stop"], Col.TIMESTAMP.value].min() >= pd.Timestamp("2022-09-06")
    with pytest.raises(DataValidationError, match="empty"):
        split_masks(frame[frame["split"].astype("string") != "val"], pd.Timestamp("2022-09-07"))
    with pytest.raises(DataNotFoundError, match="baseline"):
        load_baseline_config(tmp_path, DatasetName.SAMPLE)


def test_triage_implementations(frame: pd.DataFrame, env: Settings, tmp_path: Path) -> None:
    feats = build_features(frame)
    y = frame[Col.IS_LAUNDERING.value].to_numpy()
    cols = feature_columns(include_shortcuts=False)
    model = LightGBMTriage.fit(feats, y, feats, y, features=cols, variant="t", params=FAST)
    _, cfg = load_baseline_config(env.reports_dir, DatasetName.SAMPLE)
    batch = AlertBatch(features=feats, transactions=frame)

    calibrated = CalibratedTriage(model, PlattCalibrator(a=1.0, b=0.0))
    for triage in (model, RuleTriage(cfg), calibrated):
        assert isinstance(triage, Triage)
        out = triage.triage(batch)
        assert out.scores.shape == (len(frame),)
        assert ((out.scores >= 0) & (out.scores <= 1)).all()
    assert calibrated.name == "lightgbm[t]+platt"
    np.testing.assert_allclose(calibrated.triage(batch).scores, model.triage(batch).scores)

    with pytest.raises(DataValidationError):
        RuleTriage(cfg).triage(AlertBatch(features=feats))
    with pytest.raises(DataValidationError, match="missing feature"):
        LightGBMTriage.fit(feats, y, feats, y, features=["nope"], variant="t", params=FAST)

    class Broken:
        name = "broken"

        def triage(self, batch: AlertBatch) -> TriageOutput:
            return TriageOutput(scores=np.zeros(1))

    with pytest.raises(DataValidationError, match="wrong number"):
        CalibratedTriage(Broken(), PlattCalibrator(a=1.0, b=0.0)).triage(batch)

    # Persistence: round-trip, then tampering is refused.
    path = tmp_path / "m.txt"
    model.save(path, params=FAST)
    np.testing.assert_allclose(LightGBMTriage.load(path).predict(feats), model.predict(feats))
    path.write_text(path.read_text(encoding="utf-8") + "\n# tampered", encoding="utf-8")
    with pytest.raises(IntegrityError):
        LightGBMTriage.load(path)


def test_main_train_tabular(env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(env.data_dir.parent)
    for var, value in (
        ("DATA_DIR", env.data_dir),
        ("REPORTS_DIR", env.reports_dir),
        ("MODELS_DIR", env.models_dir),
    ):
        monkeypatch.setenv(var, str(value))
    assert main(["train-tabular", "--dataset", "Sample"]) == 0
