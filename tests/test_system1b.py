"""Fusion, System 1b evaluation and the Laya data / fine-tune wrappers (no GPU needed)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aml_sentinel.cli import generate_sample, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import ConfigurationError, DataNotFoundError, DataValidationError
from aml_sentinel.features.builder import build_features
from aml_sentinel.models.decisions import TYPOLOGY_LABELS
from aml_sentinel.models.fusion import FusionModel, logit, stack_inputs
from aml_sentinel.models.laya_data import (
    ROLE_EASY,
    ROLE_HARD,
    ROLE_POS_DOC,
    ROLE_POS_UNDOC,
    SetSizes,
    expected_answers,
    generator_structuring,
    hard_negative_mask,
    iter_jsonl_rows,
    role_of,
    sample_ibm_rows,
    write_jsonl,
)
from aml_sentinel.models.laya_finetune import (
    build_laya_dataset,
    finetune_head_only,
    laya_paths,
)
from aml_sentinel.models.laya_triage import LayaTriage, write_manifest
from aml_sentinel.models.lgbm import LgbmParams
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.models.system1b import (
    build_alert_sets,
    escalation_metrics,
    evaluate_system1b,
    typology_metrics,
)
from aml_sentinel.models.system1b_report import default_text_models, system1b_command
from aml_sentinel.models.triage import TriageOutput
from aml_sentinel.models.typology import LayaLabel
from aml_sentinel.rules.baseline import baseline_command
from tests.test_text_triage import FakeLaya

FAST = LgbmParams(max_rounds=40, early_stopping_rounds=10, min_child_samples=20, num_threads=1)
SMALL = SetSizes(
    max_positives=60,
    benign_easy=40,
    benign_hard=40,
    structuring=20,
    structuring_lookalikes=10,
    noisy_copies=1,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("s1b")
    settings = load_settings(
        _env_file=None,
        data_dir=root / "data",
        reports_dir=root / "reports",
        models_dir=root / "models",
    )
    generate_sample(settings, seed=42)
    prepare_data(settings, DatasetName.SAMPLE)
    baseline_command(settings, DatasetName.SAMPLE)
    system1a_command(settings, DatasetName.SAMPLE, params=FAST)
    return settings


@pytest.fixture(scope="module")
def frame(env: Settings) -> pd.DataFrame:
    return load_processed(env.processed_dir, DatasetName.SAMPLE)


def _scores(env: Settings) -> pd.DataFrame:
    return pd.read_parquet(env.processed_dir / "Sample.system1a_scores.parquet")


# ----------------------------------------------------------------- fusion
def _toy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    p_tab = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    p_esc = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, n), 0.01, 0.99)
    return stack_inputs(
        {"p_tabular": p_tab, "p_escalate": p_esc, "risk_expected": 1 + 3 * p_esc}
    ), y


def test_fusion_fits_explains_and_roundtrips() -> None:
    x, y = _toy()
    model = FusionModel.fit(x, y)
    p = model.predict(x)
    assert ((p >= 0) & (p <= 1)).all()
    assert model.explain()["logit_p_tabular"] > 0  # the strong signal gets weight
    clone = FusionModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(clone.predict(x), p)


def test_fusion_input_errors() -> None:
    x, y = _toy()
    model = FusionModel.fit(x, y)
    with pytest.raises(DataValidationError, match="do not match"):
        model.predict(x[["logit_p_tabular"]])
    with pytest.raises(DataValidationError, match="20 rows"):
        FusionModel.fit(x.head(5), y[:5])
    with pytest.raises(DataValidationError, match="one class"):
        FusionModel.fit(
            x,
            np.r_[np.zeros(200), np.r_[1, np.zeros(199)]].astype(int)[::-1] * 0
            + np.r_[np.zeros(200), np.ones(200)].astype(int),
        )
    with pytest.raises(DataValidationError, match="length"):
        stack_inputs({"p_a": [0.1, 0.2], "b": [1.0]})
    with pytest.raises(DataValidationError, match="non-finite"):
        stack_inputs({"p_a": [0.1, float("nan")]})
    bad = model.to_dict()
    bad["coef"] = [1.0]
    with pytest.raises(DataValidationError, match="inconsistent"):
        FusionModel.from_dict(bad)
    assert logit([0.5])[0] == pytest.approx(0.0)


# ----------------------------------------------------------------- system 1b metrics
def test_typology_and_escalation_metrics() -> None:
    m = typology_metrics(
        ["circular", "benign", None, "benign"], ["circular", "layering", "benign", "benign"]
    )
    assert m["n"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["confusion"]["benign"]["layering"] == 1
    assert typology_metrics([None], ["benign"])["accuracy"] is None
    esc = escalation_metrics(
        [0, 1] * 50, np.linspace(0, 1, 100), [0, 1, 1, 0], [0.1, 0.99, 0.2, 0.97], share=0.05
    )
    assert esc["test"]["alerts"] == 2
    assert esc["test_share"] == pytest.approx(0.5)


def test_alert_sets_follow_the_queue(frame: pd.DataFrame, env: Settings) -> None:
    sets = build_alert_sets(frame, _scores(env), rate=0.05, sample=50, seed=1)
    assert set(sets) == {"val", "test"}
    for a in sets.values():
        assert a.rows.size <= 50
        assert np.all(np.diff(a.rows) >= 0)  # time order (needed for cross-fitting)
        assert len(a.typology) == a.rows.size
        assert all(t is None or t in TYPOLOGY_LABELS for t in a.typology)
    with pytest.raises(DataValidationError):
        build_alert_sets(frame.iloc[:10], _scores(env))


def _fake_factories() -> dict[str, Any]:
    return {"laya[fake]": lambda: LayaTriage(FakeLaya(), variant="fake")}


def test_system1b_command_end_to_end(env: Settings) -> None:
    path = system1b_command(env, DatasetName.SAMPLE, text_models=_fake_factories())
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["jev_status"].startswith("not run")
    assert {"lgbm_all", "lgbm_no_shortcuts", "laya[fake]"} <= set(report["models"])
    assert "fused[lgbm_all+laya[fake]]" in report["fusion"]
    assert "fused[lgbm_all+laya[fake]] vs fused[lgbm_all]" in report["significance"]
    assert report["models"]["laya[fake]"]["latency_ms_per_alert"] is not None
    assert (path.parent / "Sample_reliability.png").stat().st_size > 0
    assert "Typology acc." in path.with_suffix(".md").read_text(encoding="utf-8")
    fusion_files = list((env.models_dir / "Sample").glob("fusion_*.json"))
    assert len(fusion_files) == 4


def test_fusion_never_sees_test_labels(frame: pd.DataFrame, env: Settings) -> None:
    """Leakage test: scrambling test labels must not change the fitted fusion models."""
    sets = build_alert_sets(frame, _scores(env), rate=0.05, sample=200, seed=3)
    feats = build_features(frame)
    model = LayaTriage(FakeLaya(), variant="fake")
    from aml_sentinel.models.system1b import run_text_model

    outs = {
        "laya[fake]": {
            s: run_text_model(model, feats, a, is_synthetic=True)[0] for s, a in sets.items()
        }
    }
    _, _, fus_a = evaluate_system1b(sets, outs, {})
    scrambled = dict(sets)
    t = sets["test"]
    scrambled["test"] = type(t)(
        rows=t.rows,
        y=(1 - t.y).astype(np.int8),
        typology=t.typology,
        tabular=t.tabular,
        queue_size=t.queue_size,
    )
    _, _, fus_b = evaluate_system1b(scrambled, outs, {})
    for combo in fus_a:
        assert fus_a[combo].to_dict() == fus_b[combo].to_dict()


def test_system1b_requires_system1a(tmp_path: Path) -> None:
    settings = load_settings(_env_file=None, data_dir=tmp_path)
    with pytest.raises(DataNotFoundError, match="train-tabular"):
        system1b_command(settings, DatasetName.SAMPLE, text_models={})


def test_default_text_models(env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    assert default_text_models(env, DatasetName.SAMPLE) == {}
    base = laya_paths(env, DatasetName.SAMPLE)["base"]
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.safetensors").write_bytes(b"w")
    write_manifest(base, source="t", revision=None)
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("JEV_API_KEY", "k")
    names = set(
        default_text_models(
            load_settings(_env_file=None, models_dir=env.models_dir), DatasetName.SAMPLE
        )
    )
    assert names == {"laya[zero_shot]", "jev"}


def test_text_model_outputs_must_align() -> None:
    out = TriageOutput(scores=np.zeros(2))
    assert out.extras is None


# ----------------------------------------------------------------- Laya data
def test_roles_and_targets() -> None:
    assert role_of(True, "CYCLE", hard=False) == ROLE_POS_DOC
    assert role_of(True, None, hard=True) == ROLE_POS_UNDOC
    assert role_of(False, None, hard=True) == ROLE_HARD
    assert role_of(False, None, hard=False) == ROLE_EASY
    assert expected_answers(ROLE_POS_DOC, LayaLabel.CIRCULAR) == {
        "escalate": True,
        "risk": 4,
        "typology": "circular",
    }
    assert "typology" not in expected_answers(ROLE_POS_UNDOC, None)
    assert expected_answers(ROLE_EASY, LayaLabel.BENIGN)["risk"] == 0


def test_laya_rows_come_from_train_split_only(frame: pd.DataFrame) -> None:
    """Leakage test: no validation or test transaction may enter Laya's training set."""
    feats = build_features(frame)
    picked = sample_ibm_rows(frame, feats, SMALL, np.random.default_rng(0))
    splits = set(frame["split"].astype(str).to_numpy()[picked["row"].to_numpy()])
    assert splits == {"train"}
    assert hard_negative_mask(feats).dtype == bool


def test_jsonl_rows_and_generator(tmp_path: Path, frame: pd.DataFrame) -> None:
    feats, meta = generator_structuring(SMALL, seed=7)
    assert (meta["role"] == ROLE_POS_DOC).sum() > 0
    assert set(meta["typology"].dropna()) == {"STRUCTURING"}
    rows = list(
        iter_jsonl_rows(
            feats.head(5), meta.head(5), noisy_copies=1, rng=np.random.default_rng(0), source="t"
        )
    )
    assert len(rows) == 10
    assert {"state", "questions", "expected", "meta"} <= set(rows[0])
    counts = write_jsonl(iter(rows), tmp_path / "x.jsonl")
    assert counts["total"] == 10


def test_build_dataset_and_finetune_wrapper(env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    stats = build_laya_dataset(env, DatasetName.SAMPLE, sizes=SMALL)
    assert stats["counts"]["total"] > 0
    paths = laya_paths(env, DatasetName.SAMPLE)
    first = json.loads(paths["train_jsonl"].read_text(encoding="utf-8").splitlines()[0])
    assert Col.FROM_ID.value not in first["state"]

    seen: dict[str, Any] = {}

    class TrainConfig:
        def __init__(self, **kw: Any) -> None:
            seen["config"] = kw

    def finetune(data: str, model_dir: str, out: str, cfg: Any, device: str) -> dict[str, Any]:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "model.safetensors").write_bytes(b"tuned")
        seen["device"] = device
        return {"epoch_loss": [0.5]}

    monkeypatch.setitem(sys.modules, "laya", types.ModuleType("laya"))
    monkeypatch.setitem(
        sys.modules, "laya.train", types.SimpleNamespace(TrainConfig=TrainConfig, finetune=finetune)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    out = finetune_head_only(env, DatasetName.SAMPLE)
    assert seen["config"]["freeze_encoder"] is True
    assert seen["device"] == "cpu"
    assert "model.safetensors" in json.loads((out / "manifest.json").read_text())["sha256"]


def test_finetune_requires_dataset(tmp_path: Path) -> None:
    settings = load_settings(_env_file=None, data_dir=tmp_path, models_dir=tmp_path / "m")
    with pytest.raises(ConfigurationError, match="build-laya-data"):
        finetune_head_only(settings, DatasetName.SAMPLE)


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("build-laya-data", "build_laya_dataset"),
        ("finetune-laya", "finetune_head_only"),
        ("triage-text", "system1b_command"),
    ],
)
def test_cli_routes_text_commands(
    command: str, target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import aml_sentinel.cli as cli

    called: list[DatasetName] = []
    monkeypatch.setattr(cli, target, lambda settings, dataset: called.append(dataset))
    monkeypatch.chdir(tmp_path)
    assert cli.main([command, "--dataset", "Sample"]) == 0
    assert called == [DatasetName.SAMPLE]
