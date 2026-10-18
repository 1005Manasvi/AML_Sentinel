"""Phase 8: smoothed calibration, ablation mechanics, overlap analysis and the report."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml_sentinel.cli import generate_sample, main, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.evaluation.report import evaluate_command, render_markdown
from aml_sentinel.evaluation.study import (
    AblationStep,
    as_probability,
    attempt_overlap,
    calibration_study,
    recent_mask,
    reliability_bands,
    rule_scores,
    run_ablation,
)
from aml_sentinel.exceptions import ConfigurationError
from aml_sentinel.models.calibration import (
    FITTERS,
    FITTERS_WITH_SMOOTHING,
    IsotonicCalibrator,
    calibrator_from_dict,
    select_method,
)
from aml_sentinel.models.fusion import FusionModel, stack_inputs
from aml_sentinel.models.laya_triage import LayaTriage
from aml_sentinel.models.system1a import load_baseline_config
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.models.system1b_report import system1b_command
from aml_sentinel.rules.baseline import baseline_command
from aml_sentinel.system2.cases import CaseIndex
from aml_sentinel.system2.evaluate import run_cases
from tests.conftest import FAST_LGBM
from tests.test_text_triage import FakeLaya


def _noisy(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.random(n)
    y = (rng.random(n) < s**3).astype(int)
    return s, y


# ---------------------------------------------------------------- smoothed isotonic
def test_smoothed_isotonic_never_reaches_zero_or_one() -> None:
    s, y = _noisy(3_000, 0)
    s[-4:], y[-4:] = 2.0, 1  # a tiny all-positive top block
    s[:4], y[:4] = -1.0, 0  # and an all-negative bottom block
    plain = IsotonicCalibrator.fit(s, y)
    smooth = IsotonicCalibrator.fit(s, y, prior_strength=2.0)
    assert max(plain.y) == 1.0
    assert min(plain.y) == 0.0
    assert 0.0 < min(smooth.y) < max(smooth.y) < 1.0
    assert all(b >= a for a, b in zip(smooth.y, smooth.y[1:], strict=False))
    assert smooth.method == "isotonic_smoothed"
    assert plain.method == "isotonic"


def test_smoothing_matters_only_for_small_blocks() -> None:
    s, y = _noisy(20_000, 1)
    plain = IsotonicCalibrator.fit(s, y)
    smooth = IsotonicCalibrator.fit(s, y, prior_strength=2.0)
    mid = np.linspace(0.3, 0.7, 50)
    assert np.max(np.abs(plain.transform(mid) - smooth.transform(mid))) < 0.02


def test_smoothed_calibrator_round_trips_and_rejects_inconsistent_json() -> None:
    s, y = _noisy(500, 2)
    cal = IsotonicCalibrator.fit(s, y, prior_strength=2.0)
    assert calibrator_from_dict(cal.to_dict()) == cal
    assert "prior_strength" not in IsotonicCalibrator.fit(s, y).to_dict()
    bad = {**cal.to_dict(), "method": "isotonic"}
    with pytest.raises(ConfigurationError):
        calibrator_from_dict(bad)


def test_default_fitters_are_unchanged_for_system1() -> None:
    assert set(FITTERS) == {"platt", "isotonic"}
    assert set(FITTERS_WITH_SMOOTHING) == {"platt", "isotonic", "isotonic_smoothed"}
    s, y = _noisy(2_000, 3)
    best, scores = select_method(s, y, fitters=FITTERS_WITH_SMOOTHING)
    assert best in scores
    assert set(scores) == set(FITTERS_WITH_SMOOTHING)


def test_fusion_records_its_calibrator_selection() -> None:
    s, y = _noisy(1_000, 4)
    x = stack_inputs({"p_a": np.clip(s, 1e-3, 1 - 1e-3)})
    model = FusionModel.fit(x, y, fitters=FITTERS_WITH_SMOOTHING)
    assert set(model.selection) == set(FITTERS_WITH_SMOOTHING)
    assert model.calibrator.method == min(model.selection, key=model.selection.__getitem__)
    again = FusionModel.from_dict(model.to_dict())
    assert again.selection == model.selection
    legacy = {k: v for k, v in model.to_dict().items() if k != "calibrator_selection"}
    assert FusionModel.from_dict(legacy).selection == {}


# ---------------------------------------------------------------- study helpers
def test_ablation_detects_an_informative_step_and_keeps_skips() -> None:
    rng = np.random.default_rng(5)
    n = 2_000
    y_val, y_test = rng.random(n) < 0.1, rng.random(n) < 0.1
    noise_val, noise_test = rng.random(n), rng.random(n)
    good_val = as_probability(np.where(y_val, 0.7, 0.3) + rng.normal(0, 0.1, n))
    good_test = as_probability(np.where(y_test, 0.7, 0.3) + rng.normal(0, 0.1, n))
    steps = [
        AblationStep("noise", {"noise": noise_val}, {"noise": noise_test}),
        AblationStep("skipped", {}, {}, skipped="not run"),
        AblationStep("+ signal", {"p_good": good_val}, {"p_good": good_test}),
    ]
    rows, scores = run_ablation(y_val, y_test, steps, seed=0)
    assert [r["name"] for r in rows] == ["noise", "skipped", "+ signal"]
    assert rows[1] == {"name": "skipped", "skipped": "not run", "note": ""}
    assert rows[0]["vs_previous"] is None
    gain = rows[2]["vs_previous"]
    assert gain["previous"] == "noise"
    assert gain["ci_low"] > 0
    assert rows[2]["inputs"] == ["noise", "logit_p_good"]
    assert rows[2]["top_5pct"]["k"] == 100
    assert rows[2]["fp_reduction_vs_first"] > 0
    assert set(scores) == {"noise", "+ signal"}


def test_reliability_bands_partition_the_scores() -> None:
    s, y = _noisy(1_000, 6)
    bands = reliability_bands(s, y, n_bins=10)
    assert sum(b["count"] for b in bands) == 1_000
    assert all(b["lo"] <= b["mean_pred"] <= b["hi"] for b in bands)
    assert [b["lo"] for b in bands] == sorted(b["lo"] for b in bands)
    assert bands[-1]["frac_pos"] > bands[0]["frac_pos"]


def test_recent_mask_is_half_open() -> None:
    end = datetime(2022, 9, 9)
    ts = pd.Series(pd.to_datetime(["2022-09-07 12:00", "2022-09-08 00:00", "2022-09-09 00:00"]))
    assert recent_mask(ts, end, timedelta(days=1)).tolist() == [False, True, False]


def test_calibration_study_reports_every_candidate() -> None:
    s, y = _noisy(4_000, 7)
    recent = np.zeros(4_000, dtype=bool)
    recent[2_000:] = True
    out = calibration_study(s[:3000], y[:3000], s[3000:], y[3000:], recent_val=recent[:3000])
    assert out["selected_on_validation"] in out["val_crossfit_ece"]
    assert set(out["test"]) == {
        "raw (uncalibrated)",
        "platt",
        "isotonic",
        "isotonic_smoothed",
        "isotonic_recent_window",
        "platt_recent_window",
    }
    assert out["test"]["isotonic_smoothed"]["share_at_one"] == 0.0


def _overlap_frame() -> pd.DataFrame:
    rows = [
        # (tx, split, from, to, label, attempt)
        (1, "train", "A", "B", 1, 10.0),
        (2, "train", "C", "D", 0, np.nan),
        (3, "test", "A", "E", 1, 10.0),  # continues attempt 10, known account A
        (4, "test", "F", "G", 1, 20.0),  # new attempt, new accounts
        (5, "test", "H", "I", 1, np.nan),  # undocumented positive
        (6, "test", "J", "K", 0, np.nan),
        (7, "test", "L", "M", 0, np.nan),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[Col.TX_ID, "split", Col.FROM_ID, Col.TO_ID, Col.IS_LAUNDERING, "attempt_id"],
    )
    return frame.astype({Col.FROM_ID: "category", Col.TO_ID: "category", "split": "category"})


def test_attempt_overlap_groups_positives() -> None:
    frame = _overlap_frame()
    # 300 mid-scored negatives push the low-scored positive out of the 200-alert queue.
    filler = pd.DataFrame(
        {
            Col.TX_ID: np.arange(100, 400),
            "split": "test",
            Col.FROM_ID: "N",
            Col.TO_ID: "O",
            Col.IS_LAUNDERING: 0,
            "attempt_id": np.nan,
        }
    )
    frame = pd.concat([frame.astype(object), filler], ignore_index=True)
    tx = [3, 4, 5, 6, 7, *range(100, 400)]
    scores = [0.9, 0.8, 0.01, 0.2, 0.05, *([0.5] * 300)]
    out = attempt_overlap(frame, tx, scores)
    assert out["test_positives"] == 3
    assert out["queue_size"] == 200
    assert out["attempt_seen_before_test"]["positives"] == 1
    assert out["attempt_new_in_test"]["positives"] == 1
    assert out["undocumented_positive"]["positives"] == 1
    assert out["account_seen_laundering_before"]["positives"] == 1
    assert out["account_new"]["positives"] == 2
    assert out["attempt_seen_before_test"]["recall_in_queue"] == 1.0
    assert out["undocumented_positive"]["recall_in_queue"] == 0.0
    assert out["account_new"]["recall_in_queue"] == 0.5


# ---------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def eval_env(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("eval")
    settings = load_settings(
        _env_file=None,
        data_dir=root / "data",
        reports_dir=root / "reports",
        models_dir=root / "models",
    )
    generate_sample(settings, seed=42)
    prepare_data(settings, DatasetName.SAMPLE)
    baseline_command(settings, DatasetName.SAMPLE)
    system1a_command(settings, DatasetName.SAMPLE, params=FAST_LGBM)
    system1b_command(
        settings,
        DatasetName.SAMPLE,
        text_models={"laya[fake]": lambda: LayaTriage(FakeLaya(), variant="fake")},
    )
    s1b = pd.read_parquet(settings.processed_dir / "Sample.system1b_scores.parquet")
    index = CaseIndex(load_processed(settings.processed_dir, DatasetName.SAMPLE))
    parts = [
        run_cases(index, s1b.loc[s1b["split"] == split, "tx_id"].tolist()).assign(split=split)
        for split in ("val", "test")
    ]
    pd.concat(parts, ignore_index=True).to_parquet(
        settings.processed_dir / "Sample.system2_evidence.parquet", index=False
    )
    return settings


def test_rule_scores_match_the_rule_baseline(eval_env: Settings) -> None:
    frame = load_processed(eval_env.processed_dir, DatasetName.SAMPLE)
    _, rules = load_baseline_config(eval_env.reports_dir, DatasetName.SAMPLE)
    scores = rule_scores(frame, rules, np.arange(len(frame)))
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    allowed = {i / len(rules.enabled) for i in range(len(rules.enabled) + 1)}
    assert set(np.unique(scores)) <= allowed


def test_evaluate_command_writes_report_and_figures(eval_env: Settings) -> None:
    path = evaluate_command(eval_env, DatasetName.SAMPLE)
    report = json.loads(path.read_text(encoding="utf-8"))
    names = [s["name"] for s in report["ablation"]["steps"]]
    assert names == [
        "rules only",
        "+ LightGBM",
        "+ Laya",
        "+ Jev (benchmark)",
        "+ graph evidence",
        "+ GNN (full stack)",
    ]
    jev = report["ablation"]["steps"][3]
    assert "not run" in jev["skipped"]
    assert report["ablation"]["steps"][5]["skipped"]  # no GNN artifacts in this env
    assert report["ablation"]["references"][0]["name"].startswith("production: System 1")
    assert report["ablation"]["p_triage_reliability_test"]
    assert "laya[fake]" in report["ablation"]["steps"][2]["note"]
    assert set(report["calibration"]["system1a_lgbm_all"]["test"]) >= {"isotonic_smoothed"}
    stages = report["latency"]["stages"]
    assert any(k.startswith("API POST /score") for k in stages)
    assert any(k.startswith("SAR draft") for k in stages)
    out = eval_env.reports_dir / "evaluation"
    for fig in report["figures"]:
        assert (out / fig).stat().st_size > 10_000
    md = (out / "Sample.md").read_text(encoding="utf-8")
    assert "## 2. Ablation on the alert queue" in md
    assert "+ Jev (benchmark)" in md
    assert render_markdown(report) == md


def test_evaluate_cli(eval_env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (
        ("DATA_DIR", eval_env.data_dir),
        ("REPORTS_DIR", eval_env.reports_dir),
        ("MODELS_DIR", eval_env.models_dir),
    ):
        monkeypatch.setenv(name, str(value))
    assert main(["evaluate", "--dataset", "Sample"]) == 0


def test_evaluate_without_system1b_has_no_ablation(eval_env: Settings, tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for f in eval_env.processed_dir.iterdir():
        if f.is_file() and "system1b" not in f.name:
            (processed / f.name).write_bytes(f.read_bytes())
    other = eval_env.model_copy(update={"data_dir": tmp_path / "data"})
    report = json.loads(evaluate_command(other, DatasetName.SAMPLE).read_text("utf-8"))
    assert report["ablation"] is None
    assert "Not available" in render_markdown(report)


def test_gnn_loader_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    from safetensors.torch import save_file

    from aml_sentinel.exceptions import IntegrityError
    from aml_sentinel.integrity import sha256_file
    from aml_sentinel.system2.gnn import GnnConfig, load_trained_gnn, train_gnn
    from aml_sentinel.system2.gnn_data import build_split_graphs
    from aml_sentinel.system2.gnn_report import alert_pair_scores

    settings = load_settings(_env_file=None, data_dir=tmp_path / "data")
    generate_sample(settings, seed=7)
    prepare_data(settings, DatasetName.SAMPLE)
    frame = load_processed(settings.processed_dir, DatasetName.SAMPLE)
    graphs = build_split_graphs(frame)
    cfg = GnnConfig(max_epochs=3, patience=2)
    trained = train_gnn(graphs["train"], graphs["val"], cfg)
    weights = tmp_path / "gnn.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in trained.model.state_dict().items()},
        str(weights),
    )
    loaded = load_trained_gnn(weights, sha256_file(weights), graphs["train"], cfg)
    np.testing.assert_allclose(
        loaded.predict(graphs["test"]), trained.predict(graphs["test"]), rtol=1e-5
    )
    test_tx = frame.loc[frame["split"].astype("string") == "test", Col.TX_ID].head(5).tolist()
    scores, found = alert_pair_scores(
        frame, graphs["test"], loaded.predict(graphs["test"]), test_tx
    )
    assert found.all()
    assert scores.shape == (5,)
    with pytest.raises(IntegrityError):
        load_trained_gnn(weights, "0" * 64, graphs["train"], cfg)
