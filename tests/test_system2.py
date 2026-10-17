"""System 2 evaluation, GNN data/model and commands (GNN tests skip without torch)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from aml_sentinel.cli import generate_sample, main, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.features.graph_features import account_codes
from aml_sentinel.models.lgbm import LgbmParams
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.rules.baseline import baseline_command
from aml_sentinel.system2.detectors import Finding, FindingKind
from aml_sentinel.system2.evaluate import evidence_features, separation, typology_agreement
from aml_sentinel.system2.gnn_data import (
    EDGE_FEATURES,
    build_period_graph,
    build_split_graphs,
    pair_max_scores,
)
from aml_sentinel.system2.report import detect_command

FAST = LgbmParams(max_rounds=30, early_stopping_rounds=5, min_child_samples=20, num_threads=1)
T0 = datetime(2022, 9, 1)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("s2")
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


# ----------------------------------------------------------------- evaluation helpers
def test_evidence_features() -> None:
    findings = [
        Finding(FindingKind.FAN_IN, ("a", "b"), (1,), 6.0),
        Finding(FindingKind.FAN_IN, ("a", "c"), (2,), 9.0),
        Finding(FindingKind.CIRCULAR, ("a", "b"), (3,), 1.0),
    ]
    f = evidence_features(findings)
    assert f["n_findings"] == 3
    assert f["n_fan_in"] == 2
    assert f["max_fan_in"] == 9
    assert f["n_structuring"] == 0
    assert f["max_structuring"] == 0


def test_separation_and_agreement() -> None:
    ev = pd.DataFrame(
        {"n_findings": [1, 0, 2, 0], **{f"n_{k.value}": [0, 0, 0, 0] for k in FindingKind}}
    )
    ev["n_circular"] = [1, 0, 1, 0]
    sep = separation(ev, [1, 0, 0, 0])
    assert sep["any"]["fires"] == 2
    assert sep["any"]["precision"] == 0.5
    assert sep["any"]["lift"] == pytest.approx(2.0)
    assert sep["structuring"]["precision"] is None
    agree = typology_agreement(ev, ["CYCLE", None, "STACK", None])
    assert agree["CYCLE"]["rate"] == 1.0
    assert agree["STACK"]["rate"] is None  # no dedicated detector
    assert agree["STACK"]["any_rate"] == 1.0


# ----------------------------------------------------------------- GNN data
def _frame() -> pd.DataFrame:
    rows = [  # (from, to, day, usd, label, split)
        ("A", "B", 0, 100.0, 0, "train"),
        ("B", "C", 0, 200.0, 1, "train"),
        ("A", "B", 1, 300.0, 0, "train"),
        ("C", "C", 1, 50.0, 0, "train"),
        ("A", "C", 2, 400.0, 1, "val"),
        ("B", "A", 3, 500.0, 0, "test"),
        ("B", "A", 3, 50.0, 1, "test"),
        ("C", "D", 3, 10.0, 0, "test"),
    ]
    return pd.DataFrame(
        {
            Col.TX_ID.value: range(len(rows)),
            Col.FROM_ID.value: [r[0] for r in rows],
            Col.TO_ID.value: [r[1] for r in rows],
            Col.TIMESTAMP.value: pd.to_datetime([T0 + timedelta(days=r[2]) for r in rows]),
            Col.AMOUNT_PAID_USD.value: [r[3] for r in rows],
            Col.IS_LAUNDERING.value: [r[4] for r in rows],
            "split": [r[5] for r in rows],
        }
    )


def test_period_graph_aggregates_pairs_without_self_loops() -> None:
    g = build_split_graphs(_frame())
    train = g["train"]
    assert train.edge_index.shape == (2, 2)  # A->B (2 payments), B->C; C->C dropped
    assert train.y.tolist().count(1) == 1
    assert train.edge_attr.shape == (2, len(EDGE_FEATURES))
    test = g["test"]
    assert test.y.sum() == 1  # B->A has one laundering payment of two
    assert test.x.shape[0] == train.x.shape[0]  # shared account index


def test_period_node_features_ignore_rows_outside_the_period() -> None:
    df = _frame()
    src, dst, n = account_codes(df)
    split = df["split"].to_numpy()
    test = build_period_graph(df, split == "test", src, dst, n)
    later = pd.concat(
        [df, df.iloc[[0]].assign(**{Col.TIMESTAMP.value: T0 + timedelta(days=9), "split": "tail"})],
        ignore_index=True,
    )
    s2, d2, n2 = account_codes(later)
    again = build_period_graph(later, later["split"].to_numpy() == "test", s2, d2, n2)
    np.testing.assert_array_equal(test.x, again.x)  # rows outside the period change nothing
    # Every period carries informative (non-constant) node features, train included.
    train = build_split_graphs(df)["train"]
    assert train.x.std(axis=0).max() > 0
    with pytest.raises(DataValidationError):
        build_period_graph(df, np.zeros(len(df), dtype=bool), src, dst, n)


def test_pair_max_scores() -> None:
    df = _frame()
    g = build_split_graphs(df)["test"]
    scores = pd.Series([0.1] * 5 + [0.3, 0.9, 0.2], index=range(8))
    out = pair_max_scores(df, (df["split"] == "test").to_numpy(), scores, g)
    assert sorted(out.tolist()) == [0.2, 0.9]


# ----------------------------------------------------------------- commands
def test_detect_command_on_sample(env: Settings) -> None:
    path = detect_command(env, DatasetName.SAMPLE, alert_limit=80)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["horizons"]) == {"strict", "next_day"}
    sep = report["horizons"]["strict"]["test"]["separation"]
    assert sep["alerts"] > 0
    assert (env.processed_dir / "Sample.system2_evidence.parquet").is_file()
    assert "Horizon `strict`" in path.with_name("Sample_detectors.md").read_text(encoding="utf-8")


def test_detect_command_needs_system1a_for_ibm(tmp_path: object) -> None:
    from pathlib import Path

    settings = load_settings(_env_file=None, data_dir=Path(str(tmp_path)))
    from aml_sentinel.system2 import report as rep

    with pytest.raises(DataNotFoundError):
        rep._alerts(settings, DatasetName.HI_SMALL, _frame())


def test_gnn_experiment_on_sample(env: Settings) -> None:
    pytest.importorskip("torch_geometric")
    from aml_sentinel.system2.gnn import GnnConfig
    from aml_sentinel.system2.gnn_report import gnn_command

    detect_command(env, DatasetName.SAMPLE, alert_limit=80)
    path = gnn_command(env, DatasetName.SAMPLE, cfg=GnnConfig(max_epochs=5, patience=2))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["pairs"]) == {
        "gnn",
        "lgbm_same_pair_inputs",
        "lgbm_all_pairmax",
        "lgbm_no_shortcuts_pairmax",
    }
    assert "vs_detectors" in report
    assert (env.models_dir / "Sample" / "gnn_sage.safetensors").is_file()
    assert "Experiment, not production" in path.with_suffix(".md").read_text(encoding="utf-8")


def test_gnn_requires_system1a(tmp_path: object) -> None:
    pytest.importorskip("torch_geometric")
    from pathlib import Path

    from aml_sentinel.system2.gnn_report import gnn_command

    with pytest.raises(DataNotFoundError):
        gnn_command(load_settings(_env_file=None, data_dir=Path(str(tmp_path))), DatasetName.SAMPLE)


def test_gnn_missing_dependency_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from aml_sentinel.exceptions import ConfigurationError
    from aml_sentinel.system2 import gnn

    monkeypatch.setitem(sys.modules, "torch_geometric", None)
    with pytest.raises(ConfigurationError, match=r"gnn.txt"):
        gnn.build_model(3, 2, 4)


@pytest.mark.parametrize(
    ("command", "target", "module"),
    [("detect-patterns", "detect_command", "cli"), ("gnn-experiment", "gnn_command", "cli")],
)
def test_cli_routes_system2(
    command: str, target: str, module: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    import aml_sentinel.cli as cli

    called: list[DatasetName] = []
    monkeypatch.setattr(cli, target, lambda settings, dataset: called.append(dataset))
    monkeypatch.chdir(str(tmp_path))
    assert main([command, "--dataset", "Sample"]) == 0
    assert called == [DatasetName.SAMPLE]
