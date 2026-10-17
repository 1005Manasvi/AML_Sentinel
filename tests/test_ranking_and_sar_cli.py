"""Final ranking and the draft-sar command, end to end on the bundled sample."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aml_sentinel.cli import generate_sample, main, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.models.laya_triage import LayaTriage
from aml_sentinel.models.lgbm import LgbmParams
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.models.system1b_report import system1b_command
from aml_sentinel.ranking import EVIDENCE, join_inputs, rank_command, run_final_ranking
from aml_sentinel.rules.baseline import baseline_command
from aml_sentinel.sar.drafting import REVIEW_STAMP
from aml_sentinel.sar.report import audit_log, draft_command
from tests.test_text_triage import FakeLaya

FAST = LgbmParams(max_rounds=30, early_stopping_rounds=5, min_child_samples=20, num_threads=1)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("rank")
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
    system1b_command(
        settings,
        DatasetName.SAMPLE,
        text_models={"laya[fake]": lambda: LayaTriage(FakeLaya(), variant="fake")},
    )
    return settings


@pytest.fixture(scope="module")
def evidence(env: Settings) -> pd.DataFrame:
    """Detector evidence for exactly the System 1b alerts (val + test)."""
    from aml_sentinel.data.processed import load_processed
    from aml_sentinel.system2.cases import CaseIndex
    from aml_sentinel.system2.evaluate import run_cases

    s1b = pd.read_parquet(env.processed_dir / "Sample.system1b_scores.parquet")
    index = CaseIndex(load_processed(env.processed_dir, DatasetName.SAMPLE))
    parts = [
        run_cases(index, s1b.loc[s1b["split"] == split, "tx_id"].tolist()).assign(split=split)
        for split in ("val", "test")
    ]
    ev = pd.concat(parts, ignore_index=True)
    ev.to_parquet(env.processed_dir / "Sample.system2_evidence.parquet", index=False)
    return ev


def test_system1b_scores_hold_val_out_of_fold_and_test(env: Settings) -> None:
    s1b = pd.read_parquet(env.processed_dir / "Sample.system1b_scores.parquet")
    assert set(s1b["split"]) == {"val", "test"}
    assert "p__fused[lgbm_all+laya[fake]]" in s1b.columns


def test_final_ranking_report(env: Settings, evidence: pd.DataFrame) -> None:
    path = rank_command(env, DatasetName.SAMPLE)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["triage_input"] == "p__fused[lgbm_all]"  # no real Laya variant present
    assert set(report["final"]["coefficients"]) >= {f"ev_{e}" for e in EVIDENCE}
    assert "graph_evidence_gain" in report
    scores = pd.read_parquet(env.processed_dir / "Sample.final_scores.parquet")
    assert scores["p_final"].between(0, 1).all()
    assert (env.models_dir / "Sample" / "final_ranker.json").is_file()


def test_final_ranker_never_sees_test_labels(env: Settings, evidence: pd.DataFrame) -> None:
    """Leakage test: scrambling test labels leaves the fitted final model unchanged."""
    s1b = pd.read_parquet(env.processed_dir / "Sample.system1b_scores.parquet")
    _, model_a, _ = run_final_ranking(s1b, evidence)
    scrambled = s1b.copy()
    is_test = scrambled["split"] == "test"
    scrambled.loc[is_test, "y"] = 1 - scrambled.loc[is_test, "y"]
    _, model_b, _ = run_final_ranking(scrambled, evidence)
    assert model_a.to_dict() == model_b.to_dict()


def test_final_ranking_input_errors(env: Settings, evidence: pd.DataFrame) -> None:
    s1b = pd.read_parquet(env.processed_dir / "Sample.system1b_scores.parquet")
    with pytest.raises(DataValidationError, match="evidence"):
        join_inputs(s1b, evidence.iloc[:3], "p__fused[lgbm_all]")
    with pytest.raises(DataValidationError, match="fused column"):
        run_final_ranking(s1b.drop(columns=[c for c in s1b if c.startswith("p__fused")]), evidence)
    only_test = s1b.loc[s1b["split"] == "test"]
    with pytest.raises(DataValidationError, match="validation"):
        run_final_ranking(only_test, evidence)
    with pytest.raises(DataNotFoundError):
        rank_command(
            load_settings(_env_file=None, data_dir=Path(env.data_dir) / "none"), DatasetName.SAMPLE
        )


def test_draft_sar_command_writes_stamped_audited_drafts(
    env: Settings, evidence: pd.DataFrame
) -> None:
    rank_command(env, DatasetName.SAMPLE)
    out = draft_command(env, DatasetName.SAMPLE, top=3)
    drafts = sorted(out.glob("CASE-*.md"))
    assert len(drafts) == 3
    for d in drafts:
        text = d.read_text(encoding="utf-8")
        assert text.startswith(REVIEW_STAMP)
        meta = json.loads(d.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["is_grounded"] is True
    log = audit_log(env)
    assert log.verify() >= 3
    rec = log.records()[-1]
    assert rec.action == "sar.generated"
    assert set(rec.details) == {"source", "grounded", "n_issues"}
    assert "Grounded" in (out / "README.md").read_text(encoding="utf-8")


def test_draft_sar_falls_back_to_system1b_scores(env: Settings, evidence: pd.DataFrame) -> None:
    final = env.processed_dir / "Sample.final_scores.parquet"
    final.unlink(missing_ok=True)
    out = draft_command(env, DatasetName.SAMPLE, top=1)
    assert len(list(out.glob("CASE-*.json"))) >= 1
    with pytest.raises(DataNotFoundError):
        draft_command(
            load_settings(_env_file=None, data_dir=Path(env.data_dir) / "none"),
            DatasetName.SAMPLE,
            top=1,
        )


def test_external_provider_selected_from_settings(
    env: Settings, evidence: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aml_sentinel.sar import report as rep

    built: list[object] = []

    class Stub:
        name = "stub:model"

        def complete(self, system: str, user: str) -> str:
            return json.dumps(
                {
                    "narrative": "ACCT-001 paid ACCT-002 in the reviewed window; "
                    "graph analysis found related activity."
                }
            )

    monkeypatch.setattr(
        rep.AnthropicClient,
        "from_settings",
        classmethod(lambda cls, s, **k: built.append(s) or Stub()),
    )
    settings = load_settings(
        _env_file=None,
        data_dir=env.data_dir,
        reports_dir=env.reports_dir,
        models_dir=env.models_dir,
        llm_provider="external",
        llm_api_key="test-key",
    )
    out = draft_command(settings, DatasetName.SAMPLE, top=1)
    meta = json.loads(next(out.glob("CASE-*.json")).read_text(encoding="utf-8"))
    assert built
    assert meta["source"] == "stub:model"  # the valid external reply was used, no fallback
    assert meta["is_grounded"] is True


@pytest.mark.parametrize("command", ["rank", "draft-sar"])
def test_cli_routes(command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import aml_sentinel.cli as cli

    target = {"rank": "rank_command", "draft-sar": "draft_command"}[command]
    called: list[DatasetName] = []
    monkeypatch.setattr(cli, target, lambda settings, dataset: called.append(dataset))
    monkeypatch.chdir(tmp_path)
    assert main([command, "--dataset", "Sample"]) == 0
    assert called == [DatasetName.SAMPLE]
