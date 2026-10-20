from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aml_sentinel.api.app import create_app
from aml_sentinel.api.scoring import build_scorer
from aml_sentinel.api.service import CaseService
from aml_sentinel.api.store import CaseStore
from aml_sentinel.audit import AuditLog
from aml_sentinel.dashboard.views import explanation_chart, explanation_table
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.features.builder import build_features
from aml_sentinel.models.explain import explain, explanation_from_row
from aml_sentinel.models.lgbm import LightGBMTriage
from tests.conftest import ANALYST_KEY, ServedSample

ANALYST = {"X-API-Key": ANALYST_KEY}


def _model(served: ServedSample) -> LightGBMTriage:
    return LightGBMTriage.load(served.settings.models_dir / "Sample" / "lightgbm_all.txt")


def test_contributions_add_up_to_the_models_log_odds(served_sample: ServedSample) -> None:
    """TreeSHAP is exact: bias + contributions == the model's own log-odds."""
    model = _model(served_sample)
    features = build_features(served_sample.frame).iloc[-50:]
    contrib = model.contributions(features)
    assert contrib.shape == (50, len(model.features) + 1)
    p = model.predict(features)
    np.testing.assert_allclose(contrib.sum(axis=1), np.log(p / (1 - p)), atol=1e-6)


def test_explanation_keeps_the_largest_contributions(served_sample: ServedSample) -> None:
    model = _model(served_sample)
    features = build_features(served_sample.frame).iloc[-20:]
    for e, row in zip(
        explain(model, features, top_k=4), model.contributions(features), strict=True
    ):
        sizes = [abs(c.contribution) for c in e.top]
        assert len(e.top) == 4
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[-1] >= np.sort(np.abs(row[:-1]))[-4] - 1e-12
        top_sum = sum(c.contribution for c in e.top)
        assert e.log_odds == pytest.approx(e.base_log_odds + top_sum + e.others)
        assert {c.feature for c in e.top} <= set(model.features)


def test_explanation_from_row_validates_inputs() -> None:
    feats = ["a", "b"]
    with pytest.raises(ValueError, match="top_k"):
        explanation_from_row("m", feats, np.zeros(2), np.zeros(3), top_k=0)
    with pytest.raises(ValueError, match="align"):
        explanation_from_row("m", feats, np.zeros(2), np.zeros(2))
    e = explanation_from_row("m", feats, np.array([1.0, 2.0]), np.array([-0.5, 2.0, 0.1]), 1)
    assert [c.feature for c in e.top] == ["b"]
    assert e.others == pytest.approx(-0.5)
    assert e.log_odds == pytest.approx(1.6)


def test_case_detail_explains_the_alert(client: TestClient) -> None:
    items = client.get("/alerts", headers=ANALYST, params={"escalated_only": True}).json()["items"]
    d = client.get(f"/cases/{items[0]['case_id']}", headers=ANALYST).json()
    e = d["explanation"]
    assert d["explanation_unavailable"] is None
    assert e["model"] == "lightgbm[all]"
    assert 1 <= len(e["top"]) <= 6
    total = e["base_log_odds"] + sum(c["contribution"] for c in e["top"]) + e["others"]
    assert total == pytest.approx(e["log_odds"], abs=1e-9)
    assert "TreeSHAP" in e["method"]


def test_scorer_rejects_unknown_transactions(served_sample: ServedSample) -> None:
    scorer, _ = build_scorer(served_sample.settings, DatasetName.SAMPLE, served_sample.frame)
    assert scorer is not None
    with pytest.raises(DataValidationError, match="not in the scoring history"):
        scorer.explain(int(served_sample.frame[Col.TX_ID].max()) + 1)


def test_no_explanation_above_the_history_cap(served_sample: ServedSample, tmp_path: Path) -> None:
    reason = "scoring new transactions is disabled; production needs an online feature store"
    service = CaseService(
        queue=served_sample.queue,
        index=served_sample.index,
        store=CaseStore(tmp_path / "c.sqlite3"),
        audit=AuditLog(tmp_path / "a.jsonl"),
        scoring_disabled_reason=reason,
    )
    c = TestClient(create_app(served_sample.settings, service=service))
    case_id = next(iter(served_sample.queue)).case_id
    d = c.get(f"/cases/{case_id}", headers=ANALYST).json()
    assert d["explanation"] is None
    assert "online feature store" in d["explanation_unavailable"]


def test_dashboard_explanation_views_escape_and_order() -> None:
    e = {
        "top": [
            {"feature": "<b>x</b>", "value": 1.0, "contribution": 0.9},
            {"feature": "y", "value": 2.0, "contribution": -0.3},
        ]
    }
    table = explanation_table(e)
    assert list(table["effect"]) == ["raises risk", "lowers risk"]
    fig = explanation_chart(e)
    assert "<b>" not in str(fig.data[0].y)  # labels are HTML-escaped
    assert list(fig.data[0].x) == [-0.3, 0.9]  # largest drawn on top
