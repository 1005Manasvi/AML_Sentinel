"""API behaviour: queue ranking, case detail, SAR workflow, decisions, audit, scoring."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aml_sentinel.api.app import create_app
from aml_sentinel.api.queue import AlertQueue, QueueEntry, build_queue, case_id_of
from aml_sentinel.api.scoring import TransactionScorer, build_scorer, load_calibrator
from aml_sentinel.api.service import CaseService, pattern_names
from aml_sentinel.api.store import CaseStore
from aml_sentinel.audit import AuditLog
from aml_sentinel.data.schema import Col, DatasetName, Transaction
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError, IntegrityError
from aml_sentinel.features.builder import build_features
from aml_sentinel.models.lgbm import LightGBMTriage
from aml_sentinel.models.system1b import ESCALATE_SHARE
from aml_sentinel.sar.drafting import REVIEW_STAMP
from tests.conftest import ADMIN_KEY, ANALYST_KEY, ServedSample, make_service

ANALYST = {"X-API-Key": ANALYST_KEY}
ADMIN = {"X-API-Key": ADMIN_KEY}
LABEL_WORDS = ("is_laundering", "typology", "attempt_id", '"y"')


def _top_escalated(client: TestClient) -> str:
    items = client.get("/alerts", headers=ANALYST, params={"escalated_only": True}).json()["items"]
    return str(items[0]["case_id"])


# ---------------------------------------------------------------- queue
def test_queue_is_ranked_by_p_triage_and_escalation_uses_validation(
    served_sample: ServedSample,
) -> None:
    q = served_sample.queue
    p = [e.p_triage for e in q]
    assert p == sorted(p, reverse=True)
    assert q.triage_source == "lgbm_all (calibrated)"  # no System 1b on the sample
    assert all(e.escalated == (e.p_triage >= q.threshold) for e in q)
    scores = pd.read_parquet(
        served_sample.settings.processed_dir / "Sample.system1a_scores.parquet"
    )
    val = scores.loc[scores["split"].astype("string") == "val", "p_all_cal"]
    queue_val = val.nlargest(200).to_numpy()
    assert q.threshold == pytest.approx(np.quantile(queue_val, 1 - ESCALATE_SHARE))
    # Evidence exists exactly for escalated alerts (System 2 runs only on those).
    assert all((e.evidence is not None) == e.escalated for e in q)


def test_queue_prefers_system1b_fused_scores(served_sample: ServedSample, tmp_path: Path) -> None:
    settings = served_sample.settings
    s1a = pd.read_parquet(settings.processed_dir / "Sample.system1a_scores.parquet")
    split = s1a["split"].astype("string")
    keep = pd.concat([s1a.loc[split == "val"].head(200), s1a.loc[split == "test"].head(200)])
    fused = pd.DataFrame(
        {
            "tx_id": keep["tx_id"].to_numpy(),
            "y": 1,  # must never be read
            "split": keep["split"].astype("string").to_numpy(),
            "p__fused[lgbm_all]": np.linspace(0.9, 0.1, len(keep)),
        }
    )
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    for name in ("Sample.parquet", "Sample.manifest.json"):
        (data / "processed" / name).write_bytes((settings.processed_dir / name).read_bytes())
    fused.to_parquet(data / "processed" / "Sample.system1b_scores.parquet")
    other = settings.model_copy(update={"data_dir": data})
    q = build_queue(other, DatasetName.SAMPLE, served_sample.frame, served_sample.index)
    assert q.triage_source == "fused[lgbm_all]"
    assert len(q) == int((fused["split"] == "test").sum())


def test_queue_needs_system1_scores(served_sample: ServedSample, tmp_path: Path) -> None:
    other = served_sample.settings.model_copy(update={"data_dir": tmp_path})
    with pytest.raises(DataNotFoundError, match="train-tabular"):
        build_queue(other, DatasetName.SAMPLE, served_sample.frame, served_sample.index)


def test_alerts_endpoint_filters_and_paginates(client: TestClient) -> None:
    full = client.get("/alerts", headers=ANALYST, params={"limit": 500}).json()
    assert full["ranking"] == "p_triage"
    assert full["total"] == len(full["items"]) == 200
    assert [i["rank"] for i in full["items"]] == list(range(1, 201))
    esc = client.get("/alerts", headers=ANALYST, params={"escalated_only": True}).json()
    assert esc["total"] == sum(i["escalated"] for i in full["items"])
    assert all(i["status"] == "open" for i in esc["items"])
    closed = client.get("/alerts", headers=ANALYST, params={"status": "auto_closed"}).json()
    assert closed["total"] == full["total"] - esc["total"]
    assert all(i["n_findings"] is None for i in closed["items"])
    page = client.get("/alerts", headers=ANALYST, params={"limit": 5, "offset": 5}).json()
    assert [i["rank"] for i in page["items"]] == [6, 7, 8, 9, 10]
    assert client.get("/alerts", headers=ANALYST, params={"limit": 0}).status_code == 422
    assert client.get("/alerts", headers=ANALYST, params={"status": "bogus"}).status_code == 422


def test_pattern_names_lists_fired_detectors() -> None:
    entry = QueueEntry(
        "CASE-1",
        1,
        pd.Timestamp("2022-09-09").to_pydatetime(),
        0.5,
        True,
        {"n_findings": 2, "n_fan_in": 1, "n_circular": 1, "n_fan_out": 0},
        None,
    )
    assert pattern_names(entry) == ["circular", "fan_in"]
    assert pattern_names(QueueEntry("CASE-2", 2, entry.timestamp, 0.1, False, None, None)) == []


# ---------------------------------------------------------------- case detail
def test_case_detail_has_evidence_but_no_labels(client: TestClient) -> None:
    case_id = _top_escalated(client)
    r = client.get(f"/cases/{case_id}", headers=ANALYST)
    assert r.status_code == 200
    d = r.json()
    assert d["case_id"] == case_id
    assert d["rank"] == 1
    assert d["alert"]["tx_id"] == int(case_id.removeprefix("CASE-"))
    assert d["subjects"] == [d["alert"]["payer"], d["alert"]["payee"]]
    assert d["status"] == "open"
    assert d["sar"] is None
    assert len(d["transactions"]) <= d["case_rows"]
    assert sum(e["count"] for e in d["edges"]) == d["case_rows"]
    assert not any(word in r.text for word in LABEL_WORDS)


def test_case_view_is_audited(client: TestClient, service: CaseService) -> None:
    case_id = _top_escalated(client)
    client.get(f"/cases/{case_id}", headers=ANALYST)
    (rec,) = service.audit_records()
    assert (rec.actor, rec.action, rec.case_id) == ("ana", "case.viewed", case_id)


def test_unknown_case_is_404(client: TestClient) -> None:
    assert client.get("/cases/CASE-123456789", headers=ANALYST).status_code == 404
    r = client.post(
        "/cases/CASE-123456789/actions",
        headers=ANALYST,
        json={"action": "dismiss", "note": "x" * 5},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------- SAR workflow
def test_sar_generate_edit_and_approve(client: TestClient, service: CaseService) -> None:
    case_id = _top_escalated(client)
    gen = client.post(f"/cases/{case_id}/sar", headers=ANALYST)
    assert gen.status_code == 201
    sar = gen.json()
    assert (sar["version"], sar["source"], sar["author"]) == (1, "template", "ana")
    assert sar["grounded"] is True
    assert sar["text"].startswith(REVIEW_STAMP)
    assert sar["text"].rstrip().endswith(REVIEW_STAMP)

    # The analyst removes the stamp and adds unsupported facts.
    edited = sar["text"].replace(REVIEW_STAMP, "") + (
        "\nAlso paid $123,456.78 on 2031-02-03 to 99999:FAKE0001."
    )
    saved = client.put(f"/cases/{case_id}/sar", headers=ANALYST, json={"text": edited}).json()
    assert saved["version"] == 2
    assert saved["source"] == "analyst"
    assert saved["text"].startswith(REVIEW_STAMP)
    assert saved["text"].count(REVIEW_STAMP) == 2
    assert {(i["kind"], i["value"]) for i in saved["issues"]} == {
        ("account", "99999:FAKE0001"),
        ("amount", "$123,456.78"),
        ("date", "2031-02-03"),
    }
    approve = {"action": "approve", "note": "ready to file"}
    refused = client.post(f"/cases/{case_id}/actions", headers=ANALYST, json=approve)
    assert refused.status_code == 409

    # Restoring the grounded text makes approval possible.
    fixed = client.put(f"/cases/{case_id}/sar", headers=ANALYST, json={"text": sar["text"]})
    assert fixed.json()["grounded"] is True
    assert (
        client.post(f"/cases/{case_id}/actions", headers=ANALYST, json=approve).status_code == 201
    )

    d = client.get(f"/cases/{case_id}", headers=ANALYST).json()
    assert d["status"] == "sar_approved"
    assert d["sar"]["version"] == 3
    assert [a["action"] for a in d["actions"]] == ["approve"]
    actions = [r.action for r in service.audit_records()]
    assert actions == [
        "sar.generated",
        "sar.edited",
        "sar.edited",
        "case.approved",
        "case.viewed",
    ]
    assert service.verify_audit() == 5


def test_edit_requires_a_generated_draft(client: TestClient) -> None:
    case_id = _top_escalated(client)
    r = client.put(f"/cases/{case_id}/sar", headers=ANALYST, json={"text": "anything"})
    assert r.status_code == 409


def test_approve_requires_a_draft(client: TestClient) -> None:
    case_id = _top_escalated(client)
    body = {"action": "approve", "note": "looks bad"}
    assert client.post(f"/cases/{case_id}/actions", headers=ANALYST, json=body).status_code == 409


def test_dismiss_and_escalate_update_status(client: TestClient) -> None:
    items = client.get("/alerts", headers=ANALYST, params={"limit": 3}).json()["items"]
    a, b = items[0]["case_id"], items[1]["case_id"]
    note = {"note": "benign payroll pattern"}
    assert (
        client.post(
            f"/cases/{a}/actions", headers=ANALYST, json={"action": "dismiss", **note}
        ).status_code
        == 201
    )
    assert (
        client.post(
            f"/cases/{b}/actions", headers=ADMIN, json={"action": "escalate", **note}
        ).status_code
        == 201
    )
    status = {
        i["case_id"]: i["status"] for i in client.get("/alerts", headers=ANALYST).json()["items"]
    }
    assert status[a] == "dismissed"
    assert status[b] == "escalated"
    dismissed = client.get("/alerts", headers=ANALYST, params={"status": "dismissed"}).json()
    assert [i["case_id"] for i in dismissed["items"]] == [a]


@pytest.mark.parametrize("note", ["", "  ", "x" * 2001])
def test_notes_are_bounded(client: TestClient, note: str) -> None:
    case_id = _top_escalated(client)
    r = client.post(
        f"/cases/{case_id}/actions", headers=ANALYST, json={"action": "dismiss", "note": note}
    )
    assert r.status_code == 422


def test_notes_never_reach_the_audit_log(client: TestClient, service: CaseService) -> None:
    case_id = _top_escalated(client)
    secret_note = "customer told me their PIN is 4321"
    client.post(
        f"/cases/{case_id}/actions",
        headers=ANALYST,
        json={"action": "escalate", "note": secret_note},
    )
    raw = service._audit.path.read_text(encoding="utf-8")
    assert "4321" not in raw
    assert f'"note_length": {len(secret_note)}' in raw


# ---------------------------------------------------------------- audit endpoint
def test_audit_endpoint_verifies_chain(client: TestClient, service: CaseService) -> None:
    case_id = _top_escalated(client)
    client.get(f"/cases/{case_id}", headers=ANALYST)
    client.post(
        f"/cases/{case_id}/actions", headers=ANALYST, json={"action": "escalate", "note": "check"}
    )
    view = client.get("/audit", headers=ADMIN).json()
    assert view["chain_ok"] is True
    assert view["chain_records"] == 2
    assert [r["action"] for r in view["records"]] == [
        "case.escalated",
        "case.viewed",
    ]  # newest first

    lines = service._audit.path.read_text(encoding="utf-8").splitlines()
    service._audit.path.write_text(
        "\n".join([lines[0].replace('"ana"', '"mallory"'), *lines[1:]]) + "\n", encoding="utf-8"
    )
    tampered = client.get("/audit", headers=ADMIN).json()
    assert tampered["chain_ok"] is False
    assert "record 1" in tampered["chain_error"]


# ---------------------------------------------------------------- store
def test_store_is_append_only(tmp_path: Path) -> None:
    store = CaseStore(tmp_path / "c.sqlite3")
    store.add_action("CASE-1", "ana", "dismiss", "fp")
    store.add_sar("CASE-1", text="t", source="template", issues=(), notes=(), author="ana")
    with sqlite3.connect(tmp_path / "c.sqlite3") as conn:
        for stmt in (
            "UPDATE case_actions SET note = 'x'",
            "DELETE FROM case_actions",
            "UPDATE sar_drafts SET text = 'x'",
            "DELETE FROM sar_drafts",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(stmt)
    assert store.actions("CASE-1")[0].note == "fp"


def test_store_rejects_unknown_actions(tmp_path: Path) -> None:
    store = CaseStore(tmp_path / "c.sqlite3")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_action("CASE-1", "ana", "delete_everything", "x")


def test_store_treats_input_as_data(tmp_path: Path) -> None:
    store = CaseStore(tmp_path / "c.sqlite3")
    hostile = "x'); DROP TABLE case_actions; --"
    store.add_action(hostile, "ana", "dismiss", hostile)
    assert store.actions(hostile)[0].note == hostile
    assert store.latest_actions() == {hostile: "dismiss"}


def test_concurrent_writes_keep_the_audit_chain(
    served_sample: ServedSample, tmp_path: Path
) -> None:
    service = make_service(served_sample, tmp_path)
    case_ids = [e.case_id for e in served_sample.queue.entries[:4]]
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for _ in range(5):
                service.act(case_ids[i % 4], "escalate", f"note {i}", f"user{i}")
        except BaseException as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert service.verify_audit() == 40
    assert sum(len(service.actions(c)) for c in case_ids) == 40


def test_concurrent_sar_saves_get_distinct_versions(tmp_path: Path) -> None:
    store = CaseStore(tmp_path / "c.sqlite3")

    def save() -> None:
        for _ in range(5):
            store.add_sar("CASE-1", text="t", source="analyst", issues=(), notes=(), author="a")

    threads = [threading.Thread(target=save) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    latest = store.latest_sar("CASE-1")
    assert latest is not None
    assert latest.version == 30


# ---------------------------------------------------------------- scoring
def _as_transaction(row: pd.Series) -> Transaction:
    return Transaction(
        timestamp=pd.Timestamp(row[Col.TIMESTAMP]).to_pydatetime(),
        from_bank=str(row[Col.FROM_BANK]),
        from_account=str(row[Col.FROM_ACCOUNT]),
        to_bank=str(row[Col.TO_BANK]),
        to_account=str(row[Col.TO_ACCOUNT]),
        amount_received=float(row[Col.AMOUNT_RECEIVED]),
        receiving_currency=str(row[Col.RECEIVING_CURRENCY]),  # type: ignore[arg-type]
        amount_paid=float(row[Col.AMOUNT_PAID]),
        payment_currency=str(row[Col.PAYMENT_CURRENCY]),  # type: ignore[arg-type]
        payment_format=str(row[Col.PAYMENT_FORMAT]),  # type: ignore[arg-type]
    )


def _last_of_its_minute(frame: pd.DataFrame, split: str) -> int:
    ts = frame[Col.TIMESTAMP]
    unique = ~ts.duplicated(keep=False)
    candidates = np.flatnonzero(
        unique.to_numpy() & (frame["split"].astype("string") == split).to_numpy()
    )
    return int(candidates[len(candidates) // 2])


def test_scoring_matches_the_training_feature_path(served_sample: ServedSample) -> None:
    """Train/serve consistency: re-scoring a known transaction reproduces its score."""
    s, frame = served_sample.settings, served_sample.frame
    i = _last_of_its_minute(frame, "test")
    scorer, reason = build_scorer(s, DatasetName.SAMPLE, frame.drop(index=frame.index[i]))
    assert scorer is not None, reason
    got = scorer.score(_as_transaction(frame.iloc[i]))

    model = LightGBMTriage.load(s.models_dir / "Sample" / "lightgbm_all.txt")
    expected = load_calibrator(s, DatasetName.SAMPLE).transform(
        model.predict(build_features(frame).iloc[[i]])
    )[0]
    assert got.p_tabular == pytest.approx(expected, abs=1e-12)
    assert got.history_rows == i


def test_scoring_never_sees_the_future(served_sample: ServedSample) -> None:
    """Leakage: rows after the transaction's timestamp cannot change its score."""
    frame = served_sample.frame
    i = _last_of_its_minute(frame, "val")
    tx = _as_transaction(frame.iloc[i])
    full = served_sample.scorer
    assert full is not None
    past_only, _ = build_scorer(served_sample.settings, DatasetName.SAMPLE, frame.iloc[: i + 1])
    assert past_only is not None
    # A burst of fabricated future payments by the same sender.
    future = pd.concat([frame.iloc[[i]]] * 50, ignore_index=True)
    future[Col.TIMESTAMP] = frame[Col.TIMESTAMP].iloc[-1] + timedelta(hours=1)
    future[Col.TX_ID] = np.arange(10**7, 10**7 + 50)
    poisoned, _ = build_scorer(
        served_sample.settings, DatasetName.SAMPLE, pd.concat([frame, future], ignore_index=True)
    )
    assert poisoned is not None
    scores = {full.score(tx).p_tabular, past_only.score(tx).p_tabular, poisoned.score(tx).p_tabular}
    assert len(scores) == 1


def test_scoring_does_not_persist(served_sample: ServedSample) -> None:
    scorer = served_sample.scorer
    assert scorer is not None
    tx = _as_transaction(served_sample.frame.iloc[-1])
    first, second = scorer.score(tx), scorer.score(tx)
    assert first.p_tabular == second.p_tabular
    assert first.history_rows == second.history_rows == len(served_sample.frame)


def test_scoring_handles_new_accounts(client: TestClient) -> None:
    body = {
        "timestamp": "2022-09-10T09:00:00",
        "from_bank": "00999",
        "from_account": "NEWACCT01",
        "to_bank": "00998",
        "to_account": "NEWACCT02",
        "amount_received": 9_900.0,
        "receiving_currency": "US Dollar",
        "amount_paid": 9_900.0,
        "payment_currency": "US Dollar",
        "payment_format": "Cash",
    }
    r = client.post("/score", headers=ANALYST, json=body)
    assert r.status_code == 200
    out = r.json()
    assert 0.0 <= out["p_tabular"] <= 1.0
    assert out["persisted"] is False
    assert out["model"] == "lightgbm[all]+isotonic"
    assert isinstance(out["rules_fired"], list)
    tz = client.post(
        "/score", headers=ANALYST, json={**body, "timestamp": "2022-09-10T09:00:00+02:00"}
    )
    assert tz.status_code == 422


def test_scoring_rejects_unknown_currency_rate(served_sample: ServedSample) -> None:
    scorer = served_sample.scorer
    assert scorer is not None
    lacking = TransactionScorer(
        served_sample.frame,
        {},  # no FX rates at all
        LightGBMTriage.load(served_sample.settings.models_dir / "Sample" / "lightgbm_all.txt"),
        load_calibrator(served_sample.settings, DatasetName.SAMPLE),
        0.5,
    )
    tx = _as_transaction(served_sample.frame.iloc[-1]).model_copy(
        update={"payment_currency": "Euro"}
    )
    with pytest.raises(DataValidationError, match="FX rate"):
        lacking.score(tx)


def test_scoring_is_refused_above_the_history_cap(
    served_sample: ServedSample, tmp_path: Path
) -> None:
    capped = served_sample.settings.model_copy(update={"api_score_max_history_rows": 100})
    scorer, reason = build_scorer(capped, DatasetName.SAMPLE, served_sample.frame)
    assert scorer is None
    assert reason is not None
    assert "online feature store" in reason
    service = CaseService(
        queue=served_sample.queue,
        index=served_sample.index,
        store=CaseStore(tmp_path / "c.sqlite3"),
        audit=AuditLog(tmp_path / "a.jsonl"),
        scoring_disabled_reason=reason,
    )
    c = TestClient(create_app(served_sample.settings, service=service))
    body = _as_transaction(served_sample.frame.iloc[-1]).model_dump(mode="json")
    r = c.post("/score", headers=ANALYST, json=body)
    assert r.status_code == 501
    assert "online feature store" in r.json()["detail"]


def test_scorer_unavailable_without_artifacts(served_sample: ServedSample, tmp_path: Path) -> None:
    other = served_sample.settings.model_copy(update={"models_dir": tmp_path})
    scorer, reason = build_scorer(other, DatasetName.SAMPLE, served_sample.frame)
    assert scorer is None
    assert reason is not None
    assert "train-tabular" in reason


def test_tampered_calibrator_is_refused(served_sample: ServedSample, tmp_path: Path) -> None:
    s = served_sample.settings
    models = tmp_path / "models" / "Sample"
    models.mkdir(parents=True)
    for f in (s.models_dir / "Sample").iterdir():
        (models / f.name).write_bytes(f.read_bytes())
    cal = models / "calibrator_all.json"
    cal.write_text(
        cal.read_text(encoding="utf-8").replace("isotonic", "isotonic "), encoding="utf-8"
    )
    other = s.model_copy(update={"models_dir": tmp_path / "models"})
    with pytest.raises(IntegrityError):
        load_calibrator(other, DatasetName.SAMPLE)


def test_case_id_format_is_shared_with_sar_and_audit() -> None:
    assert case_id_of(42) == "CASE-42"


def test_queue_lookup_helpers(served_sample: ServedSample) -> None:
    q: AlertQueue = served_sample.queue
    first = q.entries[0]
    assert q.get(first.case_id) is first
    assert q.rank_of(first.case_id) == 1
    assert q.get("CASE-0") is None
    assert q.rank_of("CASE-0") is None


# ---------------------------------------------------------------- reliability bands
def test_band_of_picks_the_containing_or_nearest_band() -> None:
    from aml_sentinel.api.queue import ReliabilityBand

    bands = (ReliabilityBand(0.0, 0.1, 10, 0.01), ReliabilityBand(0.2, 1.0, 10, 0.5))
    q = AlertQueue(DatasetName.SAMPLE, "x", 0.5, 0.05, (), bands)
    assert q.band_of(0.05) == bands[0]
    assert q.band_of(0.99) == bands[1]
    assert q.band_of(0.16) == bands[1]  # between bands: nearest edge
    assert AlertQueue(DatasetName.SAMPLE, "x", 0.5, 0.05, ()).band_of(0.5) is None


def test_reliability_is_attached_only_for_the_same_score(
    served_sample: ServedSample, tmp_path: Path
) -> None:
    s = served_sample.settings
    reports = tmp_path / "reports"
    (reports / "evaluation").mkdir(parents=True)
    band = {"lo": 0.0, "hi": 1.0, "count": 200, "mean_pred": 0.1, "frac_pos": 0.07}

    def queue_with(column: str) -> AlertQueue:
        doc = {"ablation": {"triage_column": column, "p_triage_reliability_test": [band]}}
        (reports / "evaluation" / "Sample.json").write_text(json.dumps(doc), encoding="utf-8")
        other = s.model_copy(update={"reports_dir": reports})
        return build_queue(other, DatasetName.SAMPLE, served_sample.frame, served_sample.index)

    assert queue_with("p__fused[lgbm_all]").reliability == ()  # different score: ignored
    matched = queue_with("p__lgbm_all (calibrated)")
    assert len(matched.reliability) == 1
    service = CaseService(
        queue=matched,
        index=served_sample.index,
        store=CaseStore(tmp_path / "c.sqlite3"),
        audit=AuditLog(tmp_path / "a.jsonl"),
    )
    c = TestClient(create_app(s, service=service))
    case_id = matched.entries[0].case_id
    rel = c.get(f"/cases/{case_id}", headers=ANALYST).json()["reliability"]
    assert rel["observed_rate"] == 0.07
    assert rel["alerts"] == 200
    assert rel["measured_on"].startswith("held-out test period")


def test_case_detail_without_evaluation_has_no_reliability(client: TestClient) -> None:
    case_id = _top_escalated(client)
    assert client.get(f"/cases/{case_id}", headers=ANALYST).json()["reliability"] is None
