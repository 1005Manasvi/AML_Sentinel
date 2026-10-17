"""Planted-pattern tests for the System 2 detectors and the past-only case builder."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.system2.cases import Case, CaseConfig, CaseIndex
from aml_sentinel.system2.detectors import (
    DetectorConfig,
    FindingKind,
    detect_circular,
    detect_fans,
    detect_scatter_gather,
    detect_structuring,
    run_detectors,
)

T0 = datetime(2022, 9, 5)
CFG = DetectorConfig()


def _edges(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """rows: (from, to, hours after T0, usd)."""
    return pd.DataFrame(
        {
            Col.TX_ID.value: range(len(rows)),
            Col.TIMESTAMP.value: pd.to_datetime([T0 + timedelta(hours=r[2]) for r in rows]),
            Col.FROM_ID.value: pd.array([r[0] for r in rows], dtype="string"),
            Col.TO_ID.value: pd.array([r[1] for r in rows], dtype="string"),
            Col.AMOUNT_PAID_USD.value: [r[3] for r in rows],
            Col.PAYMENT_FORMAT.value: ["ACH"] * len(rows),
        }
    )


def _case(rows: list[tuple[str, str, float, float]], focal: tuple[str, str]) -> Case:
    return Case(alert_tx_id=0, focal=focal, edges=_edges(rows), truncated=False)


# Background noise that must not trigger anything on its own.
NOISE = [("X1", "X2", 1, 50.0), ("X2", "X3", 2, 70.0), ("A", "X9", 3, 30.0)]


# ----------------------------------------------------------------- circular
def test_planted_cycle_is_found() -> None:
    rows = [("A", "B", 1, 9000), ("B", "C", 2, 8800), ("C", "D", 3, 8600), ("D", "A", 4, 8400)]
    found = detect_circular(_case(rows + NOISE, ("A", "B")), CFG)
    assert len(found) == 1
    assert found[0].kind is FindingKind.CIRCULAR
    assert set(found[0].accounts) == {"A", "B", "C", "D"}
    assert found[0].tx_ids == (0, 1, 2, 3)


def test_cycle_must_respect_time_and_amounts() -> None:
    # Each hop happens *before* the previous one, so money cannot have flowed around.
    backwards = [
        ("A", "B", 4, 9000),
        ("B", "C", 3, 9000),
        ("C", "A", 2, 9000),
        ("A", "B", 1.5, 9000),
    ]
    # Rotation start A->B@4 fails, B->C@3 fails ... but the A->B@1.5 makes B->C@3 reachable,
    # then C->A must be >=3: it is @2, so no valid ordering exists.
    assert detect_circular(_case(backwards, ("A", "B")), CFG) == []
    tiny_return = [("A", "B", 1, 9000), ("B", "A", 2, 10)]
    assert detect_circular(_case(tiny_return, ("A", "B")), CFG) == []


def test_cycle_must_touch_the_alert() -> None:
    rows = [("P", "Q", 1, 500), ("Q", "R", 2, 500), ("R", "P", 3, 500), ("A", "B", 1, 9)]
    assert detect_circular(_case(rows, ("A", "B")), CFG) == []


# ----------------------------------------------------------------- structuring
def test_planted_structuring_is_found() -> None:
    rows = [(f"S{i}", "A", i, 9_100 + 50 * i) for i in range(4)]
    found = detect_structuring(_case(rows + NOISE, ("S0", "A")), CFG)
    assert [f.kind for f in found] == [FindingKind.STRUCTURING]
    assert found[0].facts["distinct_senders"] == 4


def test_large_or_small_deposits_are_not_structuring() -> None:
    rows = [(f"S{i}", "A", i, 12_000) for i in range(4)] + [
        (f"T{i}", "A", i, 900) for i in range(4)
    ]
    assert detect_structuring(_case(rows, ("S0", "A")), CFG) == []
    two = [("S0", "A", 1, 9_500), ("S1", "A", 2, 9_600)]
    assert detect_structuring(_case(two, ("S0", "A")), CFG) == []


# ----------------------------------------------------------------- fans
def test_planted_fan_in_and_fan_out() -> None:
    fan_in = [(f"P{i}", "H", i, 4000) for i in range(6)]
    fan_out = [("H", f"Q{i}", 10 + i, 3000) for i in range(5)]
    kinds = {f.kind: f for f in detect_fans(_case(fan_in + fan_out, ("P0", "H")), CFG)}
    assert set(kinds) == {FindingKind.FAN_IN, FindingKind.FAN_OUT}
    assert kinds[FindingKind.FAN_IN].facts["counterparties"] == 6
    assert kinds[FindingKind.FAN_IN].facts["amount_cv"] == pytest.approx(0.0)


def test_small_fans_and_self_transfers_ignored() -> None:
    rows = [(f"P{i}", "H", i, 4000) for i in range(4)] + [("H", "H", 9, 1)] * 3
    assert detect_fans(_case(rows, ("P0", "H")), CFG) == []


# ----------------------------------------------------------------- scatter / gather
def test_planted_scatter_gather() -> None:
    out = [("S", f"M{i}", 1 + i, 8000) for i in range(3)]
    back = [(f"M{i}", "D", 10 + i, 7600) for i in range(3)]
    found = detect_scatter_gather(_case(out + back + NOISE, ("S", "M0")), CFG)
    sg = [f for f in found if f.kind is FindingKind.SCATTER_GATHER]
    assert len(sg) == 1
    assert sg[0].facts["intermediaries"] == 3
    assert set(sg[0].accounts) == {"S", "D", "M0", "M1", "M2"}


def test_scatter_gather_needs_onward_flow_in_time() -> None:
    out = [("S", f"M{i}", 10 + i, 8000) for i in range(3)]
    early = [(f"M{i}", "D", 1 + i, 7600) for i in range(3)]  # paid out BEFORE receiving
    found = detect_scatter_gather(_case(out + early, ("S", "M0")), CFG)
    assert not [f for f in found if f.kind is FindingKind.SCATTER_GATHER]


def test_planted_gather_scatter() -> None:
    gather = [(f"P{i}", "H", i, 5000) for i in range(3)]
    scatter = [("H", f"Q{i}", 10 + i, 4800) for i in range(3)]
    found = detect_scatter_gather(_case(gather + scatter, ("P0", "H")), CFG)
    gs = [f for f in found if f.kind is FindingKind.GATHER_SCATTER]
    assert len(gs) == 1
    assert gs[0].facts["payers"] == 3
    assert gs[0].facts["payees"] == 3


def test_benign_payroll_noise_is_quiet_for_cycles_and_structuring() -> None:
    payroll = [("EMP", f"W{i}", 1, 3_500 + 10 * i) for i in range(12)]
    kinds = {f.kind for f in run_detectors(_case(payroll, ("EMP", "W0")))}
    # A payroll run *is* a fan-out shape: the detector reports it, and System 2
    # evaluation measures how often such shapes are benign.
    assert kinds == {FindingKind.FAN_OUT}


# ----------------------------------------------------------------- case builder
def _frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    df = _edges(rows)
    df[Col.FROM_ID.value] = df[Col.FROM_ID.value].astype("category")
    df[Col.TO_ID.value] = df[Col.TO_ID.value].astype("category")
    return df.sort_values(Col.TIMESTAMP.value, kind="stable").reset_index(drop=True)


def test_case_is_past_only_and_bounded() -> None:
    rows = [
        ("A", "B", 0, 100),  # tx 0: inside lookback
        ("B", "C", 1, 100),  # tx 1: 2nd hop via B
        ("C", "D", 2, 100),  # tx 2: 3rd hop, outside hops=2 from {A, B}? C is hop-1 of B
        ("A", "B", 5, 9000),  # tx 3: the alert
        ("B", "Z", 6, 100),  # tx 4: AFTER the alert -> must never appear
        ("A", "B", -100, 1),  # tx 5: before the lookback window
    ]
    index = CaseIndex(_frame(rows))
    case = index.build(3, CaseConfig(lookback=timedelta(hours=72), hops=2))
    ids = set(case.edges[Col.TX_ID.value])
    assert 3 in ids
    assert 4 not in ids  # no lookahead
    assert 5 not in ids  # outside lookback
    assert {0, 1} <= ids
    assert case.focal == ("A", "B")
    assert case.edges[Col.TIMESTAMP.value].is_monotonic_increasing


def test_case_caps_hubs_and_records_truncation() -> None:
    rows = [(f"P{i}", "HUB", i * 0.01, 10) for i in range(200)] + [("A", "HUB", 3, 9000)]
    index = CaseIndex(_frame(rows))
    case = index.build(200, CaseConfig(max_rows_per_account=20, max_rows=500))
    assert case.truncated
    assert len(case.edges) <= 1 + 20 + 20
    capped = index.build(200, CaseConfig(max_rows_per_account=500, max_rows=30))
    assert len(capped.edges) <= 30
    assert 200 in set(capped.edges[Col.TX_ID.value])


def test_case_index_validation() -> None:
    df = _frame([("A", "B", 1, 1)])
    with pytest.raises(DataValidationError, match="unknown"):
        CaseIndex(df).build(999)
    with pytest.raises(DataValidationError, match="missing"):
        CaseIndex(df.drop(columns=[Col.AMOUNT_PAID_USD.value]))
    unsorted = pd.concat([_frame([("A", "B", 5, 1)]), _frame([("A", "B", 1, 1)])])
    with pytest.raises(DataValidationError, match="sorted"):
        CaseIndex(unsorted)


def test_investigation_horizon_is_opt_in() -> None:
    rows = [("A", "B", 0, 100), ("A", "B", 5, 9000), ("B", "Z", 6, 100), ("B", "Y", 40, 100)]
    index = CaseIndex(_frame(rows))
    strict = set(index.build(1).edges[Col.TX_ID.value])
    later = set(index.build(1, CaseConfig(horizon=timedelta(hours=24))).edges[Col.TX_ID.value])
    assert strict == {0, 1}
    assert later == {0, 1, 2}  # +1h row visible next day, +35h row still not
