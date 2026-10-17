from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.features.builder import (
    CROSS_CURRENCY,
    PAYMENT_FORMAT,
    PAYMENT_FORMAT_ORDER,
    SHORTCUT_FEATURES,
    build_features,
    feature_columns,
)
from aml_sentinel.features.graph_features import snapshot_features
from aml_sentinel.features.temporal import trailing_aggregates

T0 = datetime(2022, 9, 1)


def _tx(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """rows: (from_id, to_id, minutes after T0, usd)."""
    return pd.DataFrame(
        {
            Col.FROM_ID.value: [r[0] for r in rows],
            Col.TO_ID.value: [r[1] for r in rows],
            Col.TIMESTAMP.value: pd.to_datetime([T0 + timedelta(minutes=r[2]) for r in rows]),
            Col.AMOUNT_PAID_USD.value: [r[3] for r in rows],
        }
    )


def _full(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """A random but schema-complete processed frame, sorted by time."""
    rng = np.random.default_rng(seed)
    accounts = [f"00{i % 7}:A{i}" for i in range(25)]
    src = rng.choice(accounts, n_rows)
    dst = rng.choice(accounts, n_rows)
    minutes = np.sort(rng.integers(0, 5 * 1440, n_rows))
    cur = rng.choice(["US Dollar", "Euro"], n_rows)
    return pd.DataFrame(
        {
            Col.TIMESTAMP.value: pd.to_datetime([T0 + timedelta(minutes=int(m)) for m in minutes]),
            Col.FROM_ID.value: src,
            Col.TO_ID.value: dst,
            Col.FROM_BANK.value: [s.split(":")[0] for s in src],
            Col.TO_BANK.value: [d.split(":")[0] for d in dst],
            Col.AMOUNT_PAID.value: np.round(rng.lognormal(6, 1, n_rows), 2),
            Col.AMOUNT_PAID_USD.value: rng.lognormal(6, 1, n_rows),
            Col.PAYMENT_CURRENCY.value: cur,
            Col.RECEIVING_CURRENCY.value: np.where(rng.random(n_rows) < 0.1, "Yen", cur),
            Col.PAYMENT_FORMAT.value: rng.choice(sorted(PAYMENT_FORMAT_ORDER), n_rows),
        }
    )


# ----------------------------------------------------------------- leakage
@pytest.mark.parametrize("cutoff_day", [1, 2, 3, 4])
def test_features_never_depend_on_later_rows(cutoff_day: int) -> None:
    """Leakage test: deleting every row after a cutoff must not change earlier features."""
    full = _full()
    cutoff = pd.Timestamp(T0 + timedelta(days=cutoff_day, hours=7))
    past = full[full[Col.TIMESTAMP.value] < cutoff].reset_index(drop=True)
    on_full = build_features(full).iloc[: len(past)].reset_index(drop=True)
    on_past = build_features(past)
    pd.testing.assert_frame_equal(on_full, on_past)


def test_snapshot_uses_only_previous_days() -> None:
    # Day 0: a 3-cycle A->B->C->A. Day 1: A pays D.
    df = _tx(
        [("A", "B", 10, 100), ("B", "C", 20, 100), ("C", "A", 30, 100), ("A", "D", 1440 + 5, 50)]
    )
    snap = snapshot_features(df)
    assert (snap.iloc[:3]["src_has_history"] == 0).all()  # nothing before day 0
    assert snap.iloc[:3]["src_cycle3"].sum() == 0
    last = snap.iloc[3]
    assert last["src_cycle3"] == 1
    assert last["src_out_degree"] == 1
    assert last["src_in_degree"] == 1
    assert last["src_flow_ratio"] == pytest.approx(0.5)
    assert last["src_age_days"] == pytest.approx((1440 + 5 - 10) / 1440)
    assert last["dst_has_history"] == 0  # D is new


def test_snapshot_reciprocal_and_self_loops() -> None:
    df = _tx([("A", "B", 0, 1), ("B", "A", 1, 1), ("A", "A", 2, 9), ("A", "B", 1441, 1)])
    last = snapshot_features(df).iloc[3]
    assert last["src_reciprocal"] == 1
    assert last["src_out_tx"] == 1  # self-transfer excluded from flow stats
    assert last["src_has_history"] == 1


# ----------------------------------------------------------------- temporal
events = st.lists(
    st.tuples(st.sampled_from(["A", "B"]), st.integers(0, 2000), st.floats(0, 1e6)),
    max_size=30,
)


@settings(max_examples=150, deadline=None)
@given(events, st.integers(1, 1500))
def test_trailing_sums_match_brute_force(rows: list[tuple[str, int, float]], win: int) -> None:
    rows = sorted(rows, key=lambda r: r[1])
    df = _tx([(k, "X", m, v) for k, m, v in rows])
    agg = trailing_aggregates(
        df, key=Col.FROM_ID.value, window=timedelta(minutes=win), values=Col.AMOUNT_PAID_USD.value
    )
    for i, (k, t, _) in enumerate(rows):
        inside = [
            v
            for j, (k2, t2, v) in enumerate(rows)
            if k2 == k and t - win < t2 <= t and (t2 < t or j <= i)
        ]
        assert agg.counts[i] == len(inside)
        assert agg.sums[i] == pytest.approx(sum(inside), rel=1e-9, abs=1e-6)


def test_trailing_aggregates_edge_cases() -> None:
    empty = trailing_aggregates(_tx([]), key=Col.FROM_ID.value, window=timedelta(hours=1))
    assert empty.counts.size == 0
    df = _tx([("A", "B", 0, 1.0)])
    df.loc[0, Col.FROM_ID.value] = None
    with pytest.raises(DataValidationError):
        trailing_aggregates(df, key=Col.FROM_ID.value, window=timedelta(hours=1))


# ----------------------------------------------------------------- builder
def test_row_features_and_codes() -> None:
    df = _full(50)
    df.loc[0, Col.AMOUNT_PAID.value] = 500.0
    df.loc[0, Col.PAYMENT_FORMAT.value] = "Wire"
    feats = build_features(df)
    assert feats.loc[0, "round_amount"] == 1
    assert feats.loc[0, PAYMENT_FORMAT] == PAYMENT_FORMAT_ORDER.index("Wire")
    expected_cross = df[Col.PAYMENT_CURRENCY.value] != df[Col.RECEIVING_CURRENCY.value]
    assert (feats[CROSS_CURRENCY].astype(bool) == expected_cross).all()
    assert all(dt == np.float32 for dt in feats.dtypes)
    assert list(feats.columns) == feature_columns(include_shortcuts=True)


def test_shortcut_variant_drops_artifacts() -> None:
    cols = feature_columns(include_shortcuts=False)
    assert not set(SHORTCUT_FEATURES) & set(cols)
    assert len(cols) == len(feature_columns(include_shortcuts=True)) - len(SHORTCUT_FEATURES)


def test_builder_input_errors() -> None:
    df = _full(20)
    with pytest.raises(DataValidationError, match="missing"):
        build_features(df.drop(columns=[Col.AMOUNT_PAID_USD.value]))
    with pytest.raises(DataValidationError, match="sorted"):
        build_features(df.iloc[::-1])
