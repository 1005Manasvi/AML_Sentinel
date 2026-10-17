from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import ConfigurationError, DataValidationError
from aml_sentinel.rules.engine import (
    ALERT,
    N_RULES,
    Rule,
    RuleConfig,
    apply_rules,
    fit_static,
    select_high_risk_banks,
    sender_day_counts,
    trailing_counts,
    tune,
)

T0 = datetime(2022, 9, 1)


def _frame(rows: list[tuple[str, int]], **extra: list[object]) -> pd.DataFrame:
    """rows: (sender, minutes after T0)."""
    return pd.DataFrame(
        {
            Col.FROM_ID.value: [r[0] for r in rows],
            Col.TIMESTAMP.value: pd.to_datetime([T0 + timedelta(minutes=r[1]) for r in rows]),
            **extra,
        }
    )


# ----------------------------------------------------------------- velocity
def test_trailing_counts_hand_example() -> None:
    df = _frame([("A", 0), ("A", 10), ("B", 10), ("A", 10), ("A", 70)])
    counts = trailing_counts(df, window=timedelta(minutes=60))
    # A@70: window (10, 70] excludes the two A@10 rows (boundary is exclusive).
    assert counts.tolist() == [1, 2, 1, 3, 1]


def test_trailing_counts_edge_cases() -> None:
    assert trailing_counts(_frame([])).size == 0
    df = _frame([("A", 0)])
    df.loc[0, Col.FROM_ID.value] = None
    with pytest.raises(DataValidationError, match="missing"):
        trailing_counts(df)


events = st.lists(
    st.tuples(st.sampled_from(["A", "B", "C"]), st.integers(0, 3 * 24 * 60)), max_size=40
)


def _brute_force(rows: list[tuple[str, int]], window_min: int) -> list[int]:
    return [
        sum(
            1
            for j, (k2, t2) in enumerate(rows)
            if k2 == k and t - window_min < t2 <= t and (t2 < t or j <= i)
        )
        for i, (k, t) in enumerate(rows)
    ]


@settings(max_examples=200, deadline=None)
@given(events, st.integers(1, 2 * 24 * 60))
def test_trailing_counts_match_brute_force(rows: list[tuple[str, int]], window_min: int) -> None:
    rows = sorted(rows, key=lambda r: r[1])
    got = trailing_counts(_frame(rows), window=timedelta(minutes=window_min))
    assert got.tolist() == _brute_force(rows, window_min)


@settings(max_examples=150, deadline=None)
@given(events, events)
def test_future_rows_never_change_past_counts(
    past: list[tuple[str, int]], future: list[tuple[str, int]]
) -> None:
    """No lookahead: appending later activity leaves earlier rows' velocity unchanged."""
    past = sorted(past, key=lambda r: r[1])
    horizon = past[-1][1] + 1 if past else 0
    future = sorted(((k, horizon + t) for k, t in future), key=lambda r: r[1])
    alone = trailing_counts(_frame(past))
    combined = trailing_counts(_frame(past + future))
    assert combined[: len(past)].tolist() == alone.tolist()


# ----------------------------------------------------------------- rules
def _txs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            Col.AMOUNT_PAID_USD.value: [50.0, 20_000.0, 50.0, 50.0],
            Col.FROM_BANK.value: pd.Categorical(["001", "001", "099", "001"]),
            Col.TO_BANK.value: pd.Categorical(["002", "002", "002", "099"]),
            Col.IS_LAUNDERING.value: [0, 1, 0, 1],
        }
    )


def test_apply_rules_flags_each_rule() -> None:
    cfg = RuleConfig(velocity_max_tx=5, high_risk_banks=frozenset({"099"}))
    flags = apply_rules(_txs(), cfg, np.array([1, 1, 6, 1], dtype=np.int64))
    assert flags[Rule.AMOUNT.value].tolist() == [False, True, False, False]
    assert flags[Rule.VELOCITY.value].tolist() == [False, False, True, False]
    assert flags[Rule.JURISDICTION.value].tolist() == [False, False, True, True]
    assert flags[N_RULES].tolist() == [0, 1, 2, 1]
    assert flags[ALERT].tolist() == [False, True, True, True]


def test_disabled_rules_do_not_alert() -> None:
    cfg = RuleConfig(
        velocity_max_tx=5, high_risk_banks=frozenset({"099"}), enabled=frozenset({Rule.AMOUNT})
    )
    flags = apply_rules(_txs(), cfg, np.array([1, 1, 6, 1], dtype=np.int64))
    assert flags[ALERT].tolist() == [False, True, False, False]
    assert flags[Rule.VELOCITY.value].tolist() == [False, False, True, False]  # still reported


def test_apply_rules_length_mismatch() -> None:
    with pytest.raises(DataValidationError):
        apply_rules(_txs(), RuleConfig(), np.array([1], dtype=np.int64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amount_threshold_usd": 0},
        {"amount_threshold_usd": float("inf")},
        {"velocity_window": timedelta(seconds=30)},
        {"velocity_max_tx": 0},
        {"enabled": frozenset()},
    ],
)
def test_rule_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        RuleConfig(**kwargs)  # type: ignore[arg-type]


def test_rule_config_json_roundtrip() -> None:
    cfg = RuleConfig(
        amount_threshold_usd=5e3,
        velocity_max_tx=7,
        high_risk_banks=frozenset({"001", "0420"}),
        enabled=frozenset({Rule.AMOUNT, Rule.VELOCITY}),
    )
    assert RuleConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ConfigurationError):
        RuleConfig.from_dict({"amount_threshold_usd": 1})
    with pytest.raises(ConfigurationError):
        RuleConfig.from_dict({**cfg.to_dict(), "enabled": ["rule_teleport"]})


def test_high_risk_banks_seeded_and_sized() -> None:
    train = pd.DataFrame(
        {
            Col.FROM_BANK.value: [f"{i:03d}" for i in range(40)],
            Col.TO_BANK.value: [f"{i:03d}" for i in range(40, 80)],
        }
    )
    a = select_high_risk_banks(train, seed=1)
    assert a == select_high_risk_banks(train, seed=1)
    assert len(a) == 4  # 5% of 80
    assert a != select_high_risk_banks(train, seed=2)
    assert len(select_high_risk_banks(train.head(1), seed=1)) == 1  # at least one
    assert select_high_risk_banks(train.head(0), seed=1) == frozenset()
    with pytest.raises(ConfigurationError):
        select_high_risk_banks(train, share=0, seed=1)


def test_fit_static_uses_sender_day_quantile() -> None:
    # One hub sends 500 payments on day 0; 99 other senders send 1-3 per day.
    rows = [("HUB", m) for m in range(500)]
    rows += [(f"S{i}", d * 1440 + j) for i in range(99) for d in range(2) for j in range(1 + i % 3)]
    train = _frame(rows)
    assert sorted(sender_day_counts(train).tolist())[-1] == 500
    cfg = fit_static(train, frozenset(), quantile=0.99)
    # Entity-level p99 stays near ordinary behaviour; a row-level p99 would be ~500.
    assert cfg.velocity_max_tx <= 10
    assert cfg.amount_threshold_usd == 10_000
    with pytest.raises(DataValidationError):
        fit_static(train.head(0), frozenset())


def test_tune_finds_the_informative_rule() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.random(n) < 0.05
    train = pd.DataFrame(
        {
            # Laundering is exactly the large amounts; velocity and banks are noise.
            Col.AMOUNT_PAID_USD.value: np.where(y, 30_000.0, 500.0),
            Col.FROM_BANK.value: rng.choice(["001", "002"], n),
            Col.TO_BANK.value: rng.choice(["001", "002"], n),
            Col.IS_LAUNDERING.value: y.astype(int),
        }
    )
    velocity = rng.integers(1, 50, n).astype(np.int64)
    result = tune(train, velocity, frozenset({"002"}))
    assert result.config.enabled == frozenset({Rule.AMOUNT})
    assert result.train_f1 == pytest.approx(1.0)
    assert result.n_candidates > 50


def test_tune_rejects_empty_inputs() -> None:
    empty = pd.DataFrame(columns=[c.value for c in Col])
    with pytest.raises(DataValidationError):
        tune(empty, np.array([], dtype=np.int64), frozenset())
    with pytest.raises(ConfigurationError):
        tune(_txs(), np.ones(4, dtype=np.int64), frozenset(), amount_grid=())


def test_tune_reports_none_when_no_config_has_f1() -> None:
    no_positives = _txs().assign(**{Col.IS_LAUNDERING.value: 0})
    assert tune(no_positives, np.ones(4, dtype=np.int64), frozenset()).train_f1 is None
