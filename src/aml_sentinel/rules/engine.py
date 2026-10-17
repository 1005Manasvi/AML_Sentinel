"""Rule-based alerting baseline: the system every model must beat.

Three classic transaction-monitoring rules:

* **amount**: ``amount_paid_usd >= amount_threshold_usd``.
* **velocity**: the sender made ``>= velocity_max_tx`` payments in the trailing
  window ``(t - window, t]``, counting the current payment.
* **jurisdiction**: the sender or receiver bank is on a high-risk list. The IBM
  data has **no country field**, so the list is a seeded random sample of bank
  codes. The rule is artificial by construction and expected to be roughly
  uninformative; it is kept so the baseline has the familiar three-rule shape
  (ADR-008).

A transaction alerts if any enabled rule fires.

Leakage rules:
    Velocity only looks at the current and earlier rows. All fitted values
    (velocity percentile, bank list, tuned thresholds) use the training split only.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.evaluation.metrics import binary_metrics
from aml_sentinel.exceptions import ConfigurationError, DataValidationError

DEFAULT_AMOUNT_USD = 10_000.0  # mirrors the familiar US cash-reporting threshold
DEFAULT_WINDOW = timedelta(hours=24)
DEFAULT_VELOCITY_QUANTILE = 0.999
DEFAULT_HIGH_RISK_SHARE = 0.05

ALERT = "alert"
N_RULES = "n_rules_fired"


class Rule(StrEnum):
    """Names of the baseline rules (also the flag column names)."""

    AMOUNT = "rule_amount"
    VELOCITY = "rule_velocity"
    JURISDICTION = "rule_jurisdiction"


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Thresholds and switches for the rule engine.

    Attributes:
        amount_threshold_usd: Minimum USD amount that fires the amount rule.
        velocity_window: Trailing window for the velocity rule.
        velocity_max_tx: Payment count (in the window) that fires the velocity rule.
        high_risk_banks: Bank codes treated as high-risk (synthetic, see module doc).
        enabled: Rules that participate in the alert decision.
    """

    amount_threshold_usd: float = DEFAULT_AMOUNT_USD
    velocity_window: timedelta = DEFAULT_WINDOW
    velocity_max_tx: int = 20
    high_risk_banks: frozenset[str] = field(default_factory=frozenset)
    enabled: frozenset[Rule] = field(default_factory=lambda: frozenset(Rule))

    def __post_init__(self) -> None:
        """Validate thresholds.

        Raises:
            ConfigurationError: On non-positive thresholds, a sub-minute window or
                no enabled rules.
        """
        if not (self.amount_threshold_usd > 0 and math.isfinite(self.amount_threshold_usd)):
            raise ConfigurationError("amount_threshold_usd must be positive and finite")
        if self.velocity_window < timedelta(minutes=1):
            raise ConfigurationError("velocity_window must be at least one minute")
        if self.velocity_max_tx < 1:
            raise ConfigurationError("velocity_max_tx must be >= 1")
        if not self.enabled:
            raise ConfigurationError("at least one rule must be enabled")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (no pickle).

        Returns:
            Plain-type representation; inverse of :meth:`from_dict`.
        """
        return {
            "amount_threshold_usd": self.amount_threshold_usd,
            "velocity_window_minutes": int(self.velocity_window.total_seconds() // 60),
            "velocity_max_tx": self.velocity_max_tx,
            "high_risk_banks": sorted(self.high_risk_banks),
            "enabled": sorted(r.value for r in self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleConfig:
        """Rebuild a config from :meth:`to_dict` output.

        Args:
            data: Serialised config.

        Returns:
            The config.

        Raises:
            ConfigurationError: On missing keys or invalid values.
        """
        try:
            return cls(
                amount_threshold_usd=float(data["amount_threshold_usd"]),
                velocity_window=timedelta(minutes=int(data["velocity_window_minutes"])),
                velocity_max_tx=int(data["velocity_max_tx"]),
                high_risk_banks=frozenset(str(b) for b in data["high_risk_banks"]),
                enabled=frozenset(Rule(r) for r in data["enabled"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid rule config: {type(exc).__name__}") from None


def trailing_counts(
    frame: pd.DataFrame, *, key: str = Col.FROM_ID.value, window: timedelta = DEFAULT_WINDOW
) -> npt.NDArray[np.int64]:
    """Count each row's same-key rows in the trailing window ``(t - window, t]``.

    Rows in the same minute are ordered by their position in ``frame`` and only
    earlier ones are counted, so a row never sees anything after itself. Pass a
    frame sorted by time (as the loader produces) to keep that order meaningful.

    Args:
        frame: Transactions with ``timestamp`` and ``key`` columns.
        key: Column identifying the entity (sender account by default).
        window: Trailing window length (minute resolution).

    Returns:
        Counts aligned with ``frame`` rows; each count is at least 1.

    Raises:
        DataValidationError: If ``key`` has missing values.
    """
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    codes, _ = pd.factorize(frame[key], use_na_sentinel=True)
    if (codes < 0).any():
        raise DataValidationError(f"{key} has missing values")
    minutes = frame[Col.TIMESTAMP].to_numpy().astype("datetime64[m]").astype(np.int64)
    minutes = minutes - np.min(minutes)
    win = max(1, int(window.total_seconds() // 60))
    stride = int(np.max(minutes)) + win + 1  # keeps different keys' ranges disjoint

    order = np.lexsort((np.arange(n), minutes, codes))
    sorted_keys = codes[order].astype(np.int64) * stride + minutes[order]
    first_in_window = np.searchsorted(sorted_keys, sorted_keys - win, side="right")
    counts_sorted = np.arange(n, dtype=np.int64) - first_in_window + 1

    counts = np.empty(n, dtype=np.int64)
    counts[order] = counts_sorted
    return counts


def apply_rules(
    frame: pd.DataFrame, config: RuleConfig, velocity: npt.NDArray[np.int64]
) -> pd.DataFrame:
    """Evaluate every rule on every row.

    Args:
        frame: Transactions with ``amount_paid_usd``, ``from_bank`` and ``to_bank``.
        config: Rule thresholds.
        velocity: Output of :func:`trailing_counts` for the same rows.

    Returns:
        DataFrame (same index as ``frame``) with one bool column per :class:`Rule`
        (fired or not, regardless of ``enabled``), ``n_rules_fired`` over enabled
        rules, and ``alert``.

    Raises:
        DataValidationError: If ``velocity`` does not match ``frame`` in length.
    """
    if len(velocity) != len(frame):
        raise DataValidationError("velocity length does not match frame")
    banks = list(config.high_risk_banks)
    flags = pd.DataFrame(
        {
            Rule.AMOUNT.value: frame[Col.AMOUNT_PAID_USD].to_numpy() >= config.amount_threshold_usd,
            Rule.VELOCITY.value: velocity >= config.velocity_max_tx,
            Rule.JURISDICTION.value: (
                frame[Col.FROM_BANK].astype("string").isin(banks)
                | frame[Col.TO_BANK].astype("string").isin(banks)
            ).to_numpy(dtype=bool),
        },
        index=frame.index,
    )
    enabled = [r.value for r in Rule if r in config.enabled]
    flags[N_RULES] = flags[enabled].sum(axis=1).astype(np.int8)
    flags[ALERT] = flags[N_RULES] > 0
    return flags


def select_high_risk_banks(
    train: pd.DataFrame, *, share: float = DEFAULT_HIGH_RISK_SHARE, seed: int
) -> frozenset[str]:
    """Pick a seeded random subset of training-period bank codes as "high-risk".

    Selection ignores labels and volumes on purpose: there is no real jurisdiction
    information, so any smarter choice would be invented.

    Args:
        train: Training-split transactions.
        share: Fraction of distinct banks to select (at least one).
        seed: RNG seed.

    Returns:
        The selected bank codes.

    Raises:
        ConfigurationError: If ``share`` is not in ``(0, 1]``.
    """
    if not 0 < share <= 1:
        raise ConfigurationError("share must be in (0, 1]")
    banks = sorted(
        set(train[Col.FROM_BANK].astype("string").dropna())
        | set(train[Col.TO_BANK].astype("string").dropna())
    )
    if not banks:
        return frozenset()
    k = max(1, round(share * len(banks)))
    chosen = np.random.default_rng(seed).choice(len(banks), size=k, replace=False)
    return frozenset(banks[int(i)] for i in chosen)


def sender_day_counts(
    frame: pd.DataFrame, *, key: str = Col.FROM_ID.value
) -> npt.NDArray[np.int64]:
    """Count payments per (sender, calendar day): one value per active sender-day.

    Velocity thresholds must come from this *entity-level* distribution. A
    percentile over transaction rows is dominated by a few hub accounts that send
    tens of thousands of payments a day, which pushes the threshold so high that
    the rule never fires (measured on HI-Small: p99.9 per row was 25,705/day versus
    24/day per sender-day).

    Args:
        frame: Transactions with ``timestamp`` and ``key``.
        key: Entity column.

    Returns:
        Payment counts, one per sender-day.
    """
    days = frame[Col.TIMESTAMP].dt.normalize()
    counts = frame.groupby([frame[key], days], observed=True, sort=False).size()
    return np.asarray(counts.to_numpy(), dtype=np.int64)


def fit_static(
    train: pd.DataFrame,
    high_risk_banks: frozenset[str],
    *,
    quantile: float = DEFAULT_VELOCITY_QUANTILE,
) -> RuleConfig:
    """Build the textbook baseline: fixed 10k USD amount, velocity at a train percentile.

    Args:
        train: Training-split transactions only.
        high_risk_banks: Output of :func:`select_high_risk_banks`.
        quantile: Percentile of sender-day payment counts that sets the velocity
            threshold (no labels used).

    Returns:
        The static rule config.

    Raises:
        DataValidationError: If ``train`` is empty.
    """
    if len(train) == 0:
        raise DataValidationError("no training rows to fit velocity threshold")
    threshold = math.ceil(float(np.quantile(sender_day_counts(train), quantile)))
    return RuleConfig(velocity_max_tx=max(1, threshold), high_risk_banks=high_risk_banks)


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Best config found by :func:`tune` and the train F1 it achieved.

    Attributes:
        config: Selected config.
        train_f1: F1 on the training split (``None`` if undefined).
        n_candidates: Number of configs evaluated.
    """

    config: RuleConfig
    train_f1: float | None
    n_candidates: int


def tune(
    train: pd.DataFrame,
    train_velocity: npt.NDArray[np.int64],
    high_risk_banks: frozenset[str],
    *,
    amount_grid: Sequence[float] = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 1e6),
    velocity_grid: Sequence[int] = (2, 3, 5, 8, 10, 15, 20, 25, 50, 100),
) -> TuningResult:
    """Grid-search thresholds and rule subsets for the best F1 on **train labels only**.

    Exists so the baseline is not a strawman: later models are compared with the
    strongest rule configuration that the training data supports.

    Args:
        train: Training-split transactions (must include ``is_laundering``).
        train_velocity: Trailing counts of the same rows.
        high_risk_banks: Bank list used by the jurisdiction rule.
        amount_grid: Candidate USD thresholds.
        velocity_grid: Candidate payment counts per trailing window (absolute, so
            small fan-out bursts are reachable; see :func:`sender_day_counts`).

    Returns:
        The best config (ties broken by the first one found, in grid order).

    Raises:
        DataValidationError: If ``train`` is empty.
    """
    if len(train) == 0:
        raise DataValidationError("cannot tune on an empty training split")
    if not amount_grid or not velocity_grid:
        raise ConfigurationError("tuning grids must not be empty")
    y = train[Col.IS_LAUNDERING].to_numpy() == 1
    vel_grid = sorted({int(v) for v in velocity_grid if v >= 1})
    subsets = [frozenset(c) for k in range(1, 4) for c in itertools.combinations(Rule, k)]

    # Each rule's flags are computed once per threshold; the loop only ORs them.
    amount_usd = train[Col.AMOUNT_PAID_USD].to_numpy()
    amount_flags = {a: amount_usd >= a for a in amount_grid}
    velocity_flags = {v: train_velocity >= v for v in vel_grid}
    banks = list(high_risk_banks)
    jurisdiction = (
        train[Col.FROM_BANK].astype("string").isin(banks)
        | train[Col.TO_BANK].astype("string").isin(banks)
    ).to_numpy(dtype=bool)
    none = np.zeros(len(train), dtype=bool)

    best_score, best_cfg, n = -1.0, None, 0
    for amount, vel, enabled in itertools.product(amount_grid, vel_grid, subsets):
        alert = (
            (amount_flags[amount] if Rule.AMOUNT in enabled else none)
            | (velocity_flags[vel] if Rule.VELOCITY in enabled else none)
            | (jurisdiction if Rule.JURISDICTION in enabled else none)
        )
        f1 = binary_metrics(y, alert).f1
        n += 1
        score = -1.0 if f1 is None else f1
        if best_cfg is None or score > best_score:
            best_score = score
            best_cfg = RuleConfig(
                amount_threshold_usd=amount,
                velocity_max_tx=vel,
                high_risk_banks=high_risk_banks,
                enabled=enabled,
            )
    if best_cfg is None:  # pragma: no cover - grids are validated non-empty above
        raise ConfigurationError("no candidate configurations")
    return TuningResult(best_cfg, None if best_score < 0 else best_score, n)
