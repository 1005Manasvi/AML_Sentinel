"""Generate a small synthetic AML dataset in the IBM AMLworld file format.

The bundled sample (``data/sample/``) lets the demo, tests and CI run without
Kaggle access. It comes from this simple, seeded simulator and copies nothing
from IBM. Because it uses the same file layout, every loader and parser code path
is exercised unchanged.

Realism features, chosen so that downstream models cannot just learn "shape means
laundering":

* Benign look-alikes: payroll fan-outs, merchant fan-ins and roommate rent cycles,
  all labelled 0.
* Undocumented attempts: a fraction of laundering attempts is labelled 1 in the
  transactions file but left out of the patterns file, as in IBM's data.
* Amount noise and fee-like decrements along laundering chains, plus some
  cross-currency hops.

Security note:
    Output is fully synthetic. Account IDs are random hex strings and bank codes are
    random digits; no real identifiers are involved.
"""

from __future__ import annotations

import csv
import io
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np

from aml_sentinel.data.patterns import Typology
from aml_sentinel.data.schema import RAW_HEADER, TIMESTAMP_FORMAT, USD

# Fixed, round illustrative rates (USD per unit). They are not IBM's.
SAMPLE_FX: Final[dict[str, float]] = {
    USD: 1.0,
    "Euro": 1.10,
    "UK Pound": 1.25,
    "Swiss Franc": 1.05,
    "Yuan": 0.14,
    "Rupee": 0.012,
}
_HOME_CURRENCY_WEIGHTS: Final = (0.70, 0.12, 0.06, 0.04, 0.04, 0.04)
_BENIGN_FORMATS: Final = ("Cheque", "Credit Card", "ACH", "Cash", "Wire")
_BENIGN_FORMAT_WEIGHTS: Final = (0.38, 0.30, 0.14, 0.12, 0.06)


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Size and shape of the generated dataset.

    Attributes:
        seed: RNG seed; the same seed always yields byte-identical files.
        start: First day of activity.
        days: Number of days of activity.
        n_banks: Number of distinct bank codes.
        n_accounts: Number of accounts.
        n_benign: Number of ordinary benign transactions (look-alikes are extra).
        self_transfer_rate: Fraction of benign rows that are ``Reinvestment``
            self-transfers.
        attempts_per_typology: Laundering attempts generated per IBM typology.
        lookalikes_per_kind: Benign look-alike groups per kind.
        undocumented_rate: Fraction of attempts omitted from the patterns file.
    """

    seed: int = 42
    start: datetime = datetime(2022, 9, 1)  # noqa: DTZ001 - dataset timestamps are naive
    days: int = 10
    n_banks: int = 15
    n_accounts: int = 600
    n_benign: int = 6000
    self_transfer_rate: float = 0.08
    attempts_per_typology: int = 3
    lookalikes_per_kind: int = 4
    undocumented_rate: float = 0.2


@dataclass(frozen=True, slots=True)
class _Account:
    bank: str
    account: str
    currency: str


@dataclass(slots=True)
class _Row:
    ts: datetime
    src: _Account
    dst: _Account
    amount_usd: float
    fmt: str
    label: int

    def as_fields(self) -> list[str]:
        paid = self.amount_usd / SAMPLE_FX[self.src.currency]
        received = self.amount_usd / SAMPLE_FX[self.dst.currency]
        return [
            self.ts.strftime(TIMESTAMP_FORMAT),
            self.src.bank,
            self.src.account,
            self.dst.bank,
            self.dst.account,
            f"{received:.2f}",
            self.dst.currency,
            f"{paid:.2f}",
            self.src.currency,
            self.fmt,
            str(self.label),
        ]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One planted laundering attempt.

    Attributes:
        typology: IBM typology name.
        description: Free-text description used on the ``BEGIN`` line.
        rows: The attempt's transactions in raw CSV field order.
        documented: Whether the attempt appears in the patterns file.
    """

    typology: Typology
    description: str
    rows: tuple[tuple[str, ...], ...]
    documented: bool


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Generated rows, ready to be written in IBM file format.

    Attributes:
        transactions: All rows in raw CSV field order, sorted by timestamp.
        attempts: Laundering attempts (documented and undocumented).
        n_lookalike_rows: Number of benign look-alike rows.
    """

    transactions: tuple[tuple[str, ...], ...]
    attempts: tuple[Attempt, ...]
    n_lookalike_rows: int = field(default=0)


class _Simulator:
    """Holds the RNG and account pool for one generation run."""

    def __init__(self, cfg: SyntheticConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        banks = self._bank_codes(cfg.n_banks)
        currencies = list(SAMPLE_FX)
        self.accounts = [
            _Account(
                bank=str(self.rng.choice(banks)),
                account=f"8{int(self.rng.integers(0, 16**8)):08X}",
                currency=str(self.rng.choice(currencies, p=_HOME_CURRENCY_WEIGHTS)),
            )
            for _ in range(cfg.n_accounts)
        ]
        # Heavy-tailed activity, so a few accounts are hubs, as in real networks.
        weights = self.rng.pareto(1.5, size=cfg.n_accounts) + 1.0
        self.activity = weights / weights.sum()

    def _bank_codes(self, n: int) -> list[str]:
        # Leading zeros are significant in IBM bank codes; generate some on purpose.
        codes: set[str] = set()
        while len(codes) < n:
            width = int(self.rng.choice([3, 5, 6]))
            codes.add(f"{int(self.rng.integers(1, 10 ** (width - 1))):0{width}d}")
        return sorted(codes)

    # -------------------------------------------------------------- helpers
    def account(self) -> _Account:
        return self.accounts[int(self.rng.choice(len(self.accounts), p=self.activity))]

    def distinct_accounts(self, k: int) -> list[_Account]:
        idx = self.rng.choice(len(self.accounts), size=k, replace=False)
        return [self.accounts[int(i)] for i in idx]

    def time(self, *, day_from: float = 0.0, span_days: float | None = None) -> datetime:
        span = self.cfg.days - day_from if span_days is None else span_days
        offset = day_from + float(self.rng.uniform(0, span))
        return _minute(self.cfg.start + timedelta(days=min(offset, self.cfg.days - 1e-3)))

    def window_start(self, span_days: float) -> float:
        return float(self.rng.uniform(0, self.cfg.days - span_days))

    def amount(self, median_usd: float, sigma: float = 1.0) -> float:
        return float(self.rng.lognormal(np.log(median_usd), sigma))

    # -------------------------------------------------------------- benign
    def benign(self) -> list[_Row]:
        rows: list[_Row] = []
        for _ in range(self.cfg.n_benign):
            src = self.account()
            if self.rng.random() < self.cfg.self_transfer_rate:
                rows.append(_Row(self.time(), src, src, self.amount(5000), "Reinvestment", 0))
                continue
            dst = self.account()
            while dst == src:
                dst = self.account()
            fmt = str(self.rng.choice(_BENIGN_FORMATS, p=_BENIGN_FORMAT_WEIGHTS))
            rows.append(_Row(self.time(), src, dst, self.amount(800, 1.4), fmt, 0))
        return rows

    def lookalikes(self) -> list[_Row]:
        rows: list[_Row] = []
        for _ in range(self.cfg.lookalikes_per_kind):
            # Payroll: one employer pays many employees similar amounts on one day.
            employer, *staff = self.distinct_accounts(int(self.rng.integers(8, 16)))
            day = self.window_start(1.0)
            salary = self.amount(3500, 0.3)
            rows += [
                _Row(
                    self.time(day_from=day, span_days=0.2),
                    employer,
                    e,
                    salary * float(self.rng.uniform(0.8, 1.2)),
                    "ACH",
                    0,
                )
                for e in staff
            ]
            # Merchant: many customers pay one shop small amounts.
            shop, *customers = self.distinct_accounts(int(self.rng.integers(10, 20)))
            rows += [
                _Row(self.time(), c, shop, self.amount(60, 0.6), "Credit Card", 0)
                for c in customers
            ]
            # Roommates: rent and bill reimbursements that form a short benign cycle.
            a, b, c = self.distinct_accounts(3)
            day = self.window_start(3.0)
            for s, d in ((a, b), (b, c), (c, a)):
                rows.append(
                    _Row(
                        self.time(day_from=day, span_days=3), s, d, self.amount(700, 0.2), "Wire", 0
                    )
                )
        return rows

    # -------------------------------------------------------------- laundering
    def chain_amounts(self, n: int, start_usd: float) -> list[float]:
        # Each hop skims a small "fee", the classic layering decrement.
        amounts = [start_usd]
        for _ in range(n - 1):
            amounts.append(amounts[-1] * float(self.rng.uniform(0.90, 0.99)))
        return amounts

    def laundering(self, typology: Typology) -> tuple[str, list[_Row]]:
        builders: dict[Typology, Callable[[], tuple[str, list[_Row]]]] = {
            Typology.FAN_OUT: self._fan_out,
            Typology.FAN_IN: self._fan_in,
            Typology.CYCLE: self._cycle,
            Typology.SCATTER_GATHER: self._scatter_gather,
            Typology.GATHER_SCATTER: self._gather_scatter,
            Typology.STACK: self._stack,
            Typology.BIPARTITE: self._bipartite,
            Typology.RANDOM: self._random_walk,
        }
        return builders[typology]()

    def _bad(self, src: _Account, dst: _Account, ts: datetime, usd: float) -> _Row:
        return _Row(ts, src, dst, usd, "ACH", 1)

    def _fan_out(self) -> tuple[str, list[_Row]]:
        k = int(self.rng.integers(5, 13))
        src, *dsts = self.distinct_accounts(k + 1)
        day, total = self.window_start(3), self.amount(40000, 0.5)
        rows = [
            self._bad(
                src,
                d,
                self.time(day_from=day, span_days=3),
                total / k * float(self.rng.uniform(0.7, 1.3)),
            )
            for d in dsts
        ]
        return f"Max {k}-degree Fan-Out", rows

    def _fan_in(self) -> tuple[str, list[_Row]]:
        k = int(self.rng.integers(5, 13))
        dst, *srcs = self.distinct_accounts(k + 1)
        day = self.window_start(3)
        rows = [
            self._bad(s, dst, self.time(day_from=day, span_days=3), self.amount(4000, 0.4))
            for s in srcs
        ]
        return f"Max {k}-degree Fan-In", rows

    def _cycle(self) -> tuple[str, list[_Row]]:
        k = int(self.rng.integers(3, 7))
        nodes = self.distinct_accounts(k)
        day = self.window_start(4)
        times = sorted(self.time(day_from=day, span_days=4) for _ in range(k))
        amounts = self.chain_amounts(k, self.amount(20000, 0.5))
        rows = [self._bad(nodes[i], nodes[(i + 1) % k], times[i], amounts[i]) for i in range(k)]
        return f"Max {k} hops", rows

    def _scatter_gather(self) -> tuple[str, list[_Row]]:
        k = int(self.rng.integers(3, 8))
        src, dst, *mids = self.distinct_accounts(k + 2)
        day = self.window_start(4)
        rows: list[_Row] = []
        for m in mids:
            usd = self.amount(8000, 0.3)
            t1 = self.time(day_from=day, span_days=2)
            rows.append(self._bad(src, m, t1, usd))
            rows.append(self._bad(m, dst, self.time(day_from=day + 2, span_days=2), usd * 0.95))
        return f"Max {k}-degree Scatter-Gather", rows

    def _gather_scatter(self) -> tuple[str, list[_Row]]:
        k_in, k_out = int(self.rng.integers(3, 7)), int(self.rng.integers(3, 7))
        hub, *others = self.distinct_accounts(k_in + k_out + 1)
        day = self.window_start(4)
        rows = [
            self._bad(s, hub, self.time(day_from=day, span_days=2), self.amount(6000, 0.3))
            for s in others[:k_in]
        ]
        total = sum(r.amount_usd for r in rows) * 0.95
        rows += [
            self._bad(hub, d, self.time(day_from=day + 2, span_days=2), total / k_out)
            for d in others[k_in:]
        ]
        return f"Max {k_in}-degree Fan-In, {k_out}-degree Fan-Out", rows

    def _stack(self) -> tuple[str, list[_Row]]:
        a, b, c = (int(self.rng.integers(2, 4)) for _ in range(3))
        pool = self.distinct_accounts(a + b + c)
        layers = [pool[:a], pool[a : a + b], pool[a + b :]]
        day = self.window_start(4)
        rows: list[_Row] = []
        for hop, (left, right) in enumerate(itertools.pairwise(layers)):
            for s in left:
                for d in right:
                    rows.append(
                        self._bad(
                            s,
                            d,
                            self.time(day_from=day + 2 * hop, span_days=2),
                            self.amount(5000, 0.3),
                        )
                    )
        return "Stacked bipartite", rows

    def _bipartite(self) -> tuple[str, list[_Row]]:
        a, b = int(self.rng.integers(2, 5)), int(self.rng.integers(2, 5))
        pool = self.distinct_accounts(a + b)
        day = self.window_start(3)
        rows = [
            self._bad(s, d, self.time(day_from=day, span_days=3), self.amount(5000, 0.3))
            for s in pool[:a]
            for d in pool[a:]
            if self.rng.random() < 0.8
        ]
        if not rows:
            rows.append(self._bad(pool[0], pool[a], self.time(day_from=day, span_days=3), 5000))
        return f"{a}x{b} bipartite", rows

    def _random_walk(self) -> tuple[str, list[_Row]]:
        k = int(self.rng.integers(4, 9))
        nodes = self.distinct_accounts(k + 1)
        day = self.window_start(5)
        times = sorted(self.time(day_from=day, span_days=5) for _ in range(k))
        amounts = self.chain_amounts(k, self.amount(15000, 0.4))
        rows = [self._bad(nodes[i], nodes[i + 1], times[i], amounts[i]) for i in range(k)]
        return f"{k}-hop random walk", rows


def _minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def generate(cfg: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate a synthetic dataset.

    Args:
        cfg: Generation parameters; defaults to :class:`SyntheticConfig`.

    Returns:
        The generated dataset. The result is deterministic for a given config.
    """
    cfg = cfg or SyntheticConfig()
    sim = _Simulator(cfg)
    rows = sim.benign()
    lookalikes = sim.lookalikes()
    rows += lookalikes

    attempts: list[Attempt] = []
    for typology in Typology:
        for _ in range(cfg.attempts_per_typology):
            description, bad_rows = sim.laundering(typology)
            bad_rows.sort(key=lambda r: r.ts)
            rows += bad_rows
            attempts.append(
                Attempt(
                    typology=typology,
                    description=description,
                    rows=tuple(tuple(r.as_fields()) for r in bad_rows),
                    documented=bool(sim.rng.random() >= cfg.undocumented_rate),
                )
            )

    rows.sort(key=lambda r: r.ts)
    return SyntheticDataset(
        transactions=tuple(tuple(r.as_fields()) for r in rows),
        attempts=tuple(attempts),
        n_lookalike_rows=len(lookalikes),
    )


def _csv_lines(rows: Sequence[Sequence[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


def write_dataset(data: SyntheticDataset, out_dir: Path, *, name: str = "Sample") -> Path:
    """Write ``<name>_Trans.csv`` and ``<name>_Patterns.txt`` to ``out_dir``.

    Args:
        data: Output of :func:`generate`.
        out_dir: Destination directory (created if missing).
        name: File-name prefix.

    Returns:
        Path of the transactions file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trans = out_dir / f"{name}_Trans.csv"
    # newline="\n": byte-identical output on every OS (Windows would write CRLF).
    trans.write_text(_csv_lines([RAW_HEADER, *data.transactions]), encoding="utf-8", newline="\n")

    blocks: list[str] = []
    for attempt in data.attempts:
        if not attempt.documented:
            continue
        blocks.append(
            f"BEGIN LAUNDERING ATTEMPT - {attempt.typology.value}:  {attempt.description}\n"
            + _csv_lines(attempt.rows)
            + f"END LAUNDERING ATTEMPT - {attempt.typology.value}\n"
        )
    (out_dir / f"{name}_Patterns.txt").write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    return trans
