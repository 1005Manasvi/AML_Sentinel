"""Rule-based graph detectors for System 2 (NetworkX), run on one case at a time.

Each detector looks for one laundering shape **involving the alert's accounts** and
returns typed :class:`Finding` objects: the involved accounts and transaction IDs,
a strength value, and numeric facts. Findings are evidence for an analyst and for
the grounded SAR (Phase 6), not verdicts.

| Detector       | Shape                                                    | IBM typology   |
|----------------|----------------------------------------------------------|----------------|
| circular       | time-respecting cycle of 2..max_len accounts             | CYCLE          |
| structuring    | >= k deposits just under the threshold, distinct senders | STRUCTURING    |
| fan_in         | >= k distinct payers into one account                    | FAN-IN         |
| fan_out        | >= k distinct payees from one account                    | FAN-OUT        |
| scatter_gather | source -> >= k intermediaries -> one sink, in time order | SCATTER-GATHER |
| gather_scatter | >= k payers -> hub -> >= k payees, outflows after inflows| GATHER-SCATTER |

There is no dedicated detector for STACK, BIPARTITE or RANDOM. Those shapes only
partially overlap the ones above, and this limitation is reported in the
evaluation.

Self-transfers are ignored throughout.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.system2.cases import Case

REPORTING_THRESHOLD_USD: Final = 10_000.0


class FindingKind(StrEnum):
    """Detector names (also the finding type)."""

    CIRCULAR = "circular"
    STRUCTURING = "structuring"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    SCATTER_GATHER = "scatter_gather"
    GATHER_SCATTER = "gather_scatter"


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Detector thresholds.

    Attributes:
        cycle_max_len: Longest cycle searched.
        cycle_min_amount_ratio: Smallest/largest amount along a cycle (rejects
            unrelated small payments that happen to close a loop).
        cycle_budget: Maximum candidate cycles enumerated per case (dense cases are
            exponential); hitting it is recorded on the finding.
        structuring_band: Fraction of the threshold that counts as "just below".
        structuring_min: Minimum near-threshold deposits from distinct senders.
        fan_min: Minimum distinct counterparties for fan-in/out.
        scatter_min: Minimum intermediaries for scatter-gather / gather-scatter.
    """

    cycle_max_len: int = 6
    cycle_min_amount_ratio: float = 0.5
    cycle_budget: int = 5_000
    structuring_band: tuple[float, float] = (0.8, 1.0)
    structuring_min: int = 3
    fan_min: int = 5
    scatter_min: int = 3


@dataclass(frozen=True, slots=True)
class Finding:
    """One detected pattern.

    Attributes:
        kind: Detector that fired.
        accounts: Involved account ids (sorted).
        tx_ids: Supporting transaction ids (sorted).
        strength: Detector-specific size (cycle count, deposit count, degree...).
        facts: Numeric facts for analysts and SAR grounding.
    """

    kind: FindingKind
    accounts: tuple[str, ...]
    tx_ids: tuple[int, ...]
    strength: float
    facts: dict[str, float] = field(default_factory=dict)


def _clean(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.loc[edges[Col.FROM_ID] != edges[Col.TO_ID]]


def _finding(kind: FindingKind, rows: pd.DataFrame, strength: float, **facts: float) -> Finding:
    accounts = sorted(set(rows[Col.FROM_ID]) | set(rows[Col.TO_ID]))
    return Finding(
        kind=kind,
        accounts=tuple(str(a) for a in accounts),
        tx_ids=tuple(sorted(int(t) for t in rows[Col.TX_ID])),
        strength=float(strength),
        facts={
            "total_usd": float(rows[Col.AMOUNT_PAID_USD].sum()),
            "n_transactions": float(len(rows)),
            **facts,
        },
    )


@dataclass(frozen=True, slots=True)
class _Pair:
    times: npt.NDArray[np.datetime64]  # sorted payment times u -> v
    max_usd: float
    rows: npt.NDArray[np.int64]  # positions in the cleaned case frame


def _pair_index(e: pd.DataFrame) -> dict[tuple[str, str], _Pair]:
    """Per (payer, payee): sorted times, largest amount and row positions (built once)."""
    out: dict[tuple[str, str], _Pair] = {}
    src = e[Col.FROM_ID].to_numpy()
    dst = e[Col.TO_ID].to_numpy()
    ts = e[Col.TIMESTAMP].to_numpy()
    usd = e[Col.AMOUNT_PAID_USD].to_numpy(dtype=np.float64)
    order = np.lexsort((ts, dst, src))
    keys = list(zip(src[order], dst[order], strict=True))
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or keys[i] != keys[start]:
            idx = order[start:i]
            out[(str(keys[start][0]), str(keys[start][1]))] = _Pair(
                times=ts[idx], max_usd=float(usd[idx].max()), rows=idx.astype(np.int64)
            )
            start = i
    return out


def detect_circular(case: Case, cfg: DetectorConfig) -> list[Finding]:
    """Time-respecting cycles through either focal account.

    A cycle A->B->...->A counts when each hop has a payment no earlier than a payment
    on the previous hop (money can flow onward), and amounts along it stay within
    ``cycle_min_amount_ratio`` of each other. Enumeration stops after
    ``cycle_budget`` candidate cycles; the finding records whether that happened.

    Args:
        case: Case to inspect.
        cfg: Thresholds.

    Returns:
        At most one finding, merging every qualifying cycle.
    """
    e = _clean(case.edges).reset_index(drop=True)
    pairs = _pair_index(e)
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edges_from(pairs)
    focal = set(case.focal) & set(graph)
    if not focal:
        return []
    hit_rows: set[int] = set()
    n_cycles = examined = 0
    budget_hit = False
    for cycle in nx.simple_cycles(graph, length_bound=cfg.cycle_max_len):
        examined += 1
        if examined > cfg.cycle_budget:
            budget_hit = True
            break
        if len(cycle) < 2 or not focal & set(cycle):
            continue
        hops = [pairs[(cycle[i], cycle[(i + 1) % len(cycle)])] for i in range(len(cycle))]
        if _time_respecting(hops) and _amounts_consistent(hops, cfg.cycle_min_amount_ratio):
            n_cycles += 1
            for h in hops:
                hit_rows.update(int(r) for r in h.rows)
    if not n_cycles:
        return []
    rows = e.iloc[sorted(hit_rows)]
    return [
        _finding(
            FindingKind.CIRCULAR,
            rows,
            n_cycles,
            n_cycles=float(n_cycles),
            search_truncated=float(budget_hit),
        )
    ]


def _time_respecting(hops: list[_Pair]) -> bool:
    # Rotate the start: some starting hop must allow a non-decreasing time path.
    for start in range(len(hops)):
        t = hops[start].times[0]
        ok = True
        for k in range(1, len(hops)):
            nxt = hops[(start + k) % len(hops)].times
            j = int(np.searchsorted(nxt, t, side="left"))
            if j == nxt.size:
                ok = False
                break
            t = nxt[j]
        if ok:
            return True
    return False


def _amounts_consistent(hops: list[_Pair], ratio: float) -> bool:
    totals = [h.max_usd for h in hops]
    return max(totals) > 0 and min(totals) / max(totals) >= ratio


def detect_structuring(case: Case, cfg: DetectorConfig) -> list[Finding]:
    """Near-threshold deposits from distinct senders into a focal account.

    Args:
        case: Case to inspect.
        cfg: Thresholds.

    Returns:
        One finding per focal account that qualifies.
    """
    e = _clean(case.edges)
    lo, hi = (b * REPORTING_THRESHOLD_USD for b in cfg.structuring_band)
    near = e.loc[(e[Col.AMOUNT_PAID_USD] >= lo) & (e[Col.AMOUNT_PAID_USD] < hi)]
    out = []
    for acct in dict.fromkeys(case.focal):
        rows = near.loc[near[Col.TO_ID] == acct]
        senders = rows[Col.FROM_ID].nunique()
        if senders >= cfg.structuring_min:
            out.append(
                _finding(FindingKind.STRUCTURING, rows, senders, distinct_senders=float(senders))
            )
    return out


def detect_fans(case: Case, cfg: DetectorConfig) -> list[Finding]:
    """Fan-in / fan-out around each focal account.

    Args:
        case: Case to inspect.
        cfg: Thresholds.

    Returns:
        Findings for every focal account with enough distinct counterparties.
    """
    e = _clean(case.edges)
    out = []
    for acct in dict.fromkeys(case.focal):
        for kind, own, other in (
            (FindingKind.FAN_IN, Col.TO_ID, Col.FROM_ID),
            (FindingKind.FAN_OUT, Col.FROM_ID, Col.TO_ID),
        ):
            rows = e.loc[e[own] == acct]
            k = rows[other].nunique()
            if k >= cfg.fan_min:
                amounts = rows[Col.AMOUNT_PAID_USD].to_numpy()
                cv = float(np.std(amounts) / np.mean(amounts)) if amounts.mean() > 0 else 0.0
                out.append(_finding(kind, rows, k, counterparties=float(k), amount_cv=cv))
    return out


def detect_scatter_gather(case: Case, cfg: DetectorConfig) -> list[Finding]:
    """Scatter-gather (s -> many m -> d) and gather-scatter (many -> hub -> many).

    Scatter-gather counts intermediaries ``m`` that received from ``s`` and later paid
    the same sink ``d``. Either ``s`` or ``d`` must be a focal account. Gather-scatter
    requires a focal hub whose outflows to at least ``k`` payees come after its
    inflows from at least ``k`` payers.

    Args:
        case: Case to inspect.
        cfg: Thresholds.

    Returns:
        Findings for qualifying shapes.
    """
    e = _clean(case.edges).reset_index(drop=True)
    pairs = _pair_index(e)
    out_by: dict[str, set[str]] = defaultdict(set)
    for s, m in pairs:
        out_by[s].add(m)

    findings = []
    focal = set(case.focal)
    for s, mids in out_by.items():
        sinks: dict[str, list[str]] = defaultdict(list)
        for m in mids:
            received = pairs[(s, m)].times[0]
            for d in out_by.get(m, ()):
                # Onward flow: m paid d at or after first receiving from s.
                if d != s and pairs[(m, d)].times[-1] >= received:
                    sinks[d].append(m)
        for d, via in sinks.items():
            if len(via) >= cfg.scatter_min and ({s, d} & focal):
                idx = np.concatenate(
                    [pairs[(s, m)].rows for m in via] + [pairs[(m, d)].rows for m in via]
                )
                findings.append(
                    _finding(
                        FindingKind.SCATTER_GATHER,
                        e.iloc[np.unique(idx)],
                        len(via),
                        intermediaries=float(len(via)),
                    )
                )

    for hub in dict.fromkeys(case.focal):
        inflow = e.loc[e[Col.TO_ID] == hub]
        if inflow[Col.FROM_ID].nunique() < cfg.scatter_min:
            continue
        outflow = e.loc[(e[Col.FROM_ID] == hub) & (e[Col.TIMESTAMP] >= inflow[Col.TIMESTAMP].min())]
        if outflow[Col.TO_ID].nunique() >= cfg.scatter_min:
            rows = pd.concat([inflow, outflow])
            findings.append(
                _finding(
                    FindingKind.GATHER_SCATTER,
                    rows,
                    min(inflow[Col.FROM_ID].nunique(), outflow[Col.TO_ID].nunique()),
                    payers=float(inflow[Col.FROM_ID].nunique()),
                    payees=float(outflow[Col.TO_ID].nunique()),
                )
            )
    return findings


ALL_DETECTORS: Final = (detect_circular, detect_structuring, detect_fans, detect_scatter_gather)


def run_detectors(case: Case, cfg: DetectorConfig | None = None) -> list[Finding]:
    """Run every detector on a case.

    Args:
        case: Case to inspect.
        cfg: Thresholds.

    Returns:
        All findings, ordered by detector.
    """
    cfg = cfg or DetectorConfig()
    findings: list[Finding] = []
    for detector in ALL_DETECTORS:
        findings.extend(detector(case, cfg))
    return findings
