from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.graph.builder import (
    EDGE_FIRST_TS,
    EDGE_LAST_TS,
    EDGE_N_TX,
    EDGE_TOTAL_USD,
    NODE_BANK,
    build_account_graph,
    ego_subgraph,
)

T0 = datetime(2022, 9, 1)
ACCOUNTS = [f"00{b}:ACC{a}" for b in range(3) for a in range(4)]

tx_strategy = st.lists(
    st.tuples(
        st.sampled_from(ACCOUNTS),
        st.sampled_from(ACCOUNTS),
        st.integers(min_value=0, max_value=10 * 24 * 60),  # minutes after T0
        st.floats(min_value=0, max_value=1e7, allow_nan=False),
    ),
    max_size=60,
)


def _frame(txs: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            Col.FROM_ID.value: pd.Categorical([t[0] for t in txs]),
            Col.TO_ID.value: pd.Categorical([t[1] for t in txs]),
            Col.TIMESTAMP.value: pd.to_datetime([T0 + timedelta(minutes=t[2]) for t in txs]),
            Col.AMOUNT_PAID_USD.value: [t[3] for t in txs],
        }
    )


@settings(max_examples=150, deadline=None)
@given(tx_strategy)
def test_edges_conserve_counts_and_amounts(txs: list[tuple[str, str, int, float]]) -> None:
    g = build_account_graph(_frame(txs))
    non_loops = [t for t in txs if t[0] != t[1]]
    assert sum(d[EDGE_N_TX] for *_, d in g.edges(data=True)) == len(non_loops)
    assert sum(d[EDGE_TOTAL_USD] for *_, d in g.edges(data=True)) == pytest.approx(
        sum(t[3] for t in non_loops), rel=1e-9, abs=1e-6
    )
    assert set(g.nodes) == {a for t in non_loops for a in t[:2]}
    assert nx_selfloops(g) == 0


@settings(max_examples=150, deadline=None)
@given(tx_strategy, st.integers(0, 10 * 24 * 60), st.integers(0, 10 * 24 * 60))
def test_window_hides_rows_outside(txs: list[tuple[str, str, int, float]], a: int, b: int) -> None:
    start, end = T0 + timedelta(minutes=min(a, b)), T0 + timedelta(minutes=max(a, b))
    g = build_account_graph(_frame(txs), start=start, end=end, include_self_loops=True)
    inside = [t for t in txs if start <= T0 + timedelta(minutes=t[2]) < end]
    assert sum(d[EDGE_N_TX] for *_, d in g.edges(data=True)) == len(inside)
    for *_, d in g.edges(data=True):
        assert start <= d[EDGE_FIRST_TS] <= d[EDGE_LAST_TS] < end
    assert g.graph == {"start": start, "end": end}


@settings(max_examples=100, deadline=None)
@given(tx_strategy)
def test_degrees_match_distinct_counterparties(txs: list[tuple[str, str, int, float]]) -> None:
    g = build_account_graph(_frame(txs))
    for node in g:
        assert g.out_degree(node) == len({t[1] for t in txs if t[0] == node and t[1] != node})
        assert g.nodes[node][NODE_BANK] == node.split(":")[0]


def nx_selfloops(g: object) -> int:
    import networkx as nx

    return int(nx.number_of_selfloops(g))  # type: ignore[arg-type]


def test_missing_usd_column_rejected() -> None:
    df = _frame([("001:ACC1", "002:ACC1", 0, 1.0)]).drop(columns=[Col.AMOUNT_PAID_USD.value])
    with pytest.raises(DataValidationError, match="amount_paid_usd"):
        build_account_graph(df)


def test_ego_subgraph() -> None:
    txs = [("A:1", "B:1", 0, 1.0), ("B:1", "C:1", 1, 1.0), ("C:1", "D:1", 2, 1.0)]
    g = build_account_graph(_frame(txs))
    assert set(ego_subgraph(g, "B:1", radius=1)) == {"A:1", "B:1", "C:1"}
    with pytest.raises(DataValidationError):
        ego_subgraph(g, "Z:9")
