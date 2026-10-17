"""Build account-level transaction graphs.

Nodes are accounts (``"<bank>:<account>"``). Each directed edge ``u -> v``
aggregates every payment from ``u`` to ``v`` inside a time window, which keeps the
graph about 2-3x smaller than a multigraph with one edge per payment.

Labels are deliberately **not** copied onto the graph. Graph features feed the
models, so a label attribute would be one careless line away from target leakage.
"""

from __future__ import annotations

from datetime import datetime

import networkx as nx
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError

EDGE_N_TX = "n_tx"
EDGE_TOTAL_USD = "total_usd"
EDGE_FIRST_TS = "first_ts"
EDGE_LAST_TS = "last_ts"
NODE_BANK = "bank"


def window(
    frame: pd.DataFrame, *, start: datetime | None = None, end: datetime | None = None
) -> pd.DataFrame:
    """Select transactions in the half-open time window ``[start, end)``.

    Args:
        frame: Transactions with a ``timestamp`` column.
        start: Inclusive lower bound, or ``None`` for unbounded.
        end: Exclusive upper bound, or ``None`` for unbounded.

    Returns:
        The rows inside the window.
    """
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame[Col.TIMESTAMP] >= start
    if end is not None:
        mask &= frame[Col.TIMESTAMP] < end
    return frame.loc[mask]


def build_account_graph(
    frame: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    include_self_loops: bool = False,
) -> nx.DiGraph[str]:
    """Aggregate transactions into a directed account graph.

    Args:
        frame: Transactions with ``from_id``, ``to_id``, ``timestamp`` and
            ``amount_paid_usd`` (see :func:`aml_sentinel.data.fx.add_usd_amounts`).
        start: Inclusive window start, or ``None``.
        end: Exclusive window end, or ``None``. Pass the scoring time when building
            features so that no future edge is visible.
        include_self_loops: Keep ``u -> u`` edges. They are off by default because
            "Reinvestment" self-transfers (~10% of rows) inflate degree and flow
            features without saying anything about counterparties.

    Returns:
        A ``DiGraph`` whose edges carry ``n_tx``, ``total_usd``, ``first_ts`` and
        ``last_ts`` and whose nodes carry ``bank``. Window bounds are recorded in
        ``graph.graph["start"]`` and ``graph.graph["end"]``.

    Raises:
        DataValidationError: If a required column is missing.
    """
    required = {Col.FROM_ID, Col.TO_ID, Col.TIMESTAMP, Col.AMOUNT_PAID_USD}
    missing = {c.value for c in required} - set(frame.columns)
    if missing:
        raise DataValidationError(f"graph builder missing columns: {sorted(missing)}")

    rows = window(frame, start=start, end=end)
    if not include_self_loops:
        rows = rows.loc[rows[Col.FROM_ID].astype("string") != rows[Col.TO_ID].astype("string")]

    agg = (
        rows.groupby([Col.FROM_ID.value, Col.TO_ID.value], observed=True, sort=False)
        .agg(
            n_tx=(Col.TIMESTAMP.value, "size"),
            total_usd=(Col.AMOUNT_PAID_USD.value, "sum"),
            first_ts=(Col.TIMESTAMP.value, "min"),
            last_ts=(Col.TIMESTAMP.value, "max"),
        )
        .reset_index()
    )

    graph: nx.DiGraph[str] = nx.DiGraph(start=start, end=end)
    graph.add_edges_from(
        (
            str(u),
            str(v),
            {
                EDGE_N_TX: int(n),
                EDGE_TOTAL_USD: float(total),
                EDGE_FIRST_TS: first,
                EDGE_LAST_TS: last,
            },
        )
        for u, v, n, total, first, last in agg.itertuples(index=False, name=None)
    )
    for node, attrs in graph.nodes(data=True):
        attrs[NODE_BANK] = node.split(":", 1)[0]
    return graph


def ego_subgraph(graph: nx.DiGraph[str], node: str, *, radius: int = 2) -> nx.DiGraph[str]:
    """Return the neighbourhood of ``node`` ignoring edge direction.

    Used for case views and System 2 detectors, which only need the local
    structure around an alert.

    Args:
        graph: Account graph.
        node: Centre account node ID.
        radius: Number of hops to include.

    Returns:
        A copy of the induced subgraph.

    Raises:
        DataValidationError: If ``node`` is not in the graph.
    """
    if node not in graph:
        raise DataValidationError("node not in graph")
    ego: nx.DiGraph[str] = nx.ego_graph(graph, node, radius=radius, undirected=True)
    return ego.copy()
