"""Build per-period account-pair graphs for the GNN experiment (NumPy only).

For a period P (train, val or test):

* **Edges** are the distinct (payer, payee) pairs with at least one payment in P
  (self-transfers excluded). Edge features are log payment count, log total USD
  and log largest USD within P. The label is 1 if any payment on the pair in P was
  laundering. Labels are only ever used as targets.
* **Node features** are account statistics computed from **P's own structure**:
  degrees, payment counts, USD flows, flow ratio, reciprocity and 3-cycles. They use
  the same statistics as the System 1 snapshots, never labels.

So the GNN scores pairs in an *end-of-period, batch* setting: it sees the whole
period's structure. That differs from System 1's per-transaction real-time
setting, and the comparison in the report says so (ADR-016).

Payment format and currency are not features, which keeps the shortcut-free spirit
of ADR-010.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.schema import Col
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.features.graph_features import account_codes, node_stats

NODE_STAT_KEYS: Final = (
    "out_tx",
    "in_tx",
    "out_usd_log",
    "in_usd_log",
    "out_degree",
    "in_degree",
    "flow_ratio",
    "reciprocal",
    "cycle3",
)
EDGE_FEATURES: Final = ("log_n_tx", "log_total_usd", "log_max_usd")
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PeriodGraph:
    """One period's pair graph.

    Attributes:
        x: Node features, shape (n_nodes, d_node).
        edge_index: Pair endpoints, shape (2, n_pairs), directed payer -> payee.
        edge_attr: Pair features, shape (n_pairs, len(EDGE_FEATURES)).
        y: Pair labels (1 = any laundering payment in the period).
        pair_src: Payer account codes (same as ``edge_index[0]``).
        pair_dst: Payee account codes.
    """

    x: FloatArray
    edge_index: IntArray
    edge_attr: FloatArray
    y: npt.NDArray[np.int8]
    pair_src: IntArray
    pair_dst: IntArray


def _log1p_counts(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.log1p(np.maximum(values, 0.0))


def build_period_graph(
    frame: pd.DataFrame,
    in_period: npt.NDArray[np.bool_],
    src: IntArray,
    dst: IntArray,
    n_nodes: int,
) -> PeriodGraph:
    """Build the pair graph of one period.

    Args:
        frame: Processed transactions sorted by time.
        in_period: Rows belonging to the period.
        src: Payer codes for every row (from :func:`account_codes`).
        dst: Payee codes for every row.
        n_nodes: Size of the shared account index.

    Returns:
        The period graph.

    Raises:
        DataValidationError: If the period is empty.
    """
    if not in_period.any():
        raise DataValidationError("period has no transactions")
    ts = frame[Col.TIMESTAMP].to_numpy()
    minutes = ts.astype("datetime64[m]").astype(np.int64)
    usd = frame[Col.AMOUNT_PAID_USD].to_numpy(dtype=np.float64)
    y_rows = frame[Col.IS_LAUNDERING].to_numpy().astype(np.int8)

    # Node statistics come from the period's OWN structure, never its labels. A
    # "history before the period" design gave the train graph (which starts on the
    # first day of data) all-zero node features but not val/test: a train/test
    # distribution shift that wrecked the first run (ADR-016).
    stats = node_stats(src[in_period], dst[in_period], usd[in_period], minutes[in_period], n_nodes)

    keep = in_period & (src != dst)
    s, d = src[keep], dst[keep]
    pair_key = s * n_nodes + d
    uniq, inverse = np.unique(pair_key, return_inverse=True)
    n_pairs = uniq.size
    n_tx = np.bincount(inverse, minlength=n_pairs).astype(np.float64)
    total = np.bincount(inverse, weights=usd[keep], minlength=n_pairs)
    max_usd = np.full(n_pairs, 0.0)
    np.maximum.at(max_usd, inverse, usd[keep])
    label = np.zeros(n_pairs, dtype=np.int8)
    np.maximum.at(label, inverse, y_rows[keep])
    pair_src, pair_dst = (uniq // n_nodes).astype(np.int64), (uniq % n_nodes).astype(np.int64)

    columns = [
        _log1p_counts(stats[k])
        if k not in ("flow_ratio", "out_usd_log", "in_usd_log")
        else stats[k]
        for k in NODE_STAT_KEYS
    ]
    x = np.column_stack(columns)
    return PeriodGraph(
        x=x.astype(np.float32),
        edge_index=np.vstack([pair_src, pair_dst]),
        edge_attr=np.column_stack([np.log1p(n_tx), np.log1p(total), np.log1p(max_usd)]).astype(
            np.float32
        ),
        y=label,
        pair_src=pair_src,
        pair_dst=pair_dst,
    )


def build_split_graphs(frame: pd.DataFrame, split_col: str = "split") -> dict[str, PeriodGraph]:
    """Build train, val and test period graphs over one shared account index.

    Args:
        frame: Processed transactions with a ``split`` column.
        split_col: Name of the split column.

    Returns:
        ``{"train": ..., "val": ..., "test": ...}``.
    """
    src, dst, n_nodes = account_codes(frame)
    split = frame[split_col].astype("string").to_numpy()
    return {
        name: build_period_graph(frame, split == name, src, dst, n_nodes)
        for name in ("train", "val", "test")
    }


def pair_max_scores(
    frame: pd.DataFrame,
    in_period: npt.NDArray[np.bool_],
    tx_scores: pd.Series,
    graph: PeriodGraph,
) -> npt.NDArray[np.float64]:
    """Aggregate per-transaction scores to the graph's pairs (max over the period).

    Args:
        frame: Processed transactions.
        in_period: Rows of the period.
        tx_scores: Score per ``tx_id`` (rows without a score are ignored).
        graph: Period graph whose pair order to follow.

    Returns:
        One score per pair (0 when no transaction on the pair has a score).
    """
    src, dst, n_nodes = account_codes(frame)
    keep = in_period & (src != dst)
    scores = tx_scores.reindex(frame[Col.TX_ID].to_numpy()[keep]).to_numpy(dtype=np.float64)
    ok = ~np.isnan(scores)
    key = (src[keep] * n_nodes + dst[keep])[ok]
    pair_key = graph.pair_src * n_nodes + graph.pair_dst
    order = np.argsort(pair_key)
    pos = order[np.searchsorted(pair_key[order], key)]
    out = np.zeros(pair_key.size, dtype=np.float64)
    np.maximum.at(out, pos, scores[ok])
    return out
