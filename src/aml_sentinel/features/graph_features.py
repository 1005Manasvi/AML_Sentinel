"""Daily account-graph snapshot features, computed strictly from the past.

For every calendar day ``d``, node statistics are computed from all transactions with
``timestamp < d`` (midnight), then joined to that day's transactions for both the
sender and the receiver. A transaction therefore never sees its own day, let alone
the future. Intraday dynamics are covered by :mod:`aml_sentinel.features.temporal`.

Statistics use sparse matrices over a global account index instead of NetworkX:
counting 3-cycles on the full HI-Small graph (423k accounts) takes under a second
this way (ADR-005).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from aml_sentinel.data.schema import Col

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

NODE_FEATURES: Final = (
    "out_tx",
    "in_tx",
    "out_usd_log",
    "in_usd_log",
    "out_degree",
    "in_degree",
    "flow_ratio",
    "reciprocal",
    "cycle3",
    "has_history",
    "age_days",
)


def account_codes(frame: pd.DataFrame) -> tuple[IntArray, IntArray, int]:
    """Map sender and receiver IDs to one shared integer index.

    Args:
        frame: Transactions with ``from_id`` and ``to_id``.

    Returns:
        ``(src_codes, dst_codes, n_accounts)``.
    """
    both = pd.concat(
        [frame[Col.FROM_ID].astype("string"), frame[Col.TO_ID].astype("string")],
        ignore_index=True,
    )
    codes, uniques = pd.factorize(both)
    n = len(frame)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), len(uniques)


def node_stats(
    src: IntArray,
    dst: IntArray,
    usd: FloatArray,
    minutes: IntArray,
    n_nodes: int,
) -> dict[str, FloatArray]:
    """Compute per-account statistics from a set of historical edges.

    Args:
        src: Payer codes.
        dst: Payee codes.
        usd: USD amounts.
        minutes: Timestamps in minutes.
        n_nodes: Number of accounts in the index.

    Returns:
        Arrays of length ``n_nodes`` keyed by feature name (plus ``first_seen``).
    """
    first_seen = np.full(n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_seen, src, minutes)
    np.minimum.at(first_seen, dst, minutes)

    keep = src != dst  # self-transfers say nothing about counterparties
    s, d, a = src[keep], dst[keep], usd[keep]
    out_tx = np.bincount(s, minlength=n_nodes).astype(np.float64)
    in_tx = np.bincount(d, minlength=n_nodes).astype(np.float64)
    out_usd = np.bincount(s, weights=a, minlength=n_nodes)
    in_usd = np.bincount(d, weights=a, minlength=n_nodes)

    adj = sp.csr_matrix((np.ones(s.size, dtype=np.float32), (s, d)), shape=(n_nodes, n_nodes))
    adj.sum_duplicates()
    adj.data[:] = 1.0  # distinct counterparties, not payment counts
    out_degree = np.asarray(adj.sum(axis=1)).ravel()
    in_degree = np.asarray(adj.sum(axis=0)).ravel()
    reciprocal = np.asarray(adj.multiply(adj.T).sum(axis=1)).ravel()
    # Number of directed 3-cycles through each node: diag(A^3) = rowsum(A * (A^2)^T).
    cycle3 = np.asarray(adj.multiply((adj @ adj).T).sum(axis=1)).ravel()

    total = out_usd + in_usd
    flow_ratio = np.divide(in_usd, total, out=np.full(n_nodes, 0.5), where=total > 0)
    return {
        "out_tx": out_tx,
        "in_tx": in_tx,
        "out_usd_log": np.log1p(out_usd),
        "in_usd_log": np.log1p(in_usd),
        "out_degree": out_degree.astype(np.float64),
        "in_degree": in_degree.astype(np.float64),
        "flow_ratio": flow_ratio,
        "reciprocal": reciprocal.astype(np.float64),
        "cycle3": cycle3.astype(np.float64),
        "first_seen": first_seen.astype(np.float64),
    }


def snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute sender (``src_*``) and receiver (``dst_*``) snapshot features.

    Args:
        frame: Transactions sorted by time, with ``from_id``, ``to_id``,
            ``timestamp`` and ``amount_paid_usd``.

    Returns:
        Float32 DataFrame aligned with ``frame`` (same index), with one column per
        node feature and side, e.g. ``src_out_degree`` or ``dst_cycle3``.
    """
    src, dst, n_nodes = account_codes(frame)
    ts = frame[Col.TIMESTAMP].to_numpy()
    minutes = ts.astype("datetime64[m]").astype(np.int64)
    usd = frame[Col.AMOUNT_PAID_USD].to_numpy(dtype=np.float64)
    day = ts.astype("datetime64[D]")

    out = {
        f"{side}_{name}": np.zeros(len(frame), dtype=np.float32)
        for side in ("src", "dst")
        for name in NODE_FEATURES
    }
    for current in np.unique(day):
        rows = np.flatnonzero(day == current)
        history = minutes < current.astype("datetime64[m]").astype(np.int64)
        stats = node_stats(src[history], dst[history], usd[history], minutes[history], n_nodes)
        now = minutes[rows]
        for side, ids in (("src", src[rows]), ("dst", dst[rows])):
            seen = stats["first_seen"][ids]
            has_history = seen < np.iinfo(np.int64).max
            for name in NODE_FEATURES:
                if name == "has_history":
                    values = has_history.astype(np.float64)
                elif name == "age_days":
                    values = np.where(has_history, (now - seen) / 1440.0, 0.0)
                else:
                    values = stats[name][ids]
                out[f"{side}_{name}"][rows] = values
    return pd.DataFrame(out, index=frame.index)
