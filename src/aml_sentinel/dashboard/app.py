"""Streamlit analyst dashboard: ranked queue, case graph, SAR editor, decisions, audit log.

Run with ``make dashboard`` (``streamlit run src/aml_sentinel/dashboard/app.py``) while
the API is running. The dashboard is only an API client: it signs in with the
analyst's own key, and every view, draft and decision is authorised and audited by
the API.

Security:
    Untrusted values (accounts, notes, SAR text) are shown only through plain-text
    widgets; dynamic text in Markdown widgets goes through
    :func:`~aml_sentinel.dashboard.views.plain`. ``unsafe_allow_html`` is never used,
    and a repository test enforces that. The API key lives only in this browser
    session's ``st.session_state``, never in a cache shared between sessions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

import pandas as pd
import streamlit as st

from aml_sentinel.config import load_settings
from aml_sentinel.dashboard import client as api
from aml_sentinel.dashboard.views import (
    case_graph,
    findings_table,
    plain,
    queue_table,
    records_table,
)
from aml_sentinel.exceptions import ConfigurationError

CASE_RE: Final = re.compile(r"^CASE-\d{1,12}$")
ACTIONS: Final = ("escalate", "dismiss", "approve")
ACTION_HELP: Final = {
    "escalate": "Needs senior review",
    "dismiss": "False positive; close the case",
    "approve": "Approve the SAR draft for filing (needs a grounded draft)",
}


def _api_url() -> str:
    try:
        return load_settings().dashboard_api_url
    except ConfigurationError:
        return "http://127.0.0.1:8000"


def _sign_in() -> api.ApiClient | None:
    state = st.session_state
    if "client" in state:
        client: api.ApiClient = state["client"]
        return client
    with st.sidebar.form("sign-in"):
        key = st.text_input("API key", type="password", max_chars=256)
        submitted = st.form_submit_button("Sign in")
    if submitted and key.strip():
        candidate = api.make_client(_api_url(), key.strip())
        try:
            me = candidate.me()
        except api.ApiError as exc:
            st.sidebar.error(f"Sign-in failed: {plain(exc)}")
            return None
        state["client"], state["me"] = candidate, me
        st.rerun()
    return None


def _flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(plain(message))


def _queue_page(client: api.ApiClient) -> None:
    escalated_only = st.toggle("Escalated alerts only", value=True, key="escalated_only")
    data = client.alerts(escalated_only=escalated_only)
    st.caption(
        plain(
            f"Ranked by System 1 p_triage ({data['triage_source']}) on {data['dataset']}. "
            f"Escalation threshold {data['escalation_threshold']:.4f} "
            f"(top {data['escalate_share']:.0%} of validation alerts). "
            f"System 2 findings are evidence and do not change the order."
        )
    )
    st.dataframe(queue_table(data["items"]), hide_index=True)
    st.caption(plain(f"{data['total']} alerts shown."))
    options = [i["case_id"] for i in data["items"]]
    if options:
        chosen = st.selectbox("Case", options, key="queue_choice")
        if st.button("Open case", type="primary"):
            st.session_state["case_id"] = chosen
            st.session_state["goto"] = "Case"
            st.rerun()


def _load_case(client: api.ApiClient, case_id: str) -> dict[str, Any]:
    # Cached per browser session so that widget reruns do not re-fetch (each fetch is
    # an audited view); refreshed after any change.
    state = st.session_state
    if state.get("case_cache_id") != case_id or state.pop("refresh", False):
        state["case_cache"] = client.case(case_id)
        state["case_cache_id"] = case_id
    detail: dict[str, Any] = state["case_cache"]
    return detail


def _changed(message: str) -> None:
    st.session_state["refresh"] = True
    st.session_state["flash"] = message
    st.rerun()


def _sar_section(client: api.ApiClient, d: dict[str, Any]) -> None:
    st.subheader("SAR draft")
    case_id, sar = d["case_id"], d["sar"]
    if sar is None:
        st.caption("No draft yet. Drafts are built only from this case's data.")
        if st.button("Generate SAR draft", type="primary"):
            client.generate_sar(case_id)
            _changed("SAR draft generated.")
        return
    st.caption(
        plain(f"Version {sar['version']} by {sar['author']} ({sar['source']}), {sar['created_at']}")
    )
    if sar["grounded"]:
        st.success("Grounding check passed: every account, amount and date matches the case.")
    else:
        st.error(
            f"Grounding check failed: {len(sar['issues'])} unsupported reference(s). "
            "Fix them before approval."
        )
        st.dataframe(records_table(sar["issues"], ["kind", "value"]), hide_index=True)
    if sar["notes"]:
        st.dataframe(pd.DataFrame({"processing note": sar["notes"]}), hide_index=True)
    text = st.text_area(
        "Edit the draft, then save it as a new version (the review stamp is always kept)",
        value=sar["text"],
        height=360,
        max_chars=20_000,
        key=f"sar-{case_id}-{sar['version']}",
    )
    left, right = st.columns(2)
    if left.button("Save edited draft", type="primary"):
        client.save_sar(case_id, text)
        _changed("Edited draft saved as a new version.")
    if right.button("Regenerate draft"):
        client.generate_sar(case_id)
        _changed("SAR draft regenerated.")


def _actions_section(client: api.ApiClient, d: dict[str, Any]) -> None:
    st.subheader("Decision")
    with st.form(f"action-{d['case_id']}", clear_on_submit=True):
        action = st.radio(
            "Decision",
            ACTIONS,
            horizontal=True,
            captions=[ACTION_HELP[a] for a in ACTIONS],
        )
        note = st.text_area("Note (required, stored with the case)", max_chars=2_000)
        if st.form_submit_button("Record decision"):
            client.act(d["case_id"], str(action), note)
            _changed(f"Decision recorded: {action}.")
    st.dataframe(
        records_table(d["actions"], ["created_at", "actor", "action", "note"]), hide_index=True
    )


def _case_page(client: api.ApiClient) -> None:
    case_id = st.text_input("Case ID", value=st.session_state.get("case_id", ""), max_chars=17)
    if not case_id:
        st.info("Open a case from the queue, or type a case ID such as CASE-123.")
        return
    if not CASE_RE.match(case_id):
        st.error("Case IDs look like CASE-123.")
        return
    st.session_state["case_id"] = case_id
    d = _load_case(client, case_id)
    cols = st.columns(4)
    cols[0].metric("p_triage (ranking score)", f"{d['p_triage']:.3f}")
    cols[1].metric("Queue rank", "n/a" if d["rank"] is None else str(d["rank"]))
    cols[2].metric("Status", str(d["status"]))
    cols[3].metric("System 2 findings", str(len(d["findings"])))
    note = f"System 1 source: {d['triage_source']}."
    if d["p_final"] is not None:
        note += (
            f" Final combined score {d['p_final']:.3f} is informational and not used for "
            "ranking (it ranked worse on test, ADR-018)."
        )
    rel = d.get("reliability")
    if rel:
        note += (
            f" On the held-out test period, {rel['observed_rate']:.0%} of {rel['alerts']} alerts "
            f"with p_triage {rel['lo']:.3f}-{rel['hi']:.3f} were laundering."
        )
    if d["truncated"]:
        note += " Hub accounts were capped, so the case is truncated."
    st.caption(plain(note, max_len=600))

    st.subheader("Transaction graph (72 h before the alert)")
    pattern_accounts = [a for f in d["findings"] for a in f["accounts"]]
    st.plotly_chart(case_graph(d["edges"], d["subjects"], pattern_accounts))
    st.caption(
        "Red: the alert's payer and payee. Orange: accounts in a System 2 finding. "
        "Grey: other counterparties."
    )
    st.subheader("System 2 evidence")
    if d["findings"]:
        st.dataframe(findings_table(d["findings"]), hide_index=True)
    else:
        st.text("No predefined pattern was detected in this case's window.")
    st.subheader("Payments")
    st.caption(plain(f"{d['case_rows']} payments in the case; the most recent are listed."))
    st.dataframe(
        records_table(
            d["transactions"],
            ["date", "tx_id", "payer", "payee", "amount_usd", "payment_format"],
        ),
        hide_index=True,
    )
    _sar_section(client, d)
    _actions_section(client, d)


def _audit_page(client: api.ApiClient) -> None:
    data = client.audit()
    if data["chain_ok"]:
        st.success(f"Hash chain verified: {int(data['chain_records'])} records.")
    else:
        st.error(f"Hash chain broken: {plain(data['chain_error'])}")
    rows = [{**r, "details": json.dumps(r["details"], sort_keys=True)} for r in data["records"]]
    st.dataframe(
        records_table(rows, ["ts", "actor", "action", "case_id", "details", "hash"]),
        hide_index=True,
    )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="AML Sentinel", layout="wide")
    st.title("AML Sentinel")
    st.caption(
        "Synthetic data only. Scores are model estimates for prioritising review, "
        "not evidence of wrongdoing."
    )
    client = _sign_in()
    if client is None:
        st.info("Sign in with your API key in the sidebar.")
        return
    me = st.session_state["me"]
    st.sidebar.text(f"Signed in: {me['user']} ({me['role']})")
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    pages = ["Queue", "Case", *(["Audit log"] if me["role"] == "admin" else [])]
    if "goto" in st.session_state:
        st.session_state["page"] = st.session_state.pop("goto")
    page = st.sidebar.radio("View", pages, key="page")
    _flash()
    try:
        if page == "Queue":
            _queue_page(client)
        elif page == "Case":
            _case_page(client)
        else:
            _audit_page(client)
    except api.ApiError as exc:
        st.error(f"API error {exc.status_code}: {plain(exc)}")


if __name__ == "__main__":
    main()
