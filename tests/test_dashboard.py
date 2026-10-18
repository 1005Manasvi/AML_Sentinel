"""Dashboard: escaping helpers, graph, API client, and the Streamlit app end to end."""

from __future__ import annotations

import ast
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from aml_sentinel.dashboard import client as api
from aml_sentinel.dashboard.client import ApiClient, ApiError
from aml_sentinel.dashboard.views import (
    case_graph,
    findings_table,
    plain,
    queue_table,
    records_table,
)
from tests.conftest import ADMIN_KEY, ANALYST_KEY

APP = Path(__file__).resolve().parents[1] / "src" / "aml_sentinel" / "dashboard" / "app.py"
# Importing main (instead of running app.py as a script) lets coverage attribute the lines.
LAUNCHER = "from aml_sentinel.dashboard.app import main\n\nmain()\n"


# ---------------------------------------------------------------- views
@pytest.mark.parametrize(
    "hostile",
    [
        "![x](https://evil.example/?q=leak)",
        "[click](javascript:alert(1))",
        "<img src=x onerror=alert(1)>",
        "**bold** `code` $x^2$",
    ],
)
def test_plain_neutralises_markdown_and_html(hostile: str) -> None:
    out = plain(hostile)
    for ch in "![]()<>*`$":
        assert ch not in out.replace(f"\\{ch}", "")


def test_plain_flattens_and_truncates() -> None:
    assert plain("a\n\nb", max_len=10) == "a b"
    assert len(plain("x" * 1000, max_len=50)) == 50


def test_case_graph_escapes_labels_and_keeps_subject_edges() -> None:
    subjects = ["1:AAAA", "2:BBBB"]
    edges = [{"payer": "1:AAAA", "payee": "2:BBBB", "count": 1, "total_usd": 1.0}]
    edges += [
        {"payer": f"9:N{i}", "payee": f"9:M{i}", "count": 1, "total_usd": 1000.0 + i}
        for i in range(30)
    ]
    edges.append({"payer": "<b>x</b>", "payee": "1:AAAA", "count": 1, "total_usd": 5.0})
    fig = case_graph(edges, subjects, ["9:N0"], max_edges=5)
    nodes = fig.data[1]
    labels = " ".join(nodes.hovertext)
    assert "<b>" not in labels
    assert "&lt;b&gt;" in labels
    assert "1:AAAA" in labels
    assert "2:BBBB" in labels
    assert "hidden" in fig.layout.title.text
    colors = dict(zip(nodes.hovertext, nodes.marker.color, strict=True))
    assert colors[next(k for k in colors if k.startswith("1:AAAA"))] == "#c0392b"


def test_tables_have_stable_columns_even_when_empty() -> None:
    assert list(queue_table([]).columns)[:3] == ["rank", "case", "time"]
    assert findings_table([]).columns[0] == "pattern"
    assert list(records_table([{"a": 1, "b": 2}], ["b"]).columns) == ["b"]


# ---------------------------------------------------------------- client
def _mock(status: int, body: object) -> ApiClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["x-api-key"] == "k"
        return httpx2.Response(status, json=body)

    http = httpx2.Client(base_url="http://api", transport=httpx2.MockTransport(handler))
    return ApiClient("http://api", "k", http=http)


def test_client_maps_error_details() -> None:
    with pytest.raises(ApiError, match="case not found") as info:
        _mock(404, {"detail": "case not found"}).case("CASE-1")
    assert info.value.status_code == 404
    with pytest.raises(ApiError, match=r"body\.note: too short"):
        _mock(422, {"detail": [{"loc": ["body", "note"], "msg": "too short"}]}).act(
            "CASE-1", "dismiss", ""
        )
    with pytest.raises(ApiError, match=r"request failed \(500\)"):
        _mock(500, {"detail": None}).me()


def test_client_reports_unreachable_api() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    http = httpx2.Client(base_url="http://api", transport=httpx2.MockTransport(boom))
    with pytest.raises(ApiError, match="unreachable") as info:
        ApiClient("http://api", "k", http=http).alerts(escalated_only=True)
    assert info.value.status_code == 503


def test_make_client_builds_a_real_client() -> None:
    assert isinstance(api.make_client("http://127.0.0.1:1", "k"), ApiClient)


# ---------------------------------------------------------------- Streamlit app
def _app(client: TestClient, key: str, role: str, user: str) -> AppTest:
    at = AppTest.from_string(LAUNCHER, default_timeout=60)
    at.session_state["client"] = ApiClient("", key, http=client)
    at.session_state["me"] = {"user": user, "role": role}
    return at.run()


def _button(at: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(b for b in at.button if b.label == label)


def test_dashboard_requires_sign_in() -> None:
    at = AppTest.from_string(LAUNCHER, default_timeout=60).run()
    assert not at.exception
    assert any("Sign in" in i.value for i in at.info)


def test_dashboard_sign_in_validates_the_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "make_client", lambda _url, key: ApiClient("", key, http=client))
    at = AppTest.from_string(LAUNCHER, default_timeout=60).run()
    at.sidebar.text_input[0].input("amls_wrong")
    at.sidebar.form_submit_button[0].click().run()
    assert any("Sign-in failed" in e.value for e in at.sidebar.error)
    at.sidebar.text_input[0].input(ANALYST_KEY)
    at.sidebar.form_submit_button[0].click().run()
    assert not at.exception
    assert at.session_state["me"] == {"user": "ana", "role": "analyst"}


def test_dashboard_queue_case_sar_and_decision(client: TestClient) -> None:
    at = _app(client, ANALYST_KEY, "analyst", "ana")
    assert not at.exception
    assert at.sidebar.radio[0].options == ["Queue", "Case"]  # no audit view for analysts
    queue = at.dataframe[0].value
    assert list(queue["status"].unique()) == ["open"]  # escalated-only by default
    case_id = str(queue["case"].iloc[0])

    _button(at, "Open case").click().run()
    assert not at.exception
    assert at.session_state["case_id"] == case_id
    assert any(m.label.startswith("p_triage") for m in at.metric)

    _button(at, "Generate SAR draft").click().run()
    assert not at.exception
    assert any("Grounding check passed" in s.value for s in at.success)
    draft = next(t for t in at.text_area if t.key and t.key.startswith("sar-"))
    assert draft.value.startswith("DRAFT - requires analyst review")

    draft.input(draft.value + "\nExtra wire of $1,234,567.00.").run()
    _button(at, "Save edited draft").click().run()
    assert any("Grounding check failed" in e.value for e in at.error)

    decision = next(r for r in at.radio if r.label == "Decision")
    decision.set_value("approve")
    next(t for t in at.text_area if t.label.startswith("Note")).input("file it")
    at.form_submit_button[-1].click().run()
    assert any("409" in e.value for e in at.error)  # ungrounded draft cannot be approved

    decision = next(r for r in at.radio if r.label == "Decision")
    decision.set_value("escalate")
    next(t for t in at.text_area if t.label.startswith("Note")).input("senior review please")
    at.form_submit_button[-1].click().run()
    assert not at.exception
    assert any("Decision recorded" in s.value for s in at.success)
    history = next(d.value for d in at.dataframe if "actor" in d.value.columns)
    assert list(history["action"]) == ["escalate"]


def test_dashboard_case_id_is_validated(client: TestClient) -> None:
    at = _app(client, ANALYST_KEY, "analyst", "ana")
    at.sidebar.radio[0].set_value("Case").run()
    at.text_input[0].input("CASE-1; DROP").run()
    assert any("look like CASE-123" in e.value for e in at.error)
    at.text_input[0].input("CASE-999999999").run()
    assert any("API error 404" in e.value for e in at.error)


def test_dashboard_audit_view_for_admin(client: TestClient) -> None:
    at = _app(client, ADMIN_KEY, "admin", "adm")
    assert at.sidebar.radio[0].options == ["Queue", "Case", "Audit log"]
    at.sidebar.radio[0].set_value("Audit log").run()
    assert not at.exception
    assert any("Hash chain verified" in s.value for s in at.success)


def test_dashboard_toggle_shows_auto_closed_alerts(client: TestClient) -> None:
    at = _app(client, ANALYST_KEY, "analyst", "ana")
    at.toggle[0].set_value(False).run()
    statuses = set(at.dataframe[0].value["status"])
    assert "auto_closed" in statuses


def test_dashboard_sign_out_clears_the_session(client: TestClient) -> None:
    at = _app(client, ANALYST_KEY, "analyst", "ana")
    _button(at, "Sign out").click().run()
    assert "client" not in at.session_state


# ---------------------------------------------------------------- static guards
DASHBOARD_SRC = sorted((APP.parent).glob("*.py"))


def _calls(path: Path) -> list[ast.Call]:
    return [n for n in ast.walk(ast.parse(path.read_text("utf-8"))) if isinstance(n, ast.Call)]


def test_dashboard_never_renders_raw_html_or_markdown() -> None:
    # Checked on the syntax tree, so docstrings that mention these names do not count.
    for path in DASHBOARD_SRC:
        for call in _calls(path):
            assert all(k.arg != "unsafe_allow_html" for k in call.keywords), path.name
            if isinstance(call.func, ast.Attribute):
                assert call.func.attr not in {"html", "markdown", "write"}, path.name


def test_dashboard_never_caches_across_sessions() -> None:
    # A shared st.cache_* would serve one analyst's data (or client) to another.
    for path in DASHBOARD_SRC:
        for call in _calls(path):
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert not name.startswith("cache"), path.name


def test_streamlit_config_is_hardened() -> None:
    import tomllib

    config = tomllib.loads((APP.parents[3] / ".streamlit" / "config.toml").read_text("utf-8"))
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["enableXsrfProtection"] is True
    assert config["client"]["showErrorDetails"] == "none"
    assert config["runner"]["magicEnabled"] is False
