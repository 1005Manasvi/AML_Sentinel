"""API security: keys, roles (auth matrix), rate limits, headers, CORS, hosts, sizes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aml_sentinel.api.app import create_app
from aml_sentinel.api.auth import (
    KEY_PREFIX,
    ApiKeyStore,
    Principal,
    RateLimiter,
    Role,
    generate_api_key,
    hash_api_key,
)
from aml_sentinel.api.middleware import SECURITY_HEADERS
from aml_sentinel.config import load_settings
from aml_sentinel.exceptions import ConfigurationError
from tests.conftest import ADMIN_KEY, ANALYST_KEY, ServedSample, make_service

ANALYST = {"X-API-Key": ANALYST_KEY}
ADMIN = {"X-API-Key": ADMIN_KEY}
CASE = "CASE-1"
SCORE_BODY = {
    "timestamp": "2022-09-09T12:00:00",
    "from_bank": "001",
    "from_account": "A1",
    "to_bank": "002",
    "to_account": "B2",
    "amount_received": 10.0,
    "receiving_currency": "US Dollar",
    "amount_paid": 10.0,
    "payment_currency": "US Dollar",
    "payment_format": "Wire",
}
# (method, path, json body, minimum role)
ENDPOINTS = [
    ("GET", "/me", None, Role.ANALYST),
    ("POST", "/score", SCORE_BODY, Role.ANALYST),
    ("GET", "/alerts", None, Role.ANALYST),
    ("GET", f"/cases/{CASE}", None, Role.ANALYST),
    ("POST", f"/cases/{CASE}/sar", None, Role.ANALYST),
    ("PUT", f"/cases/{CASE}/sar", {"text": "x"}, Role.ANALYST),
    ("POST", f"/cases/{CASE}/actions", {"action": "dismiss", "note": "fp"}, Role.ANALYST),
    ("GET", "/audit", None, Role.ADMIN),
]


# ---------------------------------------------------------------- key store
def test_generated_keys_are_random_and_prefixed() -> None:
    a, b = generate_api_key(), generate_api_key()
    assert a != b
    assert a.startswith(KEY_PREFIX)
    assert len(a) >= 40


def test_key_store_authenticates_by_digest_only() -> None:
    store = ApiKeyStore.parse(f"alice:analyst:{hash_api_key('k1')},root:admin:{hash_api_key('k2')}")
    assert store.authenticate("k1") == Principal("alice", Role.ANALYST)
    assert store.authenticate("k2") == Principal("root", Role.ADMIN)
    assert store.authenticate("k3") is None
    assert store.authenticate(None) is None
    assert store.authenticate("") is None
    assert store.authenticate("x" * 10_000) is None


@pytest.mark.parametrize(
    "spec",
    [
        None,
        "",
        "alice:analyst",
        "alice:superuser:" + "a" * 64,
        "bad user:analyst:" + "a" * 64,
        "alice:analyst:nothex",
        f"alice:analyst:{'a' * 64},alice:admin:{'b' * 64}",
        f"alice:analyst:{'a' * 64},bob:admin:{'a' * 64}",
    ],
)
def test_key_store_rejects_bad_config_without_echoing_it(spec: str | None) -> None:
    with pytest.raises(ConfigurationError) as info:
        ApiKeyStore.parse(spec)
    assert "a" * 64 not in str(info.value)


def test_roles_are_ordered() -> None:
    assert Principal("x", Role.ADMIN).has(Role.ANALYST)
    assert not Principal("x", Role.ANALYST).has(Role.ADMIN)


def test_api_refuses_to_start_without_keys(served_sample: ServedSample, tmp_path: Path) -> None:
    settings = served_sample.settings.model_copy(update={"api_keys": None})
    with pytest.raises(ConfigurationError, match="API_KEYS is required"):
        create_app(settings, service=make_service(served_sample, tmp_path))


# ---------------------------------------------------------------- rate limiter
class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_bursts_then_refills() -> None:
    clock = FakeClock()
    limiter = RateLimiter(3, clock=clock)
    assert [limiter.allow("u")[0] for _ in range(4)] == [True, True, True, False]
    allowed, retry = limiter.allow("u")
    assert not allowed
    assert 0 < retry <= 20
    clock.now += 20.0  # 3/min refills one token every 20 s
    assert limiter.allow("u")[0]
    assert limiter.allow("other")[0]  # buckets are per caller


def test_rate_limiter_bounds_memory() -> None:
    limiter = RateLimiter(5, clock=FakeClock(), max_keys=10)
    for i in range(50):
        limiter.allow(f"ip:{i}")
    assert len(limiter._buckets) <= 10


def test_api_returns_429_with_retry_after(served_sample: ServedSample, tmp_path: Path) -> None:
    app = create_app(
        served_sample.settings,
        service=make_service(served_sample, tmp_path),
        limiter=RateLimiter(2, clock=FakeClock()),
    )
    c = TestClient(app)
    codes = [c.get("/me", headers=ANALYST).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert int(c.get("/me", headers=ANALYST).headers["retry-after"]) >= 1


def test_failed_auth_is_throttled_per_client(served_sample: ServedSample, tmp_path: Path) -> None:
    app = create_app(
        served_sample.settings,
        service=make_service(served_sample, tmp_path),
        limiter=RateLimiter(3, clock=FakeClock()),
    )
    c = TestClient(app)
    codes = [c.get("/me", headers={"X-API-Key": f"guess-{i}"}).status_code for i in range(5)]
    assert codes == [401, 401, 401, 429, 429]


# ---------------------------------------------------------------- auth matrix
@pytest.mark.parametrize(("method", "path", "body", "role"), ENDPOINTS)
def test_auth_matrix(
    client: TestClient, method: str, path: str, body: dict[str, object] | None, role: Role
) -> None:
    no_key = client.request(method, path, json=body)
    assert no_key.status_code == 401
    assert no_key.headers["www-authenticate"] == "ApiKey"
    wrong = client.request(method, path, json=body, headers={"X-API-Key": "amls_wrong"})
    assert wrong.status_code == 401
    analyst = client.request(method, path, json=body, headers=ANALYST)
    admin = client.request(method, path, json=body, headers=ADMIN)
    if role is Role.ADMIN:
        assert analyst.status_code == 403
    else:
        assert analyst.status_code not in (401, 403)
    assert admin.status_code not in (401, 403)


def test_health_is_public(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_docs_and_schema_are_not_exposed(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


# ---------------------------------------------------------------- transport hardening
@pytest.mark.parametrize("path", ["/health", "/alerts", "/cases/CASE-1"])
def test_security_headers_on_every_response(client: TestClient, path: str) -> None:
    r = client.get(path, headers=ANALYST)
    for name, value in SECURITY_HEADERS:
        assert r.headers[name.decode()] == value.decode()


def test_cors_allows_only_listed_origins(client: TestClient) -> None:
    pre = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "X-API-Key"}
    ok = client.options("/alerts", headers={"Origin": "http://localhost:8501", **pre})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:8501"
    evil = client.options("/alerts", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in evil.headers
    simple = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_unknown_host_header_is_rejected(client: TestClient) -> None:
    assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 400


def test_oversized_body_rejected_by_content_length(client: TestClient) -> None:
    big = b"{" + b" " * 70_000 + b"}"
    r = client.post("/score", content=big, headers={**ANALYST, "Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.headers["x-content-type-options"] == "nosniff"


def test_oversized_streamed_body_rejected(client: TestClient) -> None:
    def chunks():  # type: ignore[no-untyped-def]
        for _ in range(20):
            yield b" " * 8_192

    r = client.post(
        "/score", content=chunks(), headers={**ANALYST, "Content-Type": "application/json"}
    )
    assert r.status_code == 413


def test_validation_errors_do_not_echo_input(client: TestClient) -> None:
    secret_like = "<script>alert(1)</script>"
    r = client.post("/score", headers=ANALYST, json={**SCORE_BODY, "from_account": secret_like})
    assert r.status_code == 422
    assert secret_like not in r.text
    assert r.json()["detail"][0]["loc"] == ["body", "from_account"]


def test_case_id_is_validated(client: TestClient) -> None:
    assert client.get("/cases/CASE-1;DROP", headers=ANALYST).status_code in (404, 422)
    assert client.get("/cases/..%2Faudit", headers=ANALYST).status_code in (404, 422)
    assert client.get("/cases/CASE-999999999", headers=ANALYST).status_code == 404


def test_requests_reject_unknown_fields(client: TestClient) -> None:
    r = client.post(
        "/cases/CASE-1/actions",
        headers=ANALYST,
        json={"action": "dismiss", "note": "fine", "actor": "someone-else"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------- settings
@pytest.mark.parametrize(
    "origins", [("*",), ("http://ok:8501/path",), ("javascript:alert(1)",), ("http://*.x",)]
)
def test_settings_reject_open_cors(origins: tuple[str, ...]) -> None:
    with pytest.raises(ConfigurationError, match="API_CORS_ORIGINS"):
        load_settings(_env_file=None, api_cors_origins=origins)


def test_settings_reject_wildcard_hosts() -> None:
    with pytest.raises(ConfigurationError, match="API_ALLOWED_HOSTS"):
        load_settings(_env_file=None, api_allowed_hosts=("*",))


def test_api_keys_never_appear_in_settings_repr() -> None:
    settings = load_settings(_env_file=None, api_keys=f"a:analyst:{'f' * 64}")
    assert "f" * 64 not in repr(settings)
