"""FastAPI application: authenticated alert queue, cases, SAR drafts and case actions.

Endpoints (all need ``X-API-Key`` except ``/health``):

========================================  =========  ==========================================
Endpoint                                  Role       Purpose
========================================  =========  ==========================================
``GET /health``                           none       Liveness
``GET /me``                               analyst    Who am I
``POST /score``                           analyst    System 1a score for a new transaction
``GET /alerts``                           analyst    Ranked queue (by System 1 ``p_triage``)
``GET /cases/{case_id}``                  analyst    Case detail with System 2 evidence
``POST /cases/{case_id}/sar``             analyst    Generate a SAR draft (new version)
``PUT /cases/{case_id}/sar``              analyst    Save an edited draft (new version)
``POST /cases/{case_id}/actions``         analyst    Approve, dismiss or escalate, with a note
``GET /audit``                            admin      Audit log and chain verification
========================================  =========  ==========================================

Security controls, outermost first: security headers on every response, request-body
size limit, ``Host`` allowlist, strict CORS allowlist, API-key authentication with
roles, and per-user / per-IP rate limiting. Interactive docs and the OpenAPI schema
are disabled, because they would be unauthenticated endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from aml_sentinel import __version__
from aml_sentinel.api.auth import ApiKeyStore, Principal, RateLimiter, Role
from aml_sentinel.api.middleware import BodySizeLimitMiddleware, SecurityHeadersMiddleware
from aml_sentinel.api.schemas import (
    CASE_ID_PATTERN,
    ActionOut,
    ActionRequest,
    AlertItem,
    AlertList,
    AuditRecordOut,
    AuditView,
    CaseDetail,
    CaseStatus,
    ContributionOut,
    EdgeOut,
    ExplanationOut,
    FindingOut,
    GroundingIssueOut,
    Health,
    Me,
    ReliabilityOut,
    SarEdit,
    SarOut,
    ScoreRequest,
    ScoreResponse,
    TxOut,
)
from aml_sentinel.api.service import (
    MAX_CASE_TRANSACTIONS,
    CaseConflictError,
    CaseNotFoundError,
    CaseService,
    ScoringUnavailableError,
    aggregate_edges,
    build_service,
    pattern_names,
)
from aml_sentinel.api.store import ActionRecord, SarRecord
from aml_sentinel.config import Settings
from aml_sentinel.exceptions import AmlSentinelError, DataValidationError, IntegrityError
from aml_sentinel.models.explain import Explanation
from aml_sentinel.sar.facts import TxFact

logger = logging.getLogger(__name__)

CaseId = Annotated[str, Path(pattern=CASE_ID_PATTERN)]
_AUTH_HEADERS: Final = {"WWW-Authenticate": "ApiKey"}


@dataclass(frozen=True, slots=True)
class AppContext:
    """Per-app dependencies (no module-level state).

    Attributes:
        settings: Settings.
        keys: Configured API keys.
        limiter: Rate limiter.
        service: Case service.
    """

    settings: Settings
    keys: ApiKeyStore
    limiter: RateLimiter
    service: CaseService


def _ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _throttle(ctx: AppContext, key: str) -> None:
    allowed, retry = ctx.limiter.allow(key)
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={"Retry-After": str(max(1, round(retry)))},
        )


def authenticated(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> Principal:
    """Resolve the caller from ``X-API-Key`` and apply rate limits.

    Args:
        request: Incoming request.
        x_api_key: Presented key.

    Returns:
        The principal.

    Raises:
        HTTPException: 401 for a missing or unknown key, 429 when throttled.
    """
    ctx = _ctx(request)
    principal = ctx.keys.authenticate(x_api_key)
    if principal is None:
        # Failed attempts spend the client address's budget, which slows guessing.
        _throttle(ctx, f"ip:{_client_ip(request)}")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing or invalid API key", headers=_AUTH_HEADERS
        )
    _throttle(ctx, f"user:{principal.user}")
    return principal


def require(role: Role) -> Callable[[Principal], Principal]:
    """Dependency factory enforcing a minimum role.

    Args:
        role: Minimum role.

    Returns:
        A FastAPI dependency returning the principal.
    """

    def check(principal: Annotated[Principal, Depends(authenticated)]) -> Principal:
        if not principal.has(role):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role")
        return principal

    return check


Analyst = Annotated[Principal, Depends(require(Role.ANALYST))]
Admin = Annotated[Principal, Depends(require(Role.ADMIN))]


def _tx_out(t: TxFact) -> TxOut:
    return TxOut(
        tx_id=t.tx_id,
        date=t.date,
        payer=t.payer,
        payee=t.payee,
        amount_usd=t.amount_usd,
        payment_format=t.payment_format,
    )


def _sar_out(r: SarRecord) -> SarOut:
    return SarOut(
        case_id=r.case_id,
        version=r.version,
        text=r.text,
        source=r.source,
        grounded=r.is_grounded,
        issues=[GroundingIssueOut(kind=k, value=v) for k, v in r.issues],
        notes=list(r.notes),
        author=r.author,
        created_at=r.created_at,
    )


def _explanation_out(e: Explanation | None) -> ExplanationOut | None:
    if e is None:
        return None
    return ExplanationOut(
        model=e.model,
        base_log_odds=e.base_log_odds,
        log_odds=e.log_odds,
        top=[
            ContributionOut(feature=c.feature, value=c.value, contribution=c.contribution)
            for c in e.top
        ],
        others=e.others,
    )


def _action_out(r: ActionRecord) -> ActionOut:
    return ActionOut(actor=r.actor, action=r.action, note=r.note, created_at=r.created_at)


def _reliability_out(service: CaseService, p: float) -> ReliabilityOut | None:
    band = service.queue.band_of(p)
    if band is None:
        return None
    return ReliabilityOut(lo=band.lo, hi=band.hi, alerts=band.count, observed_rate=band.frac_pos)


def _install_error_handlers(app: FastAPI) -> None:
    def detail(code: int, message: str) -> JSONResponse:
        return JSONResponse({"detail": message}, status_code=code)

    async def not_found(_: Request, exc: Exception) -> JSONResponse:
        return detail(404, str(exc))

    async def conflict(_: Request, exc: Exception) -> JSONResponse:
        return detail(409, str(exc))

    async def unavailable(_: Request, exc: Exception) -> JSONResponse:
        return detail(501, str(exc))

    async def invalid(_: Request, exc: Exception) -> JSONResponse:
        # Project exceptions carry safe messages (counts, names), never data values.
        return detail(422, str(exc))

    async def internal(_: Request, exc: Exception) -> JSONResponse:
        logger.error("request failed", extra={"error_type": type(exc).__name__})
        return detail(500, "internal error")

    async def bad_request(_: Request, exc: Exception) -> JSONResponse:
        # FastAPI's default 422 body echoes the submitted input; return only where
        # and why validation failed.
        errors = exc.errors() if isinstance(exc, RequestValidationError) else []
        clean = [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg"), "type": e.get("type")}
            for e in errors
        ]
        return JSONResponse({"detail": clean}, status_code=422)

    app.add_exception_handler(CaseNotFoundError, not_found)
    app.add_exception_handler(CaseConflictError, conflict)
    app.add_exception_handler(ScoringUnavailableError, unavailable)
    app.add_exception_handler(DataValidationError, invalid)
    app.add_exception_handler(AmlSentinelError, internal)
    app.add_exception_handler(RequestValidationError, bad_request)


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(request: Request) -> Health:
        _throttle(_ctx(request), f"ip:{_client_ip(request)}")
        return Health()

    @app.get("/me")
    def me(principal: Analyst) -> Me:
        return Me(user=principal.user, role=principal.role.value)

    @app.post("/score")
    def score(body: ScoreRequest, request: Request, _: Analyst) -> ScoreResponse:
        result = _ctx(request).service.scorer.score(body)
        return ScoreResponse(
            p_tabular=result.p_tabular,
            model=result.model,
            alert=result.alert,
            alert_threshold=result.alert_threshold,
            rules_fired=None if result.rules_fired is None else list(result.rules_fired),
            history_rows=result.history_rows,
            latency_ms=round(result.latency_ms, 2),
        )

    @app.get("/alerts")
    def alerts(
        request: Request,
        _: Analyst,
        status_filter: Annotated[CaseStatus | None, Query(alias="status")] = None,
        escalated_only: bool = False,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> AlertList:
        service = _ctx(request).service
        rows = [
            (rank, entry, st)
            for rank, entry, st in service.alerts()
            if (not escalated_only or entry.escalated)
            and (status_filter is None or st == status_filter)
        ]
        items = [
            AlertItem(
                rank=rank,
                case_id=entry.case_id,
                tx_id=entry.tx_id,
                timestamp=entry.timestamp,
                p_triage=entry.p_triage,
                escalated=entry.escalated,
                status=st,
                n_findings=None if entry.evidence is None else entry.evidence["n_findings"],
                patterns=pattern_names(entry),
            )
            for rank, entry, st in rows[offset : offset + limit]
        ]
        q = service.queue
        return AlertList(
            dataset=q.dataset.value,
            triage_source=q.triage_source,
            escalation_threshold=q.threshold,
            escalate_share=q.escalate_share,
            total=len(rows),
            items=items,
        )

    @app.get("/cases/{case_id}")
    def case_detail(case_id: CaseId, request: Request, principal: Analyst) -> CaseDetail:
        service = _ctx(request).service
        view = service.case(case_id, principal.user)
        facts, entry = view.facts, view.entry
        sar = service.latest_sar(case_id)
        return CaseDetail(
            case_id=case_id,
            rank=view.rank,
            alert=_tx_out(facts.alert),
            subjects=list(facts.subjects),
            p_triage=entry.p_triage,
            triage_source=service.queue.triage_source,
            p_final=entry.p_final,
            reliability=_reliability_out(service, entry.p_triage),
            escalated=entry.escalated,
            status=view.status,
            case_rows=len(facts.transactions),
            truncated=view.case.truncated,
            findings=[
                FindingOut(
                    kind=p.kind,
                    accounts=list(p.accounts),
                    tx_ids=list(p.tx_ids),
                    total_usd=p.total_usd,
                    first_date=p.first_date,
                    last_date=p.last_date,
                    size=p.size,
                )
                for p in facts.patterns
            ],
            transactions=[_tx_out(t) for t in facts.transactions[-MAX_CASE_TRANSACTIONS:]],
            edges=[
                EdgeOut(payer=str(a), payee=str(b), count=int(n), total_usd=round(float(usd), 2))
                for a, b, n, usd in aggregate_edges(view.case).to_numpy(dtype=object)
            ],
            actions=[_action_out(a) for a in service.actions(case_id)],
            sar=None if sar is None else _sar_out(sar),
            explanation=_explanation_out(view.explanation),
            explanation_unavailable=view.explanation_unavailable,
        )

    @app.post("/cases/{case_id}/sar", status_code=status.HTTP_201_CREATED)
    def generate_sar(case_id: CaseId, request: Request, principal: Analyst) -> SarOut:
        return _sar_out(_ctx(request).service.generate_sar(case_id, principal.user))

    @app.put("/cases/{case_id}/sar")
    def save_sar(case_id: CaseId, body: SarEdit, request: Request, principal: Analyst) -> SarOut:
        return _sar_out(_ctx(request).service.save_sar(case_id, body.text, principal.user))

    @app.post("/cases/{case_id}/actions", status_code=status.HTTP_201_CREATED)
    def act(
        case_id: CaseId, body: ActionRequest, request: Request, principal: Analyst
    ) -> ActionOut:
        record = _ctx(request).service.act(case_id, body.action, body.note, principal.user)
        return _action_out(record)

    @app.get("/audit")
    def audit(
        request: Request, _: Admin, limit: Annotated[int, Query(ge=1, le=1000)] = 200
    ) -> AuditView:
        service = _ctx(request).service
        error: str | None = None
        try:
            verified = service.verify_audit()
        except IntegrityError as exc:
            verified, error = 0, str(exc)
        records = service.audit_records()[::-1][:limit]
        return AuditView(
            chain_ok=error is None,
            chain_records=verified,
            chain_error=error,
            records=[
                AuditRecordOut(
                    ts=r.ts,
                    actor=r.actor,
                    action=r.action,
                    case_id=r.case_id,
                    details=dict(r.details),
                    hash=r.hash,
                )
                for r in records
            ],
        )


def create_app(
    settings: Settings,
    *,
    service: CaseService | None = None,
    keys: ApiKeyStore | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the API.

    Args:
        settings: Settings.
        service: Case service (built from settings by default).
        keys: API keys (parsed from ``API_KEYS`` by default).
        limiter: Rate limiter (from ``API_RATE_LIMIT_PER_MINUTE`` by default).

    Returns:
        The application.

    Raises:
        ConfigurationError: If ``API_KEYS`` is missing or malformed (fail fast,
            before any data is loaded).
    """
    secret = None if settings.api_keys is None else settings.api_keys.get_secret_value()
    keys = keys or ApiKeyStore.parse(secret)
    service = service or build_service(settings)
    limiter = limiter or RateLimiter(settings.api_rate_limit_per_minute)

    app = FastAPI(
        title="AML Sentinel",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.ctx = AppContext(settings=settings, keys=keys, limiter=limiter, service=service)
    # add_middleware wraps: the last one added is the outermost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["X-API-Key", "Content-Type"],
        max_age=600,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.api_allowed_hosts))
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.api_max_request_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    _install_error_handlers(app)
    _register_routes(app)
    logger.info(
        "api ready",
        extra={"dataset": service.queue.dataset.value, "alerts": len(service.queue)},
    )
    return app
