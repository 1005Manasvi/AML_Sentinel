"""Pure-ASGI middleware: security headers and request-body size limits.

Both are written against the raw ASGI interface, rather than Starlette's
``BaseHTTPMiddleware``, so they also apply to error responses and never buffer bodies.
"""

from __future__ import annotations

import json
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_HEADERS: Final[tuple[tuple[bytes, bytes], ...]] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    # The API only serves JSON; nothing it returns may run scripts or be framed.
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
    (b"cache-control", b"no-store"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    # Ignored by browsers over plain HTTP; takes effect once served behind TLS.
    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
)


class SecurityHeadersMiddleware:
    """Add :data:`SECURITY_HEADERS` to every HTTP response.

    Args:
        app: Wrapped ASGI app.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run the wrapped app, injecting headers into ``http.response.start``."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                names = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in SECURITY_HEADERS if k not in names]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        await self.app(scope, receive, send_with_headers)


async def _reject(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` with 413.

    ``Content-Length`` is checked up front; chunked or lying clients are caught
    while the body streams in.

    Args:
        app: Wrapped ASGI app.
        max_bytes: Largest accepted body.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the limit, then delegate."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await _reject(send, 413, "request body too large")
            return
        seen = 0
        started = rejected = False

        async def limited_receive() -> Message:
            # Raising here would be turned into a 400 by FastAPI's body parser, so the
            # middleware answers 413 itself and tells the app the client went away.
            nonlocal seen, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    rejected = True
                    if not started:
                        await _reject(send, 413, "request body too large")
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if rejected:
                return  # the 413 has already been sent
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # After a rejection the app sees a disconnect and may raise; the client
            # already has its 413. Anything else is a genuine error.
            if not rejected:
                raise


def _content_length(scope: Scope) -> int | None:
    headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
    for name, value in headers:
        if name.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
