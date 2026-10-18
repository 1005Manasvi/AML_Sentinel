"""Typed HTTP client for the AML Sentinel API, used by the dashboard.

The dashboard holds no data and no credentials of its own: each analyst signs in with
their own API key, so every action is attributed to them in the audit log and the
API's role checks apply unchanged.
"""

from __future__ import annotations

from typing import Any, Final

import httpx2

from aml_sentinel.exceptions import AmlSentinelError

DEFAULT_TIMEOUT_S: Final = 30.0


class ApiError(AmlSentinelError):
    """The API rejected a request.

    Args:
        status_code: HTTP status.
        message: Safe, human-readable reason.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _message(response: httpx2.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail[:300]
    if isinstance(detail, list) and detail and isinstance(detail[0], dict):
        first = detail[0]
        where = ".".join(str(p) for p in first.get("loc", []))
        return f"{where}: {first.get('msg', 'invalid value')}"[:300]
    return f"request failed ({response.status_code})"


class ApiClient:
    """Calls the API with one analyst's key.

    Args:
        base_url: API base URL (ignored when ``http`` is given).
        api_key: The analyst's key, sent as ``X-API-Key``.
        http: Pre-configured HTTP client (tests pass FastAPI's ``TestClient``).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        http: httpx2.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._http = http or httpx2.Client(base_url=base_url, timeout=timeout)
        self._headers = {"X-API-Key": api_key}

    def _call(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._http.request(
                method, path, json=json, params=params, headers=self._headers
            )
        except httpx2.HTTPError:
            raise ApiError(503, "the API is unreachable") from None
        if response.status_code >= 400:
            raise ApiError(response.status_code, _message(response))
        return response.json()

    def me(self) -> dict[str, Any]:
        """The signed-in user and role."""
        result: dict[str, Any] = self._call("GET", "/me")
        return result

    def alerts(
        self, *, escalated_only: bool, status: str | None = None, limit: int = 500
    ) -> dict[str, Any]:
        """A page of the ranked queue.

        Args:
            escalated_only: Hide auto-closed alerts.
            status: Optional status filter.
            limit: Page size.

        Returns:
            The ``AlertList`` payload.
        """
        params: dict[str, Any] = {"escalated_only": escalated_only, "limit": limit}
        if status:
            params["status"] = status
        result: dict[str, Any] = self._call("GET", "/alerts", params=params)
        return result

    def case(self, case_id: str) -> dict[str, Any]:
        """Case detail (recorded as a view in the audit log)."""
        result: dict[str, Any] = self._call("GET", f"/cases/{case_id}")
        return result

    def generate_sar(self, case_id: str) -> dict[str, Any]:
        """Generate a new SAR draft version."""
        result: dict[str, Any] = self._call("POST", f"/cases/{case_id}/sar")
        return result

    def save_sar(self, case_id: str, text: str) -> dict[str, Any]:
        """Save an edited SAR draft as a new version."""
        result: dict[str, Any] = self._call("PUT", f"/cases/{case_id}/sar", json={"text": text})
        return result

    def act(self, case_id: str, action: str, note: str) -> dict[str, Any]:
        """Approve, dismiss or escalate a case with a note."""
        body = {"action": action, "note": note}
        result: dict[str, Any] = self._call("POST", f"/cases/{case_id}/actions", json=body)
        return result

    def audit(self, limit: int = 200) -> dict[str, Any]:
        """Audit log (admin only)."""
        result: dict[str, Any] = self._call("GET", "/audit", params={"limit": limit})
        return result


def make_client(base_url: str, api_key: str) -> ApiClient:
    """Build the client the dashboard uses (a seam that tests replace).

    Args:
        base_url: API base URL.
        api_key: The analyst's key.

    Returns:
        The client.
    """
    return ApiClient(base_url, api_key)
