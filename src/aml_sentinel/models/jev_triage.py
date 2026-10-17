"""Jev triage adapter: **benchmark-only**, **synthetic-data-only**, disabled by default.

Jev is a closed model hosted by TypeSafe AI. The API used here follows TypeSafe's
official reference (https://docs.typesafe.ai/api): ``POST
https://api.typesafe.ai/v1/systemone`` with ``Authorization: Bearer <key>`` and a body
of ``{model, state, questions}``. Answers come back under ``answers`` with the same
``choice`` / ``score`` / ``noul`` shapes as Laya.

Security and data protection (non-negotiable):

* Disabled unless ``JEV_ENABLED=true`` **and** ``JEV_API_KEY`` is set.
* Refuses any batch whose ``is_synthetic`` flag is not explicitly true, raising
  :class:`SyntheticDataRequiredError`. **Never send real customer data to Jev.**
* What leaves the machine: one numeric alert summary per request (no account IDs,
  bank codes, names or free text; see :mod:`aml_sentinel.models.summarizer`), the
  fixed questions, and the model name. Nothing else.
* Requests use HTTPS only. The key is never logged, and error messages include only
  the HTTP status.
* Results are not reproducible offline and may change between Jev versions, so
  every comparison with Laya must state that.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Final

from pydantic import SecretStr

from aml_sentinel.config import Settings
from aml_sentinel.exceptions import (
    ConfigurationError,
    ExternalServiceError,
    ModelOutputError,
    SyntheticDataRequiredError,
)
from aml_sentinel.models.decisions import QUESTIONS
from aml_sentinel.models.laya_triage import checked_summaries, decisions_to_output
from aml_sentinel.models.triage import AlertBatch, TriageOutput

logger = logging.getLogger(__name__)

JEV_ENDPOINT: Final = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL: Final = "jev-latest"
MAX_RESPONSE_BYTES: Final = 1 << 20

Transport = Callable[[str, Mapping[str, str], bytes, float], bytes]


def https_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> bytes:
    """POST ``body`` over HTTPS and return the response body.

    Args:
        url: Endpoint; must be ``https://``.
        headers: Request headers.
        body: JSON body.
        timeout: Seconds before giving up.

    Returns:
        Raw response bytes (at most 1 MiB).

    Raises:
        ConfigurationError: If ``url`` is not HTTPS.
        ExternalServiceError: On HTTP errors, network failures or oversized responses.
    """
    if not url.startswith("https://"):
        raise ConfigurationError("Jev endpoint must use HTTPS")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        # Scheme is checked above, so this cannot open file:// or other local URLs.
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            data: bytes = resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ExternalServiceError(f"Jev returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ExternalServiceError("Jev request failed") from None
    if len(data) > MAX_RESPONSE_BYTES:
        raise ExternalServiceError("Jev response too large")
    return data


class JevTriage:
    """Jev as a ``Triage`` model, restricted to synthetic benchmarking.

    Args:
        api_key: TypeSafe API key.
        enabled: Value of ``JEV_ENABLED``; the adapter refuses to exist when false.
        model: Jev model name or alias.
        transport: HTTP function (injectable for tests).
        timeout: Per-request timeout in seconds.

    Raises:
        ConfigurationError: If disabled or the key is missing.
    """

    def __init__(
        self,
        api_key: SecretStr | None,
        *,
        enabled: bool,
        model: str = DEFAULT_MODEL,
        transport: Transport = https_transport,
        timeout: float = 30.0,
    ) -> None:
        if not enabled:
            raise ConfigurationError("Jev is disabled (set JEV_ENABLED=true to benchmark)")
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationError("JEV_API_KEY is required when Jev is enabled")
        self._key = api_key
        self.model = model
        self._transport = transport
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> JevTriage:
        """Build from validated settings.

        Args:
            settings: Application settings.
            **kwargs: Passed to the constructor (``model``, ``transport``, ...).

        Returns:
            The adapter.
        """
        return cls(settings.jev_api_key, enabled=settings.jev_enabled, **kwargs)

    @property
    def name(self) -> str:
        """``jev[<model>]``."""
        return f"jev[{self.model}]"

    def triage(self, batch: AlertBatch) -> TriageOutput:
        """Ask Jev the shared questions for every alert summary.

        Args:
            batch: Must have ``is_synthetic=True`` and one summary per alert.

        Returns:
            Same structure as :class:`~aml_sentinel.models.laya_triage.LayaTriage`.

        Raises:
            SyntheticDataRequiredError: If the batch is not explicitly synthetic.
            DataValidationError: If summaries are missing or invalid.
            ExternalServiceError: On transport failures.
            ModelOutputError: On malformed responses.
        """
        if batch.is_synthetic is not True:
            raise SyntheticDataRequiredError(
                "Jev is benchmark-only and accepts synthetic data only; "
                "never send real customer data to a hosted third-party model"
            )
        summaries = checked_summaries(batch)
        headers = {
            "Authorization": f"Bearer {self._key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        results: list[dict[str, Any]] = []
        for summary in summaries:
            body = json.dumps({"model": self.model, "state": summary, "questions": QUESTIONS})
            raw = self._transport(JEV_ENDPOINT, headers, body.encode("utf-8"), self._timeout)
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                raise ModelOutputError("Jev response is not JSON") from None
            if not isinstance(data, dict):
                raise ModelOutputError("Jev response is not an object")
            results.append({"answers": data.get("answers"), "usage": {}})
        logger.info("jev batch scored", extra={"alerts": len(summaries), "model": self.model})
        return decisions_to_output(results, len(summaries))
