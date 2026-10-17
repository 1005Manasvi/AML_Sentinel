"""Provider-neutral LLM interface for SAR narratives, plus an Anthropic Claude adapter.

The drafting code depends only on :class:`LlmClient` (``complete(system, user)``),
so providers are swappable. The local template needs no LLM at all and is the
default (``LLM_PROVIDER=template``).

:class:`AnthropicClient` follows the official ``anthropic`` Python SDK:
``client.beta.messages.create`` with ``betas=["server-side-fallback-2026-07-01"]``
and ``fallbacks="default"``. If Claude's safety classifiers decline a request, the
API retries it server-side on Anthropic's recommended fallback model instead of
returning a refusal. Set ``use_fallbacks=False`` to turn this off.

Security:
    * Only pseudonymised facts reach this module (see
      :mod:`aml_sentinel.sar.pseudonymize`). The caller enforces this; the client
      never sees real account ids.
    * The API key comes from ``LLM_API_KEY`` as a ``SecretStr`` and is never logged.
    * Errors are re-raised as :class:`ExternalServiceError` with the status class
      only, never response bodies.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Protocol

from pydantic import SecretStr

from aml_sentinel.config import Settings
from aml_sentinel.exceptions import ConfigurationError, ExternalServiceError

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "claude-opus-5-5"
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
MAX_TOKENS: Final = 4096


class LlmClient(Protocol):
    """Anything that turns a system prompt and a user message into text."""

    @property
    def name(self) -> str:
        """Provider/model tag recorded on drafts and in the audit log."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Return the model's text reply."""
        ...


class AnthropicClient:
    """Claude via the official ``anthropic`` SDK (optional dependency).

    Args:
        api_key: Anthropic API key.
        model: Model id (default ``claude-opus-5-5``).
        timeout: Request timeout in seconds.
        use_fallbacks: Opt into server-side refusal fallbacks.
        sdk_client: Injected SDK client (tests).

    Raises:
        ConfigurationError: If the key is missing or the SDK is not installed.
    """

    def __init__(
        self,
        api_key: SecretStr | None,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        use_fallbacks: bool = True,
        sdk_client: Any | None = None,
    ) -> None:
        if sdk_client is None:
            if api_key is None or not api_key.get_secret_value().strip():
                raise ConfigurationError("LLM_API_KEY is required for the external SAR drafter")
            try:
                import anthropic
            except ImportError:
                raise ConfigurationError(
                    "anthropic is not installed; pip install -r requirements/llm.txt"
                ) from None
            sdk_client = anthropic.Anthropic(api_key=api_key.get_secret_value(), timeout=timeout)
        self._client = sdk_client
        self.model = model
        self.use_fallbacks = use_fallbacks

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> AnthropicClient:
        """Build from settings (``LLM_API_KEY``, ``LLM_MODEL``).

        Args:
            settings: Validated settings.
            **kwargs: Passed to the constructor.

        Returns:
            The client.
        """
        return cls(settings.llm_api_key, model=settings.llm_model, **kwargs)

    @property
    def name(self) -> str:
        """``anthropic:<model>``."""
        return f"anthropic:{self.model}"

    def complete(self, system: str, user: str) -> str:
        """Send one request and return the text.

        Args:
            system: System prompt.
            user: User message (delimited, pseudonymised case data).

        Returns:
            Concatenated text blocks of the reply.

        Raises:
            ExternalServiceError: On API errors, refusals or truncated output.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.use_fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        try:
            response = self._client.beta.messages.create(**kwargs)
        except Exception as exc:
            raise ExternalServiceError(f"LLM request failed: {_error_class(exc)}") from None
        if response.stop_reason == "refusal":
            raise ExternalServiceError("LLM declined the request (refusal)")
        if response.stop_reason == "max_tokens":
            raise ExternalServiceError("LLM output was truncated (max_tokens)")
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        logger.info("llm draft received", extra={"provider": self.name})
        return text


def _error_class(exc: Exception) -> str:
    # Map SDK exceptions by class name only, so no response body or key is echoed.
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")
