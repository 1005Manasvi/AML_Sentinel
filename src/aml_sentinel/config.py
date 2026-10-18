"""Application settings loaded from environment variables (and an optional ``.env``).

Settings are immutable and created once at the edge of the program (CLI, API
startup), then passed down explicitly. There is deliberately no module-level settings
singleton, so tests can build isolated settings without touching global state.

Security note:
    Secrets are typed as :class:`pydantic.SecretStr` so they never appear in ``repr``
    or logs. :func:`load_settings` rebuilds validation errors from field locations
    and messages only, because pydantic's default error text echoes the *input*,
    which could include a secret.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import ConfigurationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
# scheme://host[:port] and nothing else: a wildcard or path would widen CORS.
_ORIGIN_RE: Final = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


class Settings(BaseSettings):
    """Validated, immutable runtime configuration.

    Attributes:
        environment: Deployment environment name.
        log_level: Minimum log level.
        log_json: Emit structured JSON logs when true.
        data_dir: Root data directory (``raw/`` and ``processed/`` live under it).
        models_dir: Directory for trusted local model artifacts.
        random_seed: Global seed for reproducibility.
        train_end: Exclusive end of the training period.
        val_end: Exclusive end of the validation (calibration) period.
        test_end: Exclusive end of the test period; later rows are dropped.
        llm_provider: SAR drafting backend; ``template`` is fully local.
        llm_api_key: Key for the external LLM provider.
        llm_model: Model id for the external SAR drafter (Anthropic).
        jev_enabled: Enable the Jev benchmark adapter (synthetic data only).
        jev_api_key: Key for the Jev hosted API.
        api_keys: API credentials as comma-separated ``user:role:sha256hex`` entries.
            Only hashes are configured; plaintext keys never touch config or disk.
        api_dataset: Dataset the API serves.
        api_cors_origins: Exact browser origins allowed by CORS (no wildcards).
        api_allowed_hosts: ``Host`` header allowlist (blocks DNS rebinding).
        api_rate_limit_per_minute: Requests per minute per user (and per client IP
            for failed authentication).
        api_max_request_bytes: Largest accepted request body.
        api_score_max_history_rows: ``POST /score`` recomputes past-only features over
            the loaded history; above this size it is refused rather than slow.
        dashboard_api_url: Base URL the dashboard calls.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: LogLevel = "INFO"
    log_json: bool = True

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    random_seed: int = Field(default=42, ge=0)

    train_end: datetime = datetime(2022, 9, 7)  # noqa: DTZ001 - dataset timestamps are naive
    val_end: datetime = datetime(2022, 9, 9)  # noqa: DTZ001
    test_end: datetime = datetime(2022, 9, 11)  # noqa: DTZ001

    llm_provider: Literal["template", "external"] = "template"
    llm_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"

    jev_enabled: bool = False
    jev_api_key: SecretStr | None = None

    api_keys: SecretStr | None = None
    api_dataset: DatasetName = DatasetName.SAMPLE
    api_cors_origins: tuple[str, ...] = ("http://localhost:8501", "http://127.0.0.1:8501")
    api_allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    api_rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
    api_max_request_bytes: int = Field(default=64 * 1024, ge=1024, le=10 * 1024 * 1024)
    api_score_max_history_rows: int = Field(default=100_000, ge=0)
    dashboard_api_url: str = Field(default="http://127.0.0.1:8000", pattern=r"^https?://[^\s]+$")

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        """Fail fast on missing secrets, impossible split boundaries and open CORS."""
        if not self.train_end < self.val_end < self.test_end:
            raise ValueError("split boundaries must satisfy TRAIN_END < VAL_END < TEST_END")
        for origin in self.api_cors_origins:
            if not _ORIGIN_RE.match(origin):
                raise ValueError("API_CORS_ORIGINS entries must be exact http(s) origins")
        if "*" in self.api_allowed_hosts:
            raise ValueError("API_ALLOWED_HOSTS must not contain a wildcard")
        if self.llm_provider == "external" and not _has_secret(self.llm_api_key):
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=external")
        if self.jev_enabled and not _has_secret(self.jev_api_key):
            raise ValueError("JEV_API_KEY is required when JEV_ENABLED=true")
        return self

    @property
    def raw_dir(self) -> Path:
        """Directory holding the untouched downloaded dataset files."""
        return self.data_dir / "raw"

    @property
    def sample_dir(self) -> Path:
        """Directory holding the bundled synthetic sample (committed to git)."""
        return self.data_dir / "sample"

    @property
    def processed_dir(self) -> Path:
        """Directory holding derived, regenerable data (parquet caches)."""
        return self.data_dir / "processed"

    @property
    def cases_db(self) -> Path:
        """SQLite file holding analyst actions and SAR versions (git-ignored)."""
        return self.data_dir / "cases" / "cases.sqlite3"

    @property
    def audit_path(self) -> Path:
        """Append-only, hash-chained audit log (git-ignored)."""
        return self.data_dir / "audit" / "audit.jsonl"


def _has_secret(value: SecretStr | None) -> bool:
    return value is not None and value.get_secret_value().strip() != ""


def load_settings(**overrides: Any) -> Settings:
    """Build :class:`Settings` from the environment, failing fast on errors.

    Args:
        **overrides: Explicit field values (mainly for tests); they take precedence
            over environment variables. Pass ``_env_file=None`` to ignore ``.env``.

    Returns:
        The validated settings.

    Raises:
        ConfigurationError: If any value is missing, malformed or inconsistent. The
            message names the offending fields but never includes input values.
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigurationError(f"invalid configuration: {problems}") from None
