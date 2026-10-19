"""One-command demo: bundled synthetic sample, freshly trained small models, API + dashboard.

``aml-sentinel demo`` (``make demo``) regenerates the seed-fixed synthetic sample, fits
the rule baseline and the LightGBM + calibration stack on it (well under a minute on a
laptop CPU), then starts the authenticated API and the Streamlit dashboard. No Kaggle
data, GPU, Laya checkpoint or external API is needed. Models are trained at start
rather than committed so that no generated binary lives in git and the demo can never
drift from the code (ADR-024).

Security:
    The demo refuses to run with ``ENVIRONMENT=prod``. If ``API_KEYS`` is configured
    the demo uses it unchanged. Otherwise it mints one analyst and one admin key for
    this process only: the keys are written once to the console (never to the log),
    only their SHA-256 digests reach the server, and nothing is saved to disk. Both
    servers bind to loopback by default.
"""

from __future__ import annotations

import logging
import os

# Fixed argv and no shell; see start_dashboard.
import subprocess  # nosec B404
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import SecretStr

from aml_sentinel.api.auth import ApiKeyStore, Role, generate_api_key, hash_api_key
from aml_sentinel.config import Settings
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

DASHBOARD_APP: Final = Path(__file__).resolve().parent / "dashboard" / "app.py"
DEMO_USERS: Final = (("demo-analyst", Role.ANALYST), ("demo-admin", Role.ADMIN))

BuildStep = tuple[str, Callable[[], object]]


@dataclass(frozen=True)
class DemoCredentials:
    """Settings the demo serves with, plus any keys minted for this run.

    Attributes:
        settings: Settings with ``API_DATASET=Sample`` and usable ``API_KEYS``.
        minted: ``(user, role, key)`` for each key created by the demo; empty when
            ``API_KEYS`` was already configured.
    """

    settings: Settings
    minted: tuple[tuple[str, Role, str], ...]


def demo_credentials(settings: Settings) -> DemoCredentials:
    """Pin the demo to the Sample dataset and make sure API keys exist.

    Args:
        settings: Validated settings.

    Returns:
        The demo settings and any minted keys.

    Raises:
        ConfigurationError: If ``ENVIRONMENT=prod``; the demo mints keys and must
            never be the way a production deployment starts.
    """
    if settings.environment == "prod":
        raise ConfigurationError("the demo must not run with ENVIRONMENT=prod")
    update: dict[str, object] = {"api_dataset": DatasetName.SAMPLE}
    minted: tuple[tuple[str, Role, str], ...] = ()
    configured = settings.api_keys
    if configured is None or not configured.get_secret_value().strip():
        minted = tuple((user, role, generate_api_key()) for user, role in DEMO_USERS)
        entries = ",".join(f"{user}:{role.value}:{hash_api_key(key)}" for user, role, key in minted)
        ApiKeyStore.parse(entries)  # same validation the server applies at start-up
        update["api_keys"] = SecretStr(entries)
    return DemoCredentials(settings=settings.model_copy(update=update), minted=minted)


def build_models(steps: Sequence[BuildStep]) -> None:
    """Run the demo build steps in order (sample, parquet, baseline, LightGBM).

    Args:
        steps: ``(name, callable)`` pairs; injected by the CLI so this module does not
            depend on it and tests can substitute cheap fakes.
    """
    for name, step in steps:
        logger.info("demo build step", extra={"step": name})
        step()


def credentials_banner(minted: Sequence[tuple[str, Role, str]], dashboard_url: str) -> str:
    """Console text telling the user how to sign in.

    Args:
        minted: Keys minted for this run (may be empty).
        dashboard_url: URL of the dashboard.

    Returns:
        Text for stdout. It contains the minted keys, so it must never go to the log.
    """
    lines = ["", f"AML Sentinel demo: open {dashboard_url} and sign in with a key below."]
    if minted:
        lines.append("Keys are valid for this process only and are not saved anywhere:")
        lines.extend(f"  {role.value:<8} {user:<13} {key}" for user, role, key in minted)
    else:
        lines.append("Using the API_KEYS already configured; sign in with one of your keys.")
    lines.append("Synthetic data only. Laya is not part of the demo (see docs/RESULTS.md).")
    return "\n".join(lines) + "\n\n"


def dashboard_argv(host: str, port: int) -> list[str]:
    """Command line that starts the Streamlit dashboard with the current interpreter.

    ``python -m streamlit`` is used instead of the ``streamlit`` launcher so the same
    interpreter (and venv) runs both servers, and so Windows application control
    policies that block unsigned console launchers do not stop the demo.

    Args:
        host: Bind address.
        port: Port.

    Returns:
        The argv list.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_APP),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]


def start_dashboard(
    host: str, port: int, api_url: str, environ: Mapping[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start the dashboard as a child process pointed at the demo API.

    Args:
        host: Dashboard bind address.
        port: Dashboard port.
        api_url: API base URL the dashboard should call.
        environ: Base environment (defaults to ``os.environ``).

    Returns:
        The running process; the caller terminates it.
    """
    env = dict(os.environ if environ is None else environ)
    env["DASHBOARD_API_URL"] = api_url
    # Fixed argv built from validated ints/strings and no shell, so nothing from the
    # environment or the user can inject a command.
    return subprocess.Popen(dashboard_argv(host, port), env=env)  # noqa: S603  # nosec B603


def run_demo(
    credentials: DemoCredentials,
    *,
    host: str,
    api_port: int,
    dashboard_port: int,
    serve: Callable[[Settings, str, int], None],
    launch: Callable[[str, int, str], subprocess.Popen[bytes]] = start_dashboard,
) -> None:
    """Start the dashboard, then serve the API in the foreground until interrupted.

    Args:
        credentials: Output of :func:`demo_credentials`.
        host: Bind address for both servers (loopback unless inside a container).
        api_port: API port.
        dashboard_port: Dashboard port.
        serve: Blocking API server (injected; the CLI passes uvicorn).
        launch: Dashboard launcher (injected for tests).
    """
    # The dashboard always reaches the API over loopback: both run on this machine.
    api_url = f"http://127.0.0.1:{api_port}"
    browse_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host  # noqa: S104  # nosec B104
    dashboard = launch(host, dashboard_port, api_url)
    sys.stdout.write(
        credentials_banner(credentials.minted, f"http://{browse_host}:{dashboard_port}")
    )
    sys.stdout.flush()
    try:
        serve(credentials.settings, host, api_port)
    finally:
        dashboard.terminate()
        try:
            dashboard.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - slow shutdown only
            dashboard.kill()
