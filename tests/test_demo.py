from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from aml_sentinel import demo
from aml_sentinel.api.auth import ApiKeyStore, Principal, Role, generate_api_key, hash_api_key
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import ConfigurationError

ANY_HOST = "0.0.0.0"  # noqa: S104 - the container bind address the demo maps to localhost


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return load_settings(_env_file=None, data_dir=tmp_path, **overrides)


def test_mints_working_keys_for_each_role(tmp_path: Path) -> None:
    creds = demo.demo_credentials(_settings(tmp_path, api_dataset=DatasetName.HI_SMALL))
    assert creds.settings.api_dataset is DatasetName.SAMPLE
    assert creds.settings.api_keys is not None
    store = ApiKeyStore.parse(creds.settings.api_keys.get_secret_value())
    assert {(user, role) for user, role, _ in creds.minted} == set(demo.DEMO_USERS)
    for user, role, key in creds.minted:
        assert store.authenticate(key) == Principal(user, role)
        assert key not in creds.settings.api_keys.get_secret_value()  # only digests


def test_keys_differ_between_runs(tmp_path: Path) -> None:
    first = demo.demo_credentials(_settings(tmp_path)).minted
    second = demo.demo_credentials(_settings(tmp_path)).minted
    assert {k for *_, k in first}.isdisjoint({k for *_, k in second})


def test_reuses_configured_keys(tmp_path: Path) -> None:
    entry = f"alice:analyst:{hash_api_key(generate_api_key())}"
    creds = demo.demo_credentials(_settings(tmp_path, api_keys=entry))
    assert creds.minted == ()
    assert creds.settings.api_keys is not None
    assert creds.settings.api_keys.get_secret_value() == entry


def test_refuses_production(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="prod"):
        demo.demo_credentials(_settings(tmp_path, environment="prod"))


def test_build_models_runs_steps_in_order(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[str] = []
    caplog.set_level(logging.INFO)
    demo.build_models([(name, lambda name=name: calls.append(name)) for name in ("a", "b", "c")])
    assert calls == ["a", "b", "c"]


def test_banner_lists_minted_keys_and_review_note() -> None:
    text = demo.credentials_banner([("demo-admin", Role.ADMIN, "k3y")], "http://localhost:8501")
    assert "k3y" in text
    assert "http://localhost:8501" in text
    assert "Synthetic data only" in text
    assert "already configured" in demo.credentials_banner([], "http://x")


def test_dashboard_argv_uses_current_interpreter() -> None:
    argv = demo.dashboard_argv("127.0.0.1", 8501)
    assert argv[:3] == [sys.executable, "-m", "streamlit"]
    assert argv[4] == str(demo.DASHBOARD_APP)
    assert demo.DASHBOARD_APP.is_file()
    assert argv[argv.index("--server.address") + 1] == "127.0.0.1"


def test_start_dashboard_points_at_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(argv: list[str], env: dict[str, str]) -> str:
        seen.update(argv=argv, env=env)
        return "proc"

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    assert demo.start_dashboard("127.0.0.1", 9000, "http://127.0.0.1:8001", {"A": "1"}) == "proc"
    assert seen["env"] == {"A": "1", "DASHBOARD_API_URL": "http://127.0.0.1:8001"}
    assert "9000" in seen["argv"]


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> int:
        return 0


def test_run_demo_serves_and_always_stops_dashboard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    creds = demo.demo_credentials(_settings(tmp_path))
    proc = _FakeProc()
    launched: list[tuple[str, int, str]] = []
    served: list[tuple[Settings, str, int]] = []

    def launch(host: str, port: int, api_url: str) -> Any:
        launched.append((host, port, api_url))
        return proc

    def serve(settings: Settings, host: str, port: int) -> None:
        served.append((settings, host, port))
        raise KeyboardInterrupt  # Ctrl+C

    with pytest.raises(KeyboardInterrupt):
        demo.run_demo(
            creds, host=ANY_HOST, api_port=8001, dashboard_port=8502, serve=serve, launch=launch
        )
    assert launched == [(ANY_HOST, 8502, "http://127.0.0.1:8001")]
    assert served == [(creds.settings, ANY_HOST, 8001)]
    assert proc.terminated
    out = capsys.readouterr().out
    assert "http://localhost:8502" in out
    for *_, key in creds.minted:
        assert key in out  # shown once on the console ...
        assert key not in caplog.text  # ... and never logged
