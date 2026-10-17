from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from aml_sentinel.config import load_settings
from aml_sentinel.exceptions import ConfigurationError

SECRET = "sk-test-DO-NOT-LEAK-123"


def test_defaults_need_no_secrets() -> None:
    s = load_settings(_env_file=None)
    assert s.llm_provider == "template"
    assert s.jev_enabled is False
    assert s.raw_dir == Path("data") / "raw"
    assert s.processed_dir == Path("data") / "processed"


def test_jev_enabled_without_key_fails_fast() -> None:
    with pytest.raises(ConfigurationError, match="JEV_API_KEY"):
        load_settings(_env_file=None, jev_enabled=True)


def test_blank_key_counts_as_missing() -> None:
    with pytest.raises(ConfigurationError, match="JEV_API_KEY"):
        load_settings(_env_file=None, jev_enabled=True, jev_api_key="   ")


def test_external_llm_without_key_fails_fast() -> None:
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        load_settings(_env_file=None, llm_provider="external")


def test_secret_never_in_repr_or_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    monkeypatch.setenv("LLM_PROVIDER", "external")
    s = load_settings(_env_file=None)
    assert SECRET not in repr(s)
    assert s.llm_api_key is not None
    assert s.llm_api_key.get_secret_value() == SECRET

    # A validation error elsewhere must not echo the secret held in the input.
    monkeypatch.setenv("RANDOM_SEED", "-1")
    with pytest.raises(ConfigurationError) as info:
        load_settings(_env_file=None)
    assert SECRET not in str(info.value)
    assert "random_seed" in str(info.value)


def test_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    assert load_settings(_env_file=None).jev_enabled is True


def test_split_bounds_must_increase() -> None:
    with pytest.raises(ConfigurationError, match="TRAIN_END"):
        load_settings(_env_file=None, train_end=datetime(2022, 9, 10))


def test_settings_are_immutable() -> None:
    s = load_settings(_env_file=None)
    with pytest.raises(ValueError, match="frozen"):
        s.random_seed = 7  # type: ignore[misc]
