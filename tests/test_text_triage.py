"""Summarizer, typed-decision parsing, and the Laya / Jev adapters (with fakes, no network)."""

from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import SecretStr

from aml_sentinel.config import load_settings
from aml_sentinel.data.patterns import Typology
from aml_sentinel.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    DataValidationError,
    ExternalServiceError,
    IntegrityError,
    ModelOutputError,
    SyntheticDataRequiredError,
)
from aml_sentinel.models import jev_triage
from aml_sentinel.models.decisions import QUESTIONS, RISK_LEVELS, TYPOLOGY_LABELS, parse_answers
from aml_sentinel.models.jev_triage import JEV_ENDPOINT, JevTriage, https_transport
from aml_sentinel.models.laya_triage import (
    LayaTriage,
    download_base,
    load_agent,
    verified_digests,
    write_manifest,
)
from aml_sentinel.models.summarizer import REQUIRED_FEATURES, Noise, summarize, validate_summary
from aml_sentinel.models.triage import AlertBatch, Triage
from aml_sentinel.models.typology import TYPOLOGY_TO_LABEL, LayaLabel, laya_label

SECRET = "jev-test-key-DO-NOT-LEAK"


def _features(**over: float) -> dict[str, float]:
    base = dict.fromkeys(REQUIRED_FEATURES, 1.0)
    base.update(
        amount_usd_log=float(np.log1p(9_200.0)),
        hour=14,
        weekday=2,
        round_amount=0,
        self_transfer=0,
        same_bank=0,
        src_has_history=1,
        dst_has_history=1,
        src_flow_ratio=0.3,
        dst_flow_ratio=0.9,
        src_cycle3=2,
        dst_reciprocal=1,
    )
    base.update(over)
    return base


def _answer(escalate: float = 0.8, choice: str = "circular", risk_keys: str = "index") -> dict:
    probs = dict.fromkeys(TYPOLOGY_LABELS, 0.05)
    probs[choice] = 0.8
    levels = [0.1, 0.1, 0.1, 0.2, 0.5]
    keys = [str(i) for i in range(5)] if risk_keys == "index" else list(RISK_LEVELS)
    return {
        "typology": {"type": "choice", "choice": choice, "probabilities": probs},
        "risk": {"type": "score", "probabilities": dict(zip(keys, levels, strict=True))},
        "escalate": {"type": "noul", "noul": escalate},
    }


class FakeLaya:
    def __init__(self, truncated: bool = False, drop: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.truncated, self.drop = truncated, drop

    def predict_batch(
        self, states: list[str], questions: dict[str, Any], batch_size: int | None = None
    ) -> list[dict[str, Any]]:
        assert questions is QUESTIONS
        self.calls.append(states)
        n = len(states) - self.drop
        return [
            {"answers": _answer(0.1 + 0.8 * (i % 2)), "usage": {"truncated": self.truncated}}
            for i in range(n)
        ]


def _batch(n: int = 4, *, synthetic: bool = True) -> AlertBatch:
    summaries = tuple(summarize(_features(hour=float(i))) for i in range(n))
    return AlertBatch(
        features=pd.DataFrame({"x": range(n)}), summaries=summaries, is_synthetic=synthetic
    )


# ----------------------------------------------------------------- typology
def test_every_typology_maps_to_a_laya_label() -> None:
    assert set(TYPOLOGY_TO_LABEL) == set(Typology)
    assert laya_label(False, "CYCLE") is LayaLabel.BENIGN
    assert laya_label(True, "CYCLE") is LayaLabel.CIRCULAR
    assert laya_label(True, "FAN-OUT") is LayaLabel.MULE_NETWORK
    assert laya_label(True, "STRUCTURING") is LayaLabel.STRUCTURING
    assert laya_label(True, None) is None
    assert len(LayaLabel) < 10  # Laya degrades with many labels


# ----------------------------------------------------------------- summarizer
def test_summary_is_deterministic_and_minimal() -> None:
    text = summarize(_features())
    assert text == summarize(_features())
    assert "just below the 10k reporting threshold" in text
    assert "three-account payment cycles" in text
    assert ":" not in text.replace("14:00", "")  # no IDs of the form bank:account
    assert len(text.split()) < 200


def test_summary_noise_changes_text_but_stays_valid() -> None:
    rng = np.random.default_rng(0)
    noisy = {summarize(_features(), noise=Noise(drop_prob=0.4), rng=rng) for _ in range(10)}
    assert len(noisy) > 1
    for t in noisy:
        validate_summary(t)


def test_summary_input_errors() -> None:
    with pytest.raises(DataValidationError, match="missing"):
        summarize({"hour": 1.0})
    with pytest.raises(DataValidationError, match="finite"):
        summarize(_features(src_tx_1h=float("nan")))
    with pytest.raises(DataValidationError, match="rng"):
        summarize(_features(), noise=Noise())


@pytest.mark.parametrize("text", ["", "x" * 2000, "<script>alert(1)</script>", 'say "hi"'])
def test_validate_summary_rejects(text: str) -> None:
    with pytest.raises(DataValidationError):
        validate_summary(text)


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.floats(0, 30, allow_nan=False),
        min_size=len(REQUIRED_FEATURES),
        max_size=len(REQUIRED_FEATURES),
    )
)
def test_any_finite_features_give_a_valid_short_summary(values: list[float]) -> None:
    text = summarize(dict(zip(REQUIRED_FEATURES, values, strict=True)))
    validate_summary(text)


# ----------------------------------------------------------------- answer parsing
@pytest.mark.parametrize("keys", ["index", "text"])
def test_parse_answers(keys: str) -> None:
    parsed = parse_answers(_answer(0.7, "layering", keys))
    assert parsed.escalate_p == pytest.approx(0.7)
    assert parsed.typology == "layering"
    assert parsed.risk_level == 5
    assert parsed.risk_expected == pytest.approx(0.1 + 0.2 + 0.3 + 0.8 + 2.5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("escalate"),
        lambda a: a["typology"].update(choice="teleport"),
        lambda a: a["typology"]["probabilities"].update(benign=0.9),
        lambda a: a["escalate"].update(noul=1.5),
        lambda a: a["escalate"].update(noul="high"),
        lambda a: a["risk"]["probabilities"].update({"9": 0.1}),
        lambda a: a["risk"].update(probabilities={"0": 0.5}),
        lambda a: a["typology"].update(probabilities=None),
        lambda a: a["risk"].update(probabilities=[1]),
    ],
)
def test_parse_answers_rejects_malformed(mutate: Any) -> None:
    ans = _answer()
    mutate(ans)
    with pytest.raises(ModelOutputError):
        parse_answers(ans)
    with pytest.raises(ModelOutputError):
        parse_answers(["not", "a", "mapping"])


# ----------------------------------------------------------------- Laya adapter
def test_laya_triage_outputs() -> None:
    fake = FakeLaya()
    model = LayaTriage(fake, variant="test")
    assert isinstance(model, Triage)
    out = model.triage(_batch(4))
    assert model.name == "laya[test]"
    np.testing.assert_allclose(out.scores, [0.1, 0.9, 0.1, 0.9])
    assert out.escalate is not None
    assert out.escalate.tolist() == [False, True, False, True]
    assert out.typology == ("circular",) * 4
    assert out.extras is not None
    assert set(out.extras) >= {"p_benign", "risk_expected"}
    assert len(fake.calls[0]) == 4


@pytest.mark.parametrize(
    ("agent", "batch", "error"),
    [
        (FakeLaya(truncated=True), _batch(2), ModelOutputError),
        (FakeLaya(drop=1), _batch(2), ModelOutputError),
        (FakeLaya(), AlertBatch(features=pd.DataFrame({"x": [1]})), DataValidationError),
        (
            FakeLaya(),
            AlertBatch(features=pd.DataFrame({"x": [1]}), summaries=("<b>",)),
            DataValidationError,
        ),
    ],
)
def test_laya_triage_errors(agent: FakeLaya, batch: AlertBatch, error: type) -> None:
    with pytest.raises(error):
        LayaTriage(agent, variant="t").triage(batch)


def _checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt"
    (ckpt / "tokenizer").mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"weights")
    (ckpt / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    write_manifest(ckpt, source="test", revision="abc")
    return ckpt


def test_load_agent_verifies_and_passes_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = _checkpoint(tmp_path)
    seen: dict[str, Any] = {}

    def fake_load(path: str, device: str | None = None, expected_sha256: Any = None) -> FakeLaya:
        seen.update(path=path, digests=expected_sha256)
        return FakeLaya()

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=fake_load))
    assert isinstance(load_agent(ckpt), FakeLaya)
    assert set(seen["digests"]) == {
        "model.safetensors",
        "rl_agent_config.json",
        "tokenizer/tokenizer.json",
    }

    (ckpt / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        load_agent(ckpt)


def test_load_agent_never_lets_the_library_modify_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: laya.load rewrites tokenizer_config.json where it loads from."""
    ckpt = _checkpoint(tmp_path)
    (ckpt / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    write_manifest(ckpt, source="test", revision="abc")
    loaded_from: list[str] = []

    def mutating_load(
        path: str, device: str | None = None, expected_sha256: Any = None
    ) -> FakeLaya:
        loaded_from.append(path)
        (Path(path) / "tokenizer" / "tokenizer_config.json").write_text(
            '{"x": 1}', encoding="utf-8"
        )
        return FakeLaya()

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=mutating_load))
    load_agent(ckpt)
    load_agent(ckpt)  # a second load must still verify cleanly
    assert Path(loaded_from[0]) != ckpt
    assert not Path(loaded_from[0]).exists()  # temporary copy cleaned up
    verified_digests(ckpt)


def test_load_agent_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DataNotFoundError):
        verified_digests(tmp_path)
    (tmp_path / "manifest.json").write_text('{"sha256": {}}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"model.safetensors"):
        verified_digests(tmp_path)
    ckpt = _checkpoint(tmp_path / "x")
    monkeypatch.setitem(sys.modules, "laya", None)  # simulate "not installed"
    with pytest.raises(ConfigurationError, match="not installed"):
        load_agent(ckpt)


def test_download_base_pins_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def snapshot_download(repo: str, revision: str, local_dir: Path, allow_patterns: Any) -> str:
        calls.update(repo=repo, revision=revision)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "model.safetensors").write_bytes(b"w")
        return str(local_dir)

    fake_hub = types.SimpleNamespace(
        snapshot_download=snapshot_download,
        HfApi=lambda: types.SimpleNamespace(
            model_info=lambda repo: types.SimpleNamespace(sha="c0ffee")
        ),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    dest = download_base(tmp_path / "base")
    assert calls == {"repo": "convaiinnovations/laya", "revision": "c0ffee"}
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == "c0ffee"
    assert "model.safetensors" in manifest["sha256"]


# ----------------------------------------------------------------- Jev adapter
class FakeTransport:
    def __init__(self, response: bytes | None = None) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.response = response or json.dumps({"answers": _answer(0.6)}).encode()

    def __call__(self, url: str, headers: Any, body: bytes, timeout: float) -> bytes:
        self.requests.append((url, dict(headers), json.loads(body)))
        return self.response


def _jev(transport: FakeTransport) -> JevTriage:
    return JevTriage(SecretStr(SECRET), enabled=True, transport=transport)


def test_jev_refuses_non_synthetic_data_without_sending_anything() -> None:
    transport = FakeTransport()
    with pytest.raises(SyntheticDataRequiredError, match="synthetic"):
        _jev(transport).triage(_batch(2, synthetic=False))
    assert transport.requests == []
    # The safe default: a batch that does not say it is synthetic is refused.
    with pytest.raises(SyntheticDataRequiredError):
        _jev(transport).triage(AlertBatch(features=pd.DataFrame({"x": [1]}), summaries=("a",)))
    assert transport.requests == []


def test_jev_is_disabled_by_default_and_needs_a_key() -> None:
    with pytest.raises(ConfigurationError, match="disabled"):
        JevTriage.from_settings(load_settings(_env_file=None))
    with pytest.raises(ConfigurationError, match="JEV_API_KEY"):
        JevTriage(None, enabled=True)
    with pytest.raises(ConfigurationError, match="JEV_API_KEY"):
        JevTriage(SecretStr("  "), enabled=True)


def test_jev_sends_only_summaries_and_parses_answers() -> None:
    transport = FakeTransport()
    model = _jev(transport)
    batch = _batch(3)
    out = model.triage(batch)
    np.testing.assert_allclose(out.scores, [0.6] * 3)
    assert model.name == "jev[jev-latest]"
    url, headers, body = transport.requests[0]
    assert url == JEV_ENDPOINT
    assert url.startswith("https://")
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert set(body) == {"model", "state", "questions"}
    assert body["state"] == batch.summaries[0]  # nothing but the numeric summary
    assert SECRET not in repr(model.__dict__.get("_key"))


@pytest.mark.parametrize("response", [b"not json", b"[1, 2]", json.dumps({"answers": {}}).encode()])
def test_jev_malformed_responses(response: bytes) -> None:
    with pytest.raises(ModelOutputError):
        _jev(FakeTransport(response)).triage(_batch(1))


def test_jev_from_settings_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    model = JevTriage.from_settings(load_settings(_env_file=None), transport=FakeTransport())
    assert model.triage(_batch(1)).scores.shape == (1,)


def test_https_transport_rejects_plain_http() -> None:
    with pytest.raises(ConfigurationError, match="HTTPS"):
        https_transport("http://api.typesafe.ai/v1/systemone", {}, b"{}", 1.0)


class _Resp:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int) -> bytes:
        return self.data[:n]


def test_https_transport_errors_never_leak_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def http_error(req: Any, timeout: float) -> Any:
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(jev_triage.urllib.request, "urlopen", http_error)
    with pytest.raises(ExternalServiceError) as info:
        https_transport(JEV_ENDPOINT, {"Authorization": f"Bearer {SECRET}"}, b"{}", 1.0)
    assert "401" in str(info.value)
    assert SECRET not in str(info.value)

    def network_down(req: Any, timeout: float) -> Any:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(jev_triage.urllib.request, "urlopen", network_down)
    with pytest.raises(ExternalServiceError, match="failed"):
        https_transport(JEV_ENDPOINT, {}, b"{}", 1.0)

    big = b"x" * (jev_triage.MAX_RESPONSE_BYTES + 10)
    monkeypatch.setattr(jev_triage.urllib.request, "urlopen", lambda req, timeout: _Resp(big))
    with pytest.raises(ExternalServiceError, match="too large"):
        https_transport(JEV_ENDPOINT, {}, b"{}", 1.0)

    monkeypatch.setattr(jev_triage.urllib.request, "urlopen", lambda req, timeout: _Resp(b"ok"))
    assert https_transport(JEV_ENDPOINT, {}, b"{}", 1.0) == b"ok"
