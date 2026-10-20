from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml_sentinel.cli import main
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.evaluation.drift import (
    COMPARISONS,
    build_drift_report,
    label,
    psi,
    render_markdown,
)
from aml_sentinel.features.builder import feature_columns
from tests.conftest import ServedSample

floats = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=2, max_size=200
)


@given(floats, floats)
@settings(max_examples=150, deadline=None)
def test_psi_is_never_negative(a: list[float], b: list[float]) -> None:
    assert psi(a, b) >= 0.0


@given(floats)
@settings(max_examples=150, deadline=None)
def test_psi_of_identical_samples_is_zero(a: list[float]) -> None:
    assert psi(a, a) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_the_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    shifts = [psi(ref, rng.normal(mu, 1, 20_000)) for mu in (0.0, 0.25, 0.5, 1.0)]
    assert shifts == sorted(shifts)
    assert shifts[0] < 0.01
    assert label(shifts[0]) == "stable"
    assert label(shifts[-1]) == "major"


def test_psi_handles_binary_features_and_nans() -> None:
    before = np.array([0] * 90 + [1] * 10, dtype=float)
    after = np.array([0] * 50 + [1] * 50 + [np.nan] * 5, dtype=float)
    # Two bins (0 and 1): (0.5-0.9)ln(0.5/0.9) + (0.5-0.1)ln(0.5/0.1)
    expected = (0.5 - 0.9) * np.log(0.5 / 0.9) + (0.5 - 0.1) * np.log(0.5 / 0.1)
    assert psi(before, after) == pytest.approx(expected)


def test_psi_input_errors() -> None:
    with pytest.raises(ValueError, match="bins"):
        psi([1.0, 2.0], [1.0, 2.0], bins=1)
    with pytest.raises(ValueError, match="values"):
        psi([np.nan], [1.0])
    assert label(0.1) == "moderate"
    assert label(0.25) == "major"


def test_drift_report_on_sample(served_sample: ServedSample) -> None:
    scores = pd.read_parquet(
        served_sample.settings.processed_dir / "Sample.system1a_scores.parquet",
        columns=["tx_id", "split", "p_all_raw"],
    )
    report = build_drift_report(
        served_sample.frame, served_sample.settings, DatasetName.SAMPLE, scores
    )
    assert set(report["features"]) == {f"{a}->{b}" for a, b in COMPARISONS}
    cols = feature_columns(include_shortcuts=True)
    for values in report["features"].values():
        assert set(values) == set(cols)
        assert list(values.values()) == sorted(values.values(), reverse=True)
    assert report["score_psi_val_to_test"] >= 0
    assert sum(report["rows"].values()) <= len(served_sample.frame)
    md = render_markdown(report)
    assert "Drift report: Sample" in md
    assert "validation → test" in md
    json.dumps(report)  # serialisable


def test_drift_command_writes_reports(
    served_sample: ServedSample, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = served_sample.settings
    monkeypatch.setenv("DATA_DIR", str(s.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("MODELS_DIR", str(s.models_dir))
    monkeypatch.chdir(tmp_path)  # no stray .env
    assert main(["drift", "--dataset", "Sample"]) == 0
    out = tmp_path / "drift"
    assert (out / "Sample.json").is_file()
    assert (out / "Sample.md").is_file()
    assert (out / "Sample_psi.png").stat().st_size > 0
