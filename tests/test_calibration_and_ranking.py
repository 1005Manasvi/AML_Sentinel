from __future__ import annotations

import numpy as np
import pytest

from aml_sentinel.evaluation.ranking import (
    compare_at_matched_recall,
    compare_at_matched_volume,
    operating_points,
    pr_curve_points,
    precision_at_k,
    ranking_summary,
    top_k_mask,
)
from aml_sentinel.exceptions import ConfigurationError, DataValidationError
from aml_sentinel.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    calibrator_from_dict,
    expected_calibration_error,
    log_loss,
    reliability_table,
    select_method,
)


def _miscalibrated(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True probability p, but the model reports sqrt(p): over-confident scores."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 20, n)
    y = (rng.random(n) < p).astype(int)
    return np.sqrt(p), y


# ----------------------------------------------------------------- calibration
@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator])
def test_calibrators_reduce_ece(cls: type) -> None:
    s, y = _miscalibrated()
    before = expected_calibration_error(s, y)
    cal = cls.fit(s[:10_000], y[:10_000])
    after = expected_calibration_error(cal.transform(s[10_000:]), y[10_000:])
    assert after < before / 3
    assert calibrator_from_dict(cal.to_dict()) == cal


def test_isotonic_is_monotone_and_bounded() -> None:
    s, y = _miscalibrated()
    cal = IsotonicCalibrator.fit(s, y)
    out = cal.transform(np.linspace(-1, 2, 500))
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 0
    assert out.max() <= 1


def test_select_method_returns_heldout_scores() -> None:
    s, y = _miscalibrated()
    best, scores = select_method(s, y)
    assert best in {"platt", "isotonic"}
    assert set(scores) == {"platt", "isotonic"}
    assert scores[best] == min(scores.values())


@pytest.mark.parametrize(
    "data",
    [
        {"method": "temperature"},
        {"method": "platt"},
        {"method": "isotonic", "x": [0.5, 0.1], "y": [0, 1]},
        {"method": "isotonic", "x": [0.1], "y": [0.0, 1.0]},
    ],
)
def test_bad_calibrator_dicts(data: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        calibrator_from_dict(data)


@pytest.mark.parametrize(
    ("scores", "y"),
    [([], []), ([0.1, 0.2], [0, 1, 1]), ([0.1, 0.2], [0, 2]), ([0.1, 0.2], [1, 1])],
)
def test_fit_input_validation(scores: list[float], y: list[int]) -> None:
    with pytest.raises(DataValidationError):
        PlattCalibrator.fit(scores, y)


def test_calibration_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(perfect, y, n_bins=2) == pytest.approx(0)
    assert expected_calibration_error(perfect, y, n_bins=2, strategy="uniform") == pytest.approx(0)
    assert expected_calibration_error([], []) == 0.0
    assert brier_score(perfect, y) == 0
    assert log_loss([0.5] * 4, y) == pytest.approx(np.log(2))
    table = reliability_table([0.1, 0.1, 0.9, 0.9], y, n_bins=2)
    assert [r["frac_pos"] for r in table] == [0.0, 1.0]


# ----------------------------------------------------------------- ranking
Y = np.array([1, 0, 1, 0, 0, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])


def test_top_k_and_ties() -> None:
    assert top_k_mask(S, 3).tolist() == [True, True, True] + [False] * 5
    assert top_k_mask([0.5, 0.5, 0.5], 2).tolist() == [True, True, False]  # earliest first
    assert top_k_mask(S, -1).sum() == 0
    assert top_k_mask(S, 99).sum() == len(S)


def test_matched_volume() -> None:
    baseline = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=bool)  # 4 alerts, 1 TP, 3 FP
    out = compare_at_matched_volume(Y, S, baseline)
    assert out["model"]["alerts"] == 4
    assert out["model"]["tp"] == 2
    assert out["baseline"]["fp"] == 3
    assert out["fp_reduction"] == pytest.approx(1 - 2 / 3)


def test_matched_recall() -> None:
    baseline = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)  # recall 2/3, 4 FP
    out = compare_at_matched_recall(Y, S, baseline)
    assert out["reachable"] is True
    assert out["model"]["alerts"] == 3  # top-3 already holds 2 positives
    assert out["fp_reduction"] == pytest.approx(1 - 1 / 4)
    no_tp = compare_at_matched_recall(Y, S, np.zeros(8, dtype=bool))
    assert no_tp["reachable"] is False
    assert no_tp["fp_reduction"] is None


def test_length_mismatch() -> None:
    with pytest.raises(DataValidationError):
        compare_at_matched_volume(Y, S, np.zeros(3, dtype=bool))
    with pytest.raises(DataValidationError):
        compare_at_matched_recall(Y, S, np.zeros(3, dtype=bool))


def test_summaries() -> None:
    summary = ranking_summary(Y, S)
    assert summary["roc_auc"] is not None
    assert summary["base_rate"] == pytest.approx(3 / 8)
    assert ranking_summary([0, 0], [0.1, 0.2])["average_precision"] is None
    assert precision_at_k(Y, S, [1])["1"]["precision"] == 1.0
    assert operating_points(Y, S, [0.25])["0.25"]["alerts"] == 2
    curve = pr_curve_points(Y, S, max_points=3)
    assert len(curve["precision"]) == len(curve["recall"]) <= 3


def test_bootstrap_ap_difference() -> None:
    from aml_sentinel.evaluation.ranking import bootstrap_ap_difference

    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.2).astype(int)
    good = y + rng.normal(0, 0.5, 600)
    noise = rng.normal(0, 1, 600)
    better = bootstrap_ap_difference(y, good, noise, n_boot=200)
    assert better["diff"] > 0
    assert better["ci_low"] > 0
    assert better["p_not_better"] == 0.0
    same = bootstrap_ap_difference(y, good, good, n_boot=50)
    assert same["diff"] == 0
    assert same["p_not_better"] == 1.0
    with pytest.raises(DataValidationError):
        bootstrap_ap_difference([0, 0], [0.1, 0.2], [0.3, 0.4])
