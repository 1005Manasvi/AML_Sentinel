from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml_sentinel.evaluation.metrics import UNDOCUMENTED, binary_metrics, recall_by_group
from aml_sentinel.exceptions import DataValidationError


def test_confusion_counts_and_ratios() -> None:
    m = binary_metrics([1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1])
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 2)
    assert (m.n, m.positives, m.alerts) == (6, 3, 3)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.false_positive_rate == pytest.approx(1 / 3)
    assert m.alert_rate == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 / 3)
    assert set(m.to_dict()) >= {"tp", "precision", "recall", "false_positive_rate", "f1"}


def test_undefined_ratios_are_none() -> None:
    no_alerts = binary_metrics([1, 0], [0, 0])
    assert no_alerts.precision is None
    assert no_alerts.f1 is None
    assert binary_metrics([0, 0], [1, 0]).recall is None
    assert binary_metrics([1, 1], [1, 1]).false_positive_rate is None
    assert binary_metrics([], []).alert_rate is None
    # precision and recall both 0 -> F1 undefined, not a division error
    assert binary_metrics([1, 0], [0, 1]).f1 is None


def test_bool_inputs_accepted() -> None:
    assert binary_metrics(np.array([True, False]), np.array([True, True])).fp == 1


@pytest.mark.parametrize(
    ("y_true", "y_pred", "msg"),
    [([0, 2], [0, 1], "binary"), ([0, 1], [0, 1, 1], "lengths")],
)
def test_invalid_inputs(y_true: list[int], y_pred: list[int], msg: str) -> None:
    with pytest.raises(DataValidationError, match=msg):
        binary_metrics(y_true, y_pred)


def test_recall_by_group_counts_undocumented() -> None:
    groups = pd.Series(["CYCLE", "CYCLE", None, None, "FAN-IN"], dtype="string")
    out = recall_by_group([1, 1, 1, 0, 0], [1, 0, 1, 1, 1], groups)
    assert out == {
        "CYCLE": {"positives": 2, "caught": 1, "recall": 0.5},
        UNDOCUMENTED: {"positives": 1, "caught": 1, "recall": 1.0},
    }
