"""Per-alert explanations: which features pushed LightGBM's score up or down.

The values are exact TreeSHAP contributions computed natively by LightGBM. They
explain the LightGBM log-odds, the main input to ``p_triage``, before calibration and
fusion, and say nothing about causation: a feature that raises the score is
associated with laundering in the (synthetic) training data, not proof of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.models.lgbm import LightGBMTriage

DEFAULT_TOP_K = 6


@dataclass(frozen=True, slots=True)
class Contribution:
    """One feature's push on the score.

    Attributes:
        feature: Model input column.
        value: The feature's value for this transaction.
        contribution: TreeSHAP contribution to the log-odds (positive raises risk).
    """

    feature: str
    value: float
    contribution: float


@dataclass(frozen=True, slots=True)
class Explanation:
    """Why one transaction got its LightGBM score.

    Attributes:
        model: Model name, e.g. ``lightgbm[all]``.
        base_log_odds: Expected log-odds over training data (the starting point).
        log_odds: This transaction's log-odds; equals base plus all contributions.
        top: The largest contributions by absolute size, largest first.
        others: Sum of every contribution not in ``top``.
    """

    model: str
    base_log_odds: float
    log_odds: float
    top: tuple[Contribution, ...]
    others: float


def explanation_from_row(
    model_name: str,
    features: list[str],
    values: npt.NDArray[np.float64],
    contrib: npt.NDArray[np.float64],
    top_k: int = DEFAULT_TOP_K,
) -> Explanation:
    """Build an :class:`Explanation` from one row of values and contributions.

    Args:
        model_name: Model name.
        features: Feature names, in model input order.
        values: The row's feature values, aligned with ``features``.
        contrib: The row's contributions, aligned with ``features`` plus a final bias.
        top_k: How many contributions to keep.

    Returns:
        The explanation.

    Raises:
        ValueError: If the arrays do not match ``features`` or ``top_k`` is not positive.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(values) != len(features) or len(contrib) != len(features) + 1:
        raise ValueError("values and contributions must align with the features")
    per_feature = contrib[:-1]
    order = np.argsort(-np.abs(per_feature), kind="stable")[:top_k]
    top = tuple(Contribution(features[i], float(values[i]), float(per_feature[i])) for i in order)
    rest = float(per_feature.sum() - per_feature[order].sum())
    return Explanation(
        model=model_name,
        base_log_odds=float(contrib[-1]),
        log_odds=float(contrib.sum()),
        top=top,
        others=rest,
    )


def explain(
    model: LightGBMTriage, features: pd.DataFrame, top_k: int = DEFAULT_TOP_K
) -> list[Explanation]:
    """Explain every row of ``features``.

    Args:
        model: Trained LightGBM model.
        features: Feature frame containing the model's input columns.
        top_k: Contributions to keep per row.

    Returns:
        One explanation per row, in row order.
    """
    contrib = model.contributions(features)
    values = features[model.features].to_numpy(dtype=np.float64)
    return [
        explanation_from_row(model.name, model.features, values[i], contrib[i], top_k)
        for i in range(len(features))
    ]
