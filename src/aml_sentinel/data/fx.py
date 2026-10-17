"""Estimate exchange rates from the data and normalise amounts to USD.

The dataset mixes 15 currencies (including Bitcoin) and ships no FX table. Summing
raw amounts across currencies would be meaningless, so every currency's USD rate is
estimated as the median implied rate of its direct cross-currency transactions with
USD (``amount_received / amount_paid``). The median keeps a few odd rows from
skewing the estimate.

Rates use no labels. They must still be estimated on the training window only
(see :mod:`aml_sentinel.data.splits`) so no statistic leaks backwards from the
test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml_sentinel.data.schema import USD, Col
from aml_sentinel.exceptions import DataValidationError, FxEstimationError

DEFAULT_MIN_SAMPLES = 5


def estimate_usd_rates(
    frame: pd.DataFrame, *, min_samples: int = DEFAULT_MIN_SAMPLES
) -> dict[str, float]:
    """Estimate ``USD per 1 unit`` of every currency appearing in ``frame``.

    Args:
        frame: Transactions with canonical columns.
        min_samples: Minimum direct USD cross-currency rows required per currency.

    Returns:
        Mapping from currency name to USD rate; ``"US Dollar"`` maps to ``1.0``.

    Raises:
        FxEstimationError: If a currency has fewer than ``min_samples`` direct
            USD transactions or a non-positive estimate.
    """
    paid_cur = frame[Col.PAYMENT_CURRENCY].astype("string")
    recv_cur = frame[Col.RECEIVING_CURRENCY].astype("string")
    paid = frame[Col.AMOUNT_PAID].to_numpy(dtype=np.float64)
    recv = frame[Col.AMOUNT_RECEIVED].to_numpy(dtype=np.float64)
    usable = (paid > 0) & (recv > 0)

    to_usd = usable & (recv_cur == USD).to_numpy() & (paid_cur != USD).to_numpy()
    from_usd = usable & (paid_cur == USD).to_numpy() & (recv_cur != USD).to_numpy()
    samples = pd.concat(
        [
            pd.Series(recv[to_usd] / paid[to_usd], index=paid_cur[to_usd].to_numpy()),
            pd.Series(paid[from_usd] / recv[from_usd], index=recv_cur[from_usd].to_numpy()),
        ]
    )
    grouped = samples.groupby(level=0)
    medians, counts = grouped.median(), grouped.size()

    rates: dict[str, float] = {USD: 1.0}
    problems: list[str] = []
    seen = set(paid_cur.unique()) | set(recv_cur.unique())
    for currency in sorted(str(c) for c in seen if c != USD):
        n = int(counts.get(currency, 0))
        rate = float(medians.get(currency, np.nan))
        if n < min_samples or not np.isfinite(rate) or rate <= 0:
            problems.append(f"{currency} (n={n})")
        else:
            rates[currency] = rate
    if problems:
        raise FxEstimationError(f"cannot estimate USD rate for: {', '.join(problems)}")
    return rates


def add_usd_amounts(frame: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``amount_paid_usd`` and ``amount_received_usd``.

    Args:
        frame: Transactions with canonical columns.
        rates: Output of :func:`estimate_usd_rates`.

    Returns:
        A new DataFrame with the two USD columns added.

    Raises:
        DataValidationError: If a currency in ``frame`` has no rate.
    """
    out = frame.copy()
    for amount_col, cur_col, usd_col in (
        (Col.AMOUNT_PAID, Col.PAYMENT_CURRENCY, Col.AMOUNT_PAID_USD),
        (Col.AMOUNT_RECEIVED, Col.RECEIVING_CURRENCY, Col.AMOUNT_RECEIVED_USD),
    ):
        factor = out[cur_col].astype("string").map(rates).astype("Float64")
        n_missing = int(factor.isna().sum())
        if n_missing:
            raise DataValidationError(f"{cur_col}: {n_missing} rows with no FX rate")
        out[usd_col] = (out[amount_col] * factor.to_numpy(dtype=np.float64)).astype(np.float64)
    return out
