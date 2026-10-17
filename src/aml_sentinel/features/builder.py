"""Assemble the System 1 feature matrix.

Every feature of a transaction at time ``t`` uses only information available at
``t``:

* row features: the transaction itself;
* trailing features: past-only windows (:mod:`aml_sentinel.features.temporal`);
* snapshot features: graph state at the start of the day
  (:mod:`aml_sentinel.features.graph_features`).

Labels never enter this module.

:data:`SHORTCUT_FEATURES` lists features that exploit simulator artifacts measured in
the EDA: ACH carries 87% of HI-Small positives, and cross-currency transfers are never
laundering. Models are trained with and without them (ADR-010).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Final

import numpy as np
import pandas as pd

from aml_sentinel.data.schema import PAYMENT_FORMATS, Col
from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.features.graph_features import NODE_FEATURES, snapshot_features
from aml_sentinel.features.temporal import trailing_aggregates

# Fixed order, so integer codes never depend on which formats a split happens to contain.
PAYMENT_FORMAT_ORDER: Final = tuple(sorted(PAYMENT_FORMATS))
PAYMENT_FORMAT: Final = "payment_format_code"
CROSS_CURRENCY: Final = "cross_currency"
SHORTCUT_FEATURES: Final = (PAYMENT_FORMAT, CROSS_CURRENCY)
CATEGORICAL_FEATURES: Final = (PAYMENT_FORMAT,)

_REQUIRED = (
    Col.TIMESTAMP,
    Col.FROM_ID,
    Col.TO_ID,
    Col.FROM_BANK,
    Col.TO_BANK,
    Col.AMOUNT_PAID,
    Col.AMOUNT_PAID_USD,
    Col.PAYMENT_CURRENCY,
    Col.RECEIVING_CURRENCY,
    Col.PAYMENT_FORMAT,
)


def row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features that depend only on the transaction itself.

    Args:
        frame: Processed transactions.

    Returns:
        Float32 feature DataFrame aligned with ``frame``.
    """
    ts = frame[Col.TIMESTAMP]
    paid = frame[Col.AMOUNT_PAID].to_numpy(dtype=np.float64)
    fmt = pd.Categorical(
        frame[Col.PAYMENT_FORMAT].astype("string"), categories=list(PAYMENT_FORMAT_ORDER)
    )
    return pd.DataFrame(
        {
            "amount_usd_log": np.log1p(frame[Col.AMOUNT_PAID_USD].to_numpy(dtype=np.float64)),
            # Round amounts are a classic manual-structuring tell.
            "round_amount": (np.mod(paid, 100.0) == 0) & (paid > 0),
            "hour": ts.dt.hour,
            "weekday": ts.dt.weekday,
            "same_bank": frame[Col.FROM_BANK].astype("string")
            == frame[Col.TO_BANK].astype("string"),
            "self_transfer": frame[Col.FROM_ID].astype("string")
            == frame[Col.TO_ID].astype("string"),
            CROSS_CURRENCY: frame[Col.PAYMENT_CURRENCY].astype("string")
            != frame[Col.RECEIVING_CURRENCY].astype("string"),
            PAYMENT_FORMAT: fmt.codes,
        },
        index=frame.index,
    ).astype(np.float32)


def trailing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Past-only velocity and volume features for sender and receiver.

    Args:
        frame: Processed transactions sorted by time.

    Returns:
        Float32 feature DataFrame aligned with ``frame``.
    """
    hour, day = timedelta(hours=1), timedelta(hours=24)
    usd = Col.AMOUNT_PAID_USD.value
    out_1h = trailing_aggregates(frame, key=Col.FROM_ID.value, window=hour)
    out_24h = trailing_aggregates(frame, key=Col.FROM_ID.value, window=day, values=usd)
    in_24h = trailing_aggregates(frame, key=Col.TO_ID.value, window=day, values=usd)
    return pd.DataFrame(
        {
            "src_tx_1h": out_1h.counts,
            "src_tx_24h": out_24h.counts,
            "src_usd_24h_log": np.log1p(out_24h.sums),
            "dst_rx_24h": in_24h.counts,
            "dst_usd_24h_log": np.log1p(in_24h.sums),
        },
        index=frame.index,
    ).astype(np.float32)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the full System 1 feature matrix.

    Args:
        frame: Processed transactions sorted by time (output of ``prepare-data``).

    Returns:
        Float32 DataFrame aligned with ``frame``; column order is stable.

    Raises:
        DataValidationError: If required columns are missing or rows are unsorted.
    """
    missing = sorted({c.value for c in _REQUIRED} - set(frame.columns))
    if missing:
        raise DataValidationError(f"feature builder missing columns: {missing}")
    if not frame[Col.TIMESTAMP].is_monotonic_increasing:
        raise DataValidationError("frame must be sorted by timestamp")
    return pd.concat(
        [row_features(frame), trailing_features(frame), snapshot_features(frame)], axis=1
    )


def feature_columns(*, include_shortcuts: bool) -> list[str]:
    """Return model input columns for a feature variant.

    Args:
        include_shortcuts: Keep :data:`SHORTCUT_FEATURES` when true.

    Returns:
        Column names; this order is the model's input order.
    """
    cols = [
        "amount_usd_log",
        "round_amount",
        "hour",
        "weekday",
        "same_bank",
        "self_transfer",
        CROSS_CURRENCY,
        PAYMENT_FORMAT,
        "src_tx_1h",
        "src_tx_24h",
        "src_usd_24h_log",
        "dst_rx_24h",
        "dst_usd_24h_log",
        *(f"{side}_{n}" for side in ("src", "dst") for n in NODE_FEATURES),
    ]
    return cols if include_shortcuts else [c for c in cols if c not in SHORTCUT_FEATURES]
