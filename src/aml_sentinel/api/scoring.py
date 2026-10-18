"""Score one new transaction against the loaded history (``POST /score``).

The System 1 features of a transaction depend on *all* earlier history (daily graph
snapshots, trailing windows). There is no online feature store, so the scorer
appends the transaction to the history rows at or before its timestamp, rebuilds the
past-only features with the same code used in training, and keeps the last row.
That costs time linear in the history: roughly 0.1 s on the 6.3k-row bundled sample,
but minutes on HI-Small's 5M rows. Above ``API_SCORE_MAX_HISTORY_ROWS`` the scorer is
therefore disabled, and the endpoint says so instead of hanging. Production would
maintain these aggregates incrementally in a streaming feature store.

Security:
    * Scored transactions are **not** added to the history. Otherwise any API
      client could change other accounts' features (data poisoning) by posting
      fabricated payments.
    * Only history at or before the transaction's timestamp is used, so a
      back-dated request cannot see the future.
    * The model and calibrator are loaded only after their SHA-256 checksums match
      the values recorded at training time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_manifest, processed_paths
from aml_sentinel.data.schema import Col, DatasetName, Transaction, account_node_id
from aml_sentinel.exceptions import DataNotFoundError, DataValidationError
from aml_sentinel.features.builder import build_features
from aml_sentinel.integrity import verify_checksum
from aml_sentinel.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibrator_from_dict,
)
from aml_sentinel.models.lgbm import LightGBMTriage
from aml_sentinel.models.rule_triage import VELOCITY_FEATURE
from aml_sentinel.models.system1a import load_baseline_config
from aml_sentinel.models.system1b import queue_size
from aml_sentinel.rules.engine import Rule, RuleConfig, apply_rules

logger = logging.getLogger(__name__)

VARIANT: Final = "all"
_HISTORY_COLUMNS: Final = (
    Col.TX_ID,
    Col.TIMESTAMP,
    Col.FROM_BANK,
    Col.FROM_ACCOUNT,
    Col.TO_BANK,
    Col.TO_ACCOUNT,
    Col.AMOUNT_RECEIVED,
    Col.RECEIVING_CURRENCY,
    Col.AMOUNT_PAID,
    Col.PAYMENT_CURRENCY,
    Col.PAYMENT_FORMAT,
    Col.FROM_ID,
    Col.TO_ID,
    Col.AMOUNT_PAID_USD,
    Col.AMOUNT_RECEIVED_USD,
)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Outcome of scoring one transaction.

    Attributes:
        p_tabular: Calibrated LightGBM probability (validation-fitted; see ADR-011
            for its drift on test).
        model: Model and calibrator name.
        alert: True if the score reaches the System 1a alert-queue threshold.
        alert_threshold: That threshold (validation-fitted).
        rules_fired: Enabled baseline rules that fire, or ``None`` if no baseline.
        history_rows: Past rows the features were computed from.
        latency_ms: Wall time for features and scoring.
    """

    p_tabular: float
    model: str
    alert: bool
    alert_threshold: float
    rules_fired: tuple[str, ...] | None
    history_rows: int
    latency_ms: float


class TransactionScorer:
    """Past-only System 1a scoring of new transactions.

    Args:
        history: Processed transactions, time-sorted.
        fx_rates: USD per unit of each non-USD currency (fitted on train).
        model: Verified LightGBM model.
        calibrator: Verified calibrator for ``model``.
        alert_threshold: Calibrated score at the bottom of the validation alert queue.
        rules: Validation-selected rule baseline, if available.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        fx_rates: dict[str, float],
        model: LightGBMTriage,
        calibrator: PlattCalibrator | IsotonicCalibrator,
        alert_threshold: float,
        rules: RuleConfig | None = None,
    ) -> None:
        if not history[Col.TIMESTAMP].is_monotonic_increasing:
            raise DataValidationError("scoring history must be time-sorted")
        hist = history[[c.value for c in _HISTORY_COLUMNS]].reset_index(drop=True)
        # The processed frame stores ids as categoricals. A new account is not one of
        # the categories and would silently become NaN, so ids become plain strings.
        for col in hist.columns:
            if isinstance(hist[col].dtype, pd.CategoricalDtype):
                hist[col] = hist[col].astype("string")
        self._history = hist
        self._ts = self._history[Col.TIMESTAMP].to_numpy()
        self._next_id = int(self._history[Col.TX_ID].max()) + 1 if len(history) else 0
        self._fx = {**fx_rates, "US Dollar": 1.0}
        self._model = model
        self._calibrator = calibrator
        self._alert_threshold = alert_threshold
        self._rules = rules

    @property
    def model_name(self) -> str:
        """``lightgbm[<variant>]+<calibration method>``."""
        return f"{self._model.name}+{self._calibrator.method}"

    def _row(self, tx: Transaction) -> pd.DataFrame:
        try:
            paid_usd = tx.amount_paid * self._fx[tx.payment_currency]
            received_usd = tx.amount_received * self._fx[tx.receiving_currency]
        except KeyError:
            raise DataValidationError("no FX rate for the transaction's currency") from None
        values = {
            Col.TX_ID.value: self._next_id,
            Col.TIMESTAMP.value: pd.Timestamp(tx.timestamp),
            Col.FROM_BANK.value: tx.from_bank,
            Col.FROM_ACCOUNT.value: tx.from_account,
            Col.TO_BANK.value: tx.to_bank,
            Col.TO_ACCOUNT.value: tx.to_account,
            Col.AMOUNT_RECEIVED.value: tx.amount_received,
            Col.RECEIVING_CURRENCY.value: tx.receiving_currency,
            Col.AMOUNT_PAID.value: tx.amount_paid,
            Col.PAYMENT_CURRENCY.value: tx.payment_currency,
            Col.PAYMENT_FORMAT.value: tx.payment_format,
            Col.FROM_ID.value: account_node_id(tx.from_bank, tx.from_account),
            Col.TO_ID.value: account_node_id(tx.to_bank, tx.to_account),
            Col.AMOUNT_PAID_USD.value: paid_usd,
            Col.AMOUNT_RECEIVED_USD.value: received_usd,
        }
        row = pd.DataFrame({k: [v] for k, v in values.items()})
        # Match history dtypes (categoricals, strings) so concat does not upcast.
        return row.astype(self._history.dtypes.to_dict())

    def score(self, tx: Transaction) -> ScoreResult:
        """Score ``tx`` from history at or before its timestamp.

        Args:
            tx: Validated, unlabeled transaction with a naive timestamp.

        Returns:
            The result. Nothing is persisted.

        Raises:
            DataValidationError: If the currency has no FX rate.
        """
        start = time.perf_counter()
        cut = int(np.searchsorted(self._ts, np.datetime64(tx.timestamp), side="right"))
        frame = pd.concat([self._history.iloc[:cut], self._row(tx)], ignore_index=True)
        features = build_features(frame).iloc[[-1]]
        p = float(self._calibrator.transform(self._model.predict(features))[0])
        fired: tuple[str, ...] | None = None
        if self._rules is not None:
            velocity = features[VELOCITY_FEATURE].to_numpy().astype(np.int64)
            flags = apply_rules(frame.iloc[[-1]], self._rules, velocity)
            fired = tuple(
                r.value for r in Rule if r in self._rules.enabled and bool(flags[r.value].iloc[0])
            )
        ms = (time.perf_counter() - start) * 1000
        logger.info("transaction scored", extra={"history_rows": cut, "ms": round(ms, 1)})
        return ScoreResult(
            p_tabular=p,
            model=self.model_name,
            alert=p >= self._alert_threshold,
            alert_threshold=self._alert_threshold,
            rules_fired=fired,
            history_rows=cut,
            latency_ms=ms,
        )


def load_calibrator(
    settings: Settings, dataset: DatasetName, variant: str = VARIANT
) -> PlattCalibrator | IsotonicCalibrator:
    """Load a calibrator after checking it against the System 1a report's checksum.

    Args:
        settings: Settings.
        dataset: Dataset variant.
        variant: Feature variant.

    Returns:
        The calibrator.

    Raises:
        DataNotFoundError: If the report or calibrator is missing.
        IntegrityError: If the calibrator does not match the recorded checksum.
    """
    report_path = settings.reports_dir / "system1a" / f"{dataset.value}.json"
    if not report_path.is_file():
        raise DataNotFoundError(f"{report_path.name} missing; run train-tabular first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = str(report["variants"][variant]["artifacts"]["calibrator_sha256"])
    path = settings.models_dir / dataset.value / f"calibrator_{variant}.json"
    verify_checksum(path, expected)
    return calibrator_from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_scorer(
    settings: Settings, dataset: DatasetName, frame: pd.DataFrame
) -> tuple[TransactionScorer | None, str | None]:
    """Build the scorer, or explain why scoring is unavailable.

    Args:
        settings: Settings (paths and the history-size cap).
        dataset: Dataset variant.
        frame: Processed transactions.

    Returns:
        ``(scorer, None)``, or ``(None, reason)`` when the history is too large or the
        System 1a artifacts are missing.
    """
    if len(frame) > settings.api_score_max_history_rows:
        return None, (
            f"scoring new transactions is disabled for histories above "
            f"{settings.api_score_max_history_rows:,} rows (this dataset has {len(frame):,}); "
            "production needs an online feature store"
        )
    try:
        model = LightGBMTriage.load(settings.models_dir / dataset.value / f"lightgbm_{VARIANT}.txt")
        calibrator = load_calibrator(settings, dataset)
        scores = pd.read_parquet(
            settings.processed_dir / f"{dataset.value}.system1a_scores.parquet",
            columns=["split", f"p_{VARIANT}_cal"],
        )
    except (DataNotFoundError, FileNotFoundError):
        return None, "System 1a artifacts are missing; run train-tabular first"
    val = scores.loc[scores["split"].astype("string") == "val", f"p_{VARIANT}_cal"]
    k = queue_size(len(val))
    threshold = float(np.sort(val.to_numpy(dtype=np.float64))[::-1][k - 1])
    manifest = load_manifest(processed_paths(settings.processed_dir, dataset)[1])
    try:
        _, rules = load_baseline_config(settings.reports_dir, dataset)
    except DataNotFoundError:
        rules = None
    scorer = TransactionScorer(
        frame, dict(manifest["fx_usd_rates"]), model, calibrator, threshold, rules
    )
    return scorer, None
