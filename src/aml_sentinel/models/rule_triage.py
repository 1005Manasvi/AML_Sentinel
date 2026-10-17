"""Adapter exposing the rule engine through the ``Triage`` protocol.

The score is the fraction of enabled rules that fired. It is a coarse ranking with at
most four distinct values, so ablations can treat "rules only" exactly like any other
model.
"""

from __future__ import annotations

import numpy as np

from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.models.triage import AlertBatch, TriageOutput
from aml_sentinel.rules.engine import N_RULES, RuleConfig, apply_rules

VELOCITY_FEATURE = "src_tx_24h"


class RuleTriage:
    """Rule-engine baseline as a ``Triage`` model.

    Args:
        config: Rule configuration (e.g. the validation-selected baseline).
    """

    def __init__(self, config: RuleConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        """Always ``"rules"``."""
        return "rules"

    def triage(self, batch: AlertBatch) -> TriageOutput:
        """Score a batch by the fraction of enabled rules fired.

        Args:
            batch: Needs ``transactions`` and the ``src_tx_24h`` feature, which is the
                same past-only 24h sender count the rule engine uses.

        Returns:
            Scores in [0, 1].

        Raises:
            DataValidationError: If ``transactions`` or the velocity feature is absent.
        """
        if batch.transactions is None or VELOCITY_FEATURE not in batch.features.columns:
            raise DataValidationError("rule triage needs transactions and src_tx_24h")
        velocity = batch.features[VELOCITY_FEATURE].to_numpy().astype(np.int64)
        flags = apply_rules(batch.transactions, self.config, velocity)
        fired = flags[N_RULES].to_numpy(dtype=np.float64)
        return TriageOutput(scores=fired / len(self.config.enabled))
