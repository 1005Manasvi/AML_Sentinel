"""Map simulator typologies onto the five labels Laya is asked to choose from.

The owner approved this mapping (ADR-012):

| Laya label     | Source typologies                                            |
|----------------|--------------------------------------------------------------|
| circular       | CYCLE                                                        |
| mule_network   | FAN-IN, FAN-OUT (one hub, many one-hop counterparties)       |
| layering       | SCATTER-GATHER, GATHER-SCATTER, STACK, BIPARTITE, RANDOM     |
| structuring    | STRUCTURING (synthetic generator only; absent from IBM data) |
| benign         | label 0                                                      |

Laundering rows with no documented typology have **no** typology label. They are
still positives for the risk and escalation questions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from aml_sentinel.data.patterns import Typology


class LayaLabel(StrEnum):
    """Closed label set for the typology question. Five labels, well under Laya's limit."""

    STRUCTURING = "structuring"
    CIRCULAR = "circular"
    MULE_NETWORK = "mule_network"
    LAYERING = "layering"
    BENIGN = "benign"


TYPOLOGY_TO_LABEL: Final[dict[Typology, LayaLabel]] = {
    Typology.CYCLE: LayaLabel.CIRCULAR,
    Typology.FAN_IN: LayaLabel.MULE_NETWORK,
    Typology.FAN_OUT: LayaLabel.MULE_NETWORK,
    Typology.SCATTER_GATHER: LayaLabel.LAYERING,
    Typology.GATHER_SCATTER: LayaLabel.LAYERING,
    Typology.STACK: LayaLabel.LAYERING,
    Typology.BIPARTITE: LayaLabel.LAYERING,
    Typology.RANDOM: LayaLabel.LAYERING,
    Typology.STRUCTURING: LayaLabel.STRUCTURING,
}


def laya_label(is_laundering: bool, typology: str | None) -> LayaLabel | None:
    """Return the Laya typology label for one transaction.

    Args:
        is_laundering: Ground-truth label.
        typology: Simulator typology, or ``None`` / empty if undocumented.

    Returns:
        ``BENIGN`` for negatives, the mapped label for documented positives, or
        ``None`` for undocumented positives (their typology is unknown).
    """
    if not is_laundering:
        return LayaLabel.BENIGN
    if not typology:
        return None
    return TYPOLOGY_TO_LABEL[Typology(typology)]
