"""Turn an alert's features into a short plain-text summary for Laya (or Jev).

Design rules:

* **Numbers only.** Every sentence is rendered from a numeric feature through a fixed
  template. No account IDs, bank codes, names or other free text reach the summary,
  so there is nothing to prompt-inject and nothing personal to leak (data
  minimisation for the benchmark-only Jev adapter).
* **No shortcut fields.** Payment format and the cross-currency flag are left out
  (ADR-010), so the text models cannot simply relearn the simulator artifacts.
* **Bounded length.** Roughly 120-180 words, far below Laya's 512-token context.
  The adapter still checks Laya's own ``usage["truncated"]`` flag.
* **Augmentation for training only.** ``noise`` drops some facts, jitters numbers
  and shuffles sentence order, so a fine-tuned model cannot memorise one template.
  Inference always uses the canonical, noise-free rendering.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from aml_sentinel.exceptions import DataValidationError

REPORTING_THRESHOLD_USD: Final = 10_000.0
MAX_SUMMARY_CHARS: Final = 1_500
WEEKDAYS: Final = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# The only characters a summary may contain: defence in depth against anything but
# rendered numbers and fixed template words slipping through.
_ALLOWED: Final = re.compile(r"^[A-Za-z0-9 .,:;%()$/\-\n]+$")

REQUIRED_FEATURES: Final = (
    "amount_usd_log",
    "round_amount",
    "hour",
    "weekday",
    "same_bank",
    "self_transfer",
    "src_tx_1h",
    "src_tx_24h",
    "src_usd_24h_log",
    "dst_rx_24h",
    "dst_usd_24h_log",
    "src_out_degree",
    "src_in_degree",
    "src_flow_ratio",
    "src_reciprocal",
    "src_cycle3",
    "src_has_history",
    "src_age_days",
    "dst_out_degree",
    "dst_in_degree",
    "dst_flow_ratio",
    "dst_reciprocal",
    "dst_cycle3",
    "dst_has_history",
    "dst_age_days",
)


@dataclass(frozen=True, slots=True)
class Noise:
    """Training-time augmentation settings.

    Attributes:
        drop_prob: Probability of omitting each optional sentence.
        jitter: Relative multiplicative noise applied to displayed numbers.
        shuffle: Shuffle the order of the sender/receiver sentences.
    """

    drop_prob: float = 0.15
    jitter: float = 0.05
    shuffle: bool = True


def _usd(log_value: float) -> float:
    return float(math.expm1(max(log_value, 0.0)))


def _money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:,.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:,.1f}k"
    return f"${v:,.0f}"


def _account_sentences(
    f: Mapping[str, float], side: str, role: str, num: Callable[[float], float]
) -> list[str]:
    if f[f"{side}_has_history"] < 0.5:
        return [f"The {role} account has no activity before today (new account)."]
    out = [
        f"The {role} account is {num(f[f'{side}_age_days']):.1f} days old in the data, "
        f"pays {num(f[f'{side}_out_degree']):.0f} distinct counterparties and is paid by "
        f"{num(f[f'{side}_in_degree']):.0f}.",
        f"{100 * f[f'{side}_flow_ratio']:.0f}% of past money flow for the {role} account is "
        "incoming.",
    ]
    if f[f"{side}_reciprocal"] > 0:
        out.append(f"The {role} account has {f[f'{side}_reciprocal']:.0f} two-way counterparties.")
    if f[f"{side}_cycle3"] > 0:
        out.append(
            f"The {role} account sits on {f[f'{side}_cycle3']:.0f} three-account payment cycles."
        )
    return out


def summarize(
    features: Mapping[str, float],
    *,
    noise: Noise | None = None,
    rng: np.random.Generator | None = None,
) -> str:
    """Render one alert summary.

    Args:
        features: One feature row (see :data:`REQUIRED_FEATURES`).
        noise: Augmentation settings (training only); ``None`` for canonical text.
        rng: Random generator; required when ``noise`` is given.

    Returns:
        The summary text.

    Raises:
        DataValidationError: If a feature is missing or non-finite, or the result
            contains disallowed characters or exceeds :data:`MAX_SUMMARY_CHARS`.
    """
    missing = [k for k in REQUIRED_FEATURES if k not in features]
    if missing:
        raise DataValidationError(f"summary missing features: {missing}")
    if any(not math.isfinite(float(features[k])) for k in REQUIRED_FEATURES):
        raise DataValidationError("summary features must be finite")
    if noise is not None and rng is None:
        raise DataValidationError("rng is required when noise is enabled")
    f = {k: float(features[k]) for k in REQUIRED_FEATURES}

    def num(v: float) -> float:
        if noise is None or rng is None or noise.jitter <= 0:
            return v
        return v * float(rng.uniform(1 - noise.jitter, 1 + noise.jitter))

    amount = num(_usd(f["amount_usd_log"]))
    head = [
        f"Payment of {_money(amount)} on a {WEEKDAYS[int(f['weekday']) % 7]} at "
        f"{int(f['hour']):02d}:00.",
    ]
    facts: list[str] = []
    if 0.8 * REPORTING_THRESHOLD_USD <= amount < REPORTING_THRESHOLD_USD:
        facts.append("The amount is just below the 10k reporting threshold.")
    if f["round_amount"] > 0.5:
        facts.append("The amount is a round number.")
    if f["self_transfer"] > 0.5:
        facts.append("The sender and receiver are the same account.")
    elif f["same_bank"] > 0.5:
        facts.append("Both accounts are at the same bank.")
    facts += [
        f"In the last hour the sender made {num(f['src_tx_1h']):.0f} payments; in the last "
        f"24 hours {num(f['src_tx_24h']):.0f} payments totalling "
        f"{_money(num(_usd(f['src_usd_24h_log'])))}.",
        f"In the last 24 hours the receiver got {num(f['dst_rx_24h']):.0f} payments totalling "
        f"{_money(num(_usd(f['dst_usd_24h_log'])))}.",
    ]
    sender = _account_sentences(f, "src", "sender", num)
    receiver = _account_sentences(f, "dst", "receiver", num)

    body = facts + sender + receiver
    if noise is not None and rng is not None:
        body = [s for s in body if rng.random() >= noise.drop_prob] or body[:1]
        if noise.shuffle:
            body = [body[i] for i in rng.permutation(len(body))]
    text = " ".join(head + body)
    validate_summary(text)
    return text


def validate_summary(text: str) -> None:
    """Check that a summary is short and contains only template characters.

    Args:
        text: Summary text.

    Raises:
        DataValidationError: If the text is empty, too long or has disallowed characters.
    """
    if not text or len(text) > MAX_SUMMARY_CHARS:
        raise DataValidationError("summary is empty or too long")
    if not _ALLOWED.match(text):
        raise DataValidationError("summary contains disallowed characters")
