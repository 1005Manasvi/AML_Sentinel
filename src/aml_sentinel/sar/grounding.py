"""Verify that every account, amount and date in a draft exists in the case facts.

The checks run on the *tokenised* draft, before re-identification:

* every ``ACCT-nnn`` token must belong to the case;
* every dollar amount written ``$1,234.56`` must equal a case amount (to the cent);
* every ``YYYY-MM-DD`` date must be a case date.

Mismatches are returned as issues and shown to the analyst. A draft with issues is
never silently accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from aml_sentinel.sar.facts import CaseFacts

ACCOUNT_RE: Final = re.compile(r"\bACCT-\d{3}\b")
AMOUNT_RE: Final = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
DATE_RE: Final = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Regulatory constants a narrative may cite even though no payment has that amount.
ALLOWED_CONSTANTS_USD: Final = frozenset({10_000.00})


@dataclass(frozen=True, slots=True)
class GroundingIssue:
    """One ungrounded reference.

    Attributes:
        kind: ``"account"``, ``"amount"`` or ``"date"``.
        value: The offending text as written in the draft.
    """

    kind: str
    value: str


def check_grounding(text: str, facts: CaseFacts) -> list[GroundingIssue]:
    """List every reference in ``text`` that the facts do not support.

    Args:
        text: Tokenised draft text.
        facts: Tokenised case facts.

    Returns:
        Issues, empty when the draft is fully grounded.
    """
    issues: list[GroundingIssue] = []
    accounts = facts.accounts
    for token in sorted(set(ACCOUNT_RE.findall(text))):
        if token not in accounts:
            issues.append(GroundingIssue("account", token))
    cents = {round(a * 100) for a in facts.amounts | ALLOWED_CONSTANTS_USD}
    for raw in sorted(set(AMOUNT_RE.findall(text))):
        value = float(raw.replace(",", ""))
        if round(value * 100) not in cents:
            issues.append(GroundingIssue("amount", f"${raw}"))
    dates = facts.dates
    for d in sorted(set(DATE_RE.findall(text))):
        if d not in dates:
            issues.append(GroundingIssue("date", d))
    return issues
