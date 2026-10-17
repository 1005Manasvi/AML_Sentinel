"""Replace account identifiers with case-local tokens before any drafting.

``ACCT-001`` is always the alert's payer and ``ACCT-002`` its payee. Other accounts
are numbered in order of first appearance. The mapping lives only in memory on this
machine. Drafts are produced over tokens, and real identifiers are restored locally
only after the draft passes validation.

Amounts, dates and payment formats are kept, because the narrative must state them
exactly. They are synthetic and carry no identity once accounts are tokenised. With
real data, a separate privacy review would have to decide whether amounts may be
shared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Final

from aml_sentinel.exceptions import DataValidationError
from aml_sentinel.sar.facts import CaseFacts, PatternFact, TxFact

TOKEN_RE: Final = re.compile(r"\bACCT-\d{3}\b")


@dataclass(frozen=True, slots=True)
class Pseudonymized:
    """Tokenised facts plus the local re-identification map.

    Attributes:
        facts: Facts with every account id replaced by a token.
        to_real: ``{token: real account id}`` (never leaves the machine).
    """

    facts: CaseFacts
    to_real: dict[str, str]

    def reidentify(self, text: str) -> str:
        """Replace tokens in ``text`` with real account ids.

        Args:
            text: Text containing tokens.

        Returns:
            Text with real identifiers.

        Raises:
            DataValidationError: If the text contains a token not in the map.
        """

        def swap(match: re.Match[str]) -> str:
            token = match.group(0)
            if token not in self.to_real:
                raise DataValidationError("draft references an unknown account token")
            return self.to_real[token]

        return TOKEN_RE.sub(swap, text)


def pseudonymize(facts: CaseFacts) -> Pseudonymized:
    """Tokenise every account id in the fact sheet.

    Args:
        facts: Real case facts.

    Returns:
        Tokenised facts and the re-identification map.

    Raises:
        DataValidationError: If a case has more than 999 accounts.
    """
    order: list[str] = list(dict.fromkeys(facts.subjects))
    for t in facts.transactions:
        for a in (t.payer, t.payee):
            if a not in order:
                order.append(a)
    for p in facts.patterns:
        order.extend(a for a in p.accounts if a not in order)
    if len(order) > 999:
        raise DataValidationError("too many accounts to tokenise")
    to_token = {real: f"ACCT-{i:03d}" for i, real in enumerate(order, start=1)}

    def tx(t: TxFact) -> TxFact:
        return replace(t, payer=to_token[t.payer], payee=to_token[t.payee])

    def pat(p: PatternFact) -> PatternFact:
        return replace(p, accounts=tuple(to_token[a] for a in p.accounts))

    tokenised = replace(
        facts,
        alert=tx(facts.alert),
        subjects=(to_token[facts.subjects[0]], to_token[facts.subjects[1]]),
        transactions=tuple(tx(t) for t in facts.transactions),
        patterns=tuple(pat(p) for p in facts.patterns),
    )
    return Pseudonymized(facts=tokenised, to_real={v: k for k, v in to_token.items()})
