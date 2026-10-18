"""Checks for analyst-edited SAR drafts.

Generated drafts are checked over pseudonymised tokens (:mod:`aml_sentinel.sar.grounding`).
An analyst edits the *re-identified* text, so it contains real account ids. This
module maps known ids back to their tokens, flags any other account-like id, and then
runs the same amount, date and token checks as for generated drafts.

The review stamp is enforced on the server: whatever the analyst submits, the stored
text starts and ends with ``DRAFT - requires analyst review``.
"""

from __future__ import annotations

import re
from typing import Final

from aml_sentinel.sar.drafting import REVIEW_STAMP
from aml_sentinel.sar.facts import CaseFacts
from aml_sentinel.sar.grounding import GroundingIssue, check_grounding
from aml_sentinel.sar.pseudonymize import pseudonymize

# "<bank>:<account>" as built by account_node_id. At least four characters after the
# colon, so clock times such as 10:30 are not mistaken for accounts.
ACCOUNT_ID_RE: Final = re.compile(r"\b\d{1,10}:[0-9A-Za-z]{4,32}\b")
_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def restamp(text: str) -> str:
    """Normalise text and wrap it in the mandatory review stamp.

    Existing stamp lines are removed first, so stamping is idempotent and an analyst
    cannot remove or duplicate it. Control characters (other than newlines and tabs)
    are dropped.

    Args:
        text: Draft text as edited.

    Returns:
        The stamp, the body, a blank line and the stamp again.
    """
    clean = _CONTROL_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    body = "\n".join(line for line in clean.split("\n") if line.strip() != REVIEW_STAMP)
    return f"{REVIEW_STAMP}\n{body.strip()}\n\n{REVIEW_STAMP}\n"


def check_reviewed_text(text: str, facts: CaseFacts) -> list[GroundingIssue]:
    """Ground an edited draft that uses real account ids.

    Args:
        text: Edited draft (real identifiers).
        facts: Real case facts.

    Returns:
        Issues: unknown account ids, plus every ungrounded token, amount or date.
    """
    pseudo = pseudonymize(facts)
    to_token = {real: token for token, real in pseudo.to_real.items()}
    # Longest first, so an id that is a prefix of another never splits it.
    known = re.compile(
        r"(?<![0-9A-Za-z:])("
        + "|".join(re.escape(r) for r in sorted(to_token, key=len, reverse=True))
        + r")(?![0-9A-Za-z])"
    )
    tokenised = known.sub(lambda m: to_token[m.group(1)], text)
    unknown = sorted(set(ACCOUNT_ID_RE.findall(tokenised)))
    issues = [GroundingIssue("account", value) for value in unknown]
    return issues + check_grounding(tokenised, pseudo.facts)
