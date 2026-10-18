"""Edited SAR drafts: review stamp enforcement and grounding over real account ids."""

from __future__ import annotations

from aml_sentinel.sar.drafting import REVIEW_STAMP, draft_sar
from aml_sentinel.sar.facts import CaseFacts, PatternFact, TxFact
from aml_sentinel.sar.review import check_reviewed_text, restamp

A, B, C = "010:AAAA1111", "020:BBBB2222", "030:CC"  # C is short: not matched by the id regex


def _facts() -> CaseFacts:
    txs = (
        TxFact(1, "2022-09-08", A, B, 9_500.00, "Cash"),
        TxFact(2, "2022-09-09", B, C, 9_400.00, "Cash"),
    )
    pattern = PatternFact("fan_out", (B, C), (2,), 9_400.00, "2022-09-09", "2022-09-09", 1.0)
    return CaseFacts("CASE-2", txs[1], (B, C), txs, (pattern,), risk_score=0.4)


def test_restamp_is_idempotent_and_cannot_be_removed() -> None:
    once = restamp("Body text.")
    assert once == f"{REVIEW_STAMP}\nBody text.\n\n{REVIEW_STAMP}\n"
    assert restamp(once) == once
    assert restamp(f"{REVIEW_STAMP}\n{REVIEW_STAMP}\nBody text.") == once


def test_restamp_drops_control_characters_and_normalises_newlines() -> None:
    out = restamp("line1\r\nline2\x00\x1b[31m\rline3\tok")
    assert "\x00" not in out
    assert "\x1b" not in out
    assert "line1\nline2[31m\nline3\tok" in out


def test_generated_draft_passes_review_check() -> None:
    facts = _facts()
    draft = draft_sar(facts)
    assert draft.is_grounded
    assert check_reviewed_text(draft.text, facts) == []


def test_unknown_accounts_amounts_and_dates_are_flagged() -> None:
    text = f"{A} paid {B} $9,500.00 on 2022-09-08. Also 999:ZZZZ0000 got $1.00 on 2030-01-01."
    issues = {(i.kind, i.value) for i in check_reviewed_text(text, _facts())}
    assert issues == {("account", "999:ZZZZ0000"), ("amount", "$1.00"), ("date", "2030-01-01")}


def test_known_short_ids_and_clock_times_are_not_flagged() -> None:
    text = f"{B} paid {C} $9,400.00 on 2022-09-09 at 10:30."
    assert check_reviewed_text(text, _facts()) == []


def test_typed_tokens_must_still_exist() -> None:
    issues = check_reviewed_text("ACCT-001 and ACCT-077 were involved.", _facts())
    assert [(i.kind, i.value) for i in issues] == [("account", "ACCT-077")]
