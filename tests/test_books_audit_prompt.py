"""The audit prompt must tell the model how to treat gap units, with a valid example id."""

from __future__ import annotations

from test_books_chapter_hybrid import _ledger

from clipper.books import chapter_hybrid as hybrid


def _audit_user_text() -> str:
    ledger = _ledger()
    window = hybrid.audit_windows(ledger)[0]
    messages = hybrid._messages("audit", {"window": window, "response_contract": {"x": 1}})
    return " ".join(str(m["content"]) for m in messages)


def test_audit_guidance_says_gap_decisions_omit_with_empty_evidence() -> None:
    text = _audit_user_text()
    assert "empty list" in text
    assert "must be omit" in text
    assert "Never cite a gap id" in text
