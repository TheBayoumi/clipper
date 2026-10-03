"""The diagnostic verifier binds model choices to immutable source and headline."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clipper.editorial_structured_claims import audit_structured_claims

HEADLINE = "The speaker says fighters can be cut if boring"
UNITS = [
    "You guys are independent contractors.",
    "You can be a great fighter and get cut.",
    "You're too boring, right?",
]


class Reviewer:
    def __init__(self, *, source_event: str = "hypothetical", omit_word: bool = False) -> None:
        self.calls: list[tuple[str, dict, dict]] = []
        self.source_event = source_event
        self.omit_word = omit_word

    def _review_completion(self, prompt, payload, properties, tokens):
        self.calls.append((prompt, payload, properties))
        if "word_index" in payload:
            last = len(payload["word_index"]) - (2 if self.omit_word else 1)
            return {"claims": [{"kind": "event_modality", "first_word": 0, "last_word": last}]}
        if "claim" in payload:
            return {
                "verdict": "supported",
                "claimed_reporting_status": "actual_report",
                "claimed_event_status": "hypothetical",
                "first_unit": 0,
                "last_unit": 2,
                "reason": "The speaker reports the conditional cut rule.",
            }
        assert "headline" not in payload and "claim" not in payload
        return {
            "source_reporting_status": "actual_report",
            "source_event_status": self.source_event,
            "reason": "The cited rule describes a possible cut.",
        }


def audit(reviewer: Reviewer, **changes):
    return audit_structured_claims(
        reviewer,
        HEADLINE,
        deepcopy(UNITS),
        source_video_id="_kDrxucOx9g",
        source_sha256="a" * 64,
        transcript_sha256="b" * 64,
        **changes,
    )


def test_structured_review_preserves_whole_claim_and_blind_source_scope():
    reviewer = Reviewer()
    result = audit(reviewer)
    assert result["validated"]["all_claims_supported"] is True
    assert result["validated"]["production_approved"] is False
    assert result["validated"]["whole_headline"]["source_text"] == " ".join(UNITS)
    assert ["claim" in payload for _, payload, _ in reviewer.calls] == [
        False,
        True,
        False,
        True,
        False,
    ]


def test_structured_review_rejects_uncovered_headline_words():
    with pytest.raises(RuntimeError, match="omitted headline words"):
        audit(Reviewer(omit_word=True))


def test_structured_review_cannot_support_hypothetical_as_actual():
    with pytest.raises(ValueError, match="mismatched"):
        audit(Reviewer(source_event="actual_event"))


def test_structured_review_needs_delivered_speech():
    with pytest.raises(ValueError, match="delivered source speech"):
        audit_structured_claims(
            Reviewer(),
            "No source",
            [],
            source_video_id="_kDrxucOx9g",
            source_sha256="a" * 64,
            transcript_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "phase,replacement,error",
    [
        ("words", {"claims": []}, "omitted atomic"),
        ("words", {"claims": [{"kind": "attribution"}]}, "invalid word-span"),
        (
            "words",
            {"claims": [{"kind": "attribution", "first_word": 0, "last_word": 99}]},
            "outside the exact headline",
        ),
        ("claim", {"extra": 1}, "invalid claim assessment"),
        ("claim", {"first_unit": -1}, "invalid source citation"),
        ("scope", {"extra": 1}, "invalid source-scope"),
    ],
)
def test_structured_review_rejects_invalid_model_contracts(phase, replacement, error):
    class BrokenReviewer(Reviewer):
        def _review_completion(self, prompt, payload, properties, tokens):
            result = super()._review_completion(prompt, payload, properties, tokens)
            current = (
                "words" if "word_index" in payload else "claim" if "claim" in payload else "scope"
            )
            return {**result, **replacement} if current == phase else result

    with pytest.raises(RuntimeError, match=error):
        audit(BrokenReviewer())
