"""Blind source-answer and Python-owned source QA contracts."""

from __future__ import annotations

import pytest

from clipper.editorial_qa import audit_source_qa


class Reviewer:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def _review_completion(self, prompt, payload, properties, tokens):
        self.calls.append((prompt, payload))
        return next(self.responses)


def proposition(headline, answer, scope="actual_event"):
    return {
        "claim_text": headline,
        "question": "What happened to the speaker?",
        "claimed_answer": answer,
        "claimed_scope": scope,
    }


def source_answer(quote, scope="actual_event", first=0, last=0):
    return {
        "answer_quote": quote,
        "source_scope": scope,
        "first_unit": first,
        "last_unit": last,
    }


def test_blind_source_answer_and_full_coverage_can_support():
    headline = "The speaker stopped sparring"
    reviewer = Reviewer(
        {"questions": [proposition(headline, "stopped sparring")]},
        source_answer("stopped sparring"),
        {"relation": "equivalent", "reason": "Same completed action."},
    )
    audit = audit_source_qa(reviewer, headline, ["He stopped sparring yesterday."])
    assert audit["verdict"] == "supported"
    assert set(reviewer.calls[1][1]) == {"question", "source_units"}
    assert "headline" not in reviewer.calls[1][1]
    assert "stopped sparring" not in reviewer.calls[1][1]["question"]


def test_future_plan_cannot_support_completed_event_without_comparator_call():
    headline = "The speaker already stopped sparring"
    reviewer = Reviewer(
        {"questions": [proposition(headline, "already stopped sparring")]},
        source_answer("going to reassess in the future", "future_plan"),
    )
    audit = audit_source_qa(reviewer, headline, ["I am going to reassess in the future."])
    assert audit["verdict"] == "uncertain"
    assert len(reviewer.calls) == 2


def test_missing_headline_relation_never_approves():
    headline = "The speaker stopped sparring after six wins"
    reviewer = Reviewer(
        {"questions": [proposition("The speaker stopped sparring", "stopped sparring")]},
        source_answer("stopped sparring"),
        {"relation": "equivalent", "reason": "Same action."},
    )
    audit = audit_source_qa(reviewer, headline, ["He stopped sparring."])
    assert audit["verdict"] == "uncertain"
    assert audit["uncovered_headline_words"] == ["after", "six", "wins"]


def test_uncited_or_paraphrased_source_answer_is_contract_error():
    headline = "The speaker stopped sparring"
    reviewer = Reviewer(
        {"questions": [proposition(headline, "stopped sparring")]},
        source_answer("quit training"),
    )
    with pytest.raises(RuntimeError, match="exact cited source passage"):
        audit_source_qa(reviewer, headline, ["He stopped sparring yesterday."])
