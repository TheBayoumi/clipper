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
        "claimed_event_status": scope,
    }


def source_answer(scope="actual_event", first=0, last=0, answerable=1):
    return {
        "answerable": answerable,
        "source_event_status": scope,
        "first_unit": first,
        "last_unit": last,
    }


def test_blind_source_answer_and_full_coverage_can_support():
    headline = "The speaker stopped sparring"
    reviewer = Reviewer(
        {"questions": [proposition(headline, "stopped sparring")]},
        source_answer(),
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
        source_answer("future_plan"),
    )
    audit = audit_source_qa(reviewer, headline, ["I am going to reassess in the future."])
    assert audit["verdict"] == "uncertain"
    assert len(reviewer.calls) == 2


def test_missing_headline_relation_never_approves():
    headline = "The speaker stopped sparring after six wins"
    reviewer = Reviewer(
        {"questions": [proposition("The speaker stopped sparring", "stopped sparring")]},
        source_answer(),
        {"relation": "equivalent", "reason": "Same action."},
    )
    audit = audit_source_qa(reviewer, headline, ["He stopped sparring."])
    assert audit["verdict"] == "uncertain"
    assert audit["uncovered_headline_words"] == ["after", "six", "wins"]


def test_uncited_source_answer_is_contract_error():
    headline = "The speaker stopped sparring"
    reviewer = Reviewer(
        {"questions": [proposition(headline, "stopped sparring")]},
        source_answer("unknown", -1, -1, answerable=1),
    )
    with pytest.raises(RuntimeError, match="answerability, event status and evidence"):
        audit_source_qa(reviewer, headline, ["He stopped sparring yesterday."])


def test_source_evidence_is_extracted_by_python_even_if_model_would_paraphrase():
    headline = "Mighty Mouse told the speaker to spar less"
    reviewer = Reviewer(
        {"questions": [proposition(headline, "spar less")]},
        source_answer(),
        {"relation": "equivalent", "reason": "The advice is to reduce sparring."},
    )
    audit = audit_source_qa(reviewer, headline, ["He says I spar too much."])
    assert audit["verdict"] == "supported"
    assert reviewer.calls[2][1]["source_evidence"] == "He says I spar too much."
