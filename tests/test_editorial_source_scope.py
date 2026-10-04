"""An exact citation remains unapproved until responsiveness and scope qualify."""

from __future__ import annotations

import pytest

from clipper.editorial_source_scope import assess_source_answer_scope

QUESTION = "What was the actual broadcast setting while Bobby paced?"
UNITS = [
    "Sean went over the fighter meeting instructions.",
    "We're on live TV, don't do this. Bobby paced in the back.",
]


def answer():
    return {
        "question": QUESTION,
        "status": "answered",
        "answer_quote": "We're on live TV",
        "citation": {
            "first_unit": 1,
            "last_unit": 1,
            "text": UNITS[1],
        },
    }


def test_source_scope_sees_question_quote_and_full_context_but_no_headline_answer():
    seen = []

    def completion(prompt, payload, schema, tokens):
        seen.append(payload)
        assert "substring can be exact yet nonresponsive" in prompt
        assert schema["scope"]["enum"]
        assert tokens == 80
        return {"responsiveness": "does_not_answer", "scope": "quoted_instruction"}

    result = assess_source_answer_scope(
        question=QUESTION, source_answer=answer(), delivered_units=UNITS, completion=completion
    )
    assert set(seen[0]) == {
        "question",
        "answer_quote",
        "cited_first_unit",
        "cited_last_unit",
        "source_units",
    }
    assert result["scope"] == "quoted_instruction"
    assert result["production_approved"] is False


def test_source_scope_abstained_answer_uses_no_model_call():
    source_answer = {"question": QUESTION, "status": "unknown"}
    result = assess_source_answer_scope(
        question=QUESTION,
        source_answer=source_answer,
        delivered_units=UNITS,
        completion=lambda *_: (_ for _ in ()).throw(AssertionError("no model call")),
    )
    assert result["responsiveness"] == "uncertain"
    assert result["scope"] == "unknown"
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"question": "Different?"}, "does not match"),
        ({"answer_quote": "invented"}, "not exact delivered"),
        ({"citation": {"first_unit": 2, "last_unit": 2, "text": ""}}, "positions"),
        ({"citation": {"first_unit": 1, "last_unit": 1, "text": "changed"}}, "not exact"),
        ({"citation": None}, "exact cited answer"),
    ],
)
def test_source_scope_rejects_tampered_answer_or_citation(mutation, error):
    with pytest.raises(ValueError, match=error):
        assess_source_answer_scope(
            question=QUESTION,
            source_answer={**answer(), **mutation},
            delivered_units=UNITS,
            completion=lambda *_: {},
        )


@pytest.mark.parametrize(
    "proposal,error",
    [
        ({}, "missing or unknown"),
        ({"responsiveness": "answers_question", "scope": "invented"}, "invalid labels"),
    ],
)
def test_source_scope_rejects_invalid_model_response(proposal, error):
    with pytest.raises(ValueError, match=error):
        assess_source_answer_scope(
            question=QUESTION,
            source_answer=answer(),
            delivered_units=UNITS,
            completion=lambda *_: proposal,
        )


def test_source_scope_requires_question_and_delivered_speech():
    with pytest.raises(ValueError, match="factual question"):
        assess_source_answer_scope(
            question="not a question",
            source_answer=answer(),
            delivered_units=UNITS,
            completion=lambda *_: {},
        )
    with pytest.raises(ValueError, match="delivered speech"):
        assess_source_answer_scope(
            question=QUESTION,
            source_answer=answer(),
            delivered_units=[],
            completion=lambda *_: {},
        )
