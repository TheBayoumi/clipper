"""Candidate answers cannot bypass exact source citation or source-only scope."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clipper.editorial_answer_comparison import compare_claim_answer

QUESTION = "Who paced at the fighter meeting?"
UNITS = ["Sean Shelby spoke.", "Bobby Green paced behind him."]


def source_answer():
    return {
        "question": QUESTION,
        "status": "answered",
        "answer_quote": "Bobby Green",
        "citation": {"first_unit": 1, "last_unit": 1, "text": UNITS[1]},
    }


def scope_review():
    answer = source_answer()
    return {
        "question": QUESTION,
        "answer_quote": answer["answer_quote"],
        "citation": answer["citation"],
        "responsiveness": "answers_question",
        "scope": "actual_event_or_state",
        "diagnostic_only": True,
        "production_approved": False,
    }


def compare(**changes):
    args = {
        "question": QUESTION,
        "proposed_answer": "Sean Shelby",
        "source_answer": source_answer(),
        "scope_review": scope_review(),
        "delivered_units": UNITS,
        "completion": lambda *_: {
            "responsiveness": "answers_question",
            "source_scope": "actual_event_or_state",
            "relation": "different",
        },
        **changes,
    }
    return compare_claim_answer(**args)


def test_comparison_preserves_role_difference_and_never_approves():
    seen = []

    def completion(prompt, payload, schema, tokens):
        seen.append(payload)
        assert "actual-versus-conditional" in prompt
        assert schema["relation"]["enum"] == ["equivalent", "different", "uncertain"]
        assert "responsiveness" in schema
        assert "source_scope" in schema
        assert tokens == 96
        return {
            "responsiveness": "answers_question",
            "source_scope": "actual_event_or_state",
            "relation": "different",
        }

    result = compare(completion=completion)
    assert seen[0]["proposed_answer"] == "Sean Shelby"
    assert seen[0]["source_answer_quote"] == "Bobby Green"
    assert "source_scope" not in seen[0]
    assert len(seen[0]["source_units"]) == 2
    assert result["relation"] == "different"
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "responsiveness,source_scope",
    [
        ("does_not_answer", "actual_event_or_state"),
        ("answers_question", "quoted_instruction"),
        ("uncertain", "unknown"),
    ],
)
def test_independent_scope_disagreement_vetoes_equivalent_label(responsiveness, source_scope):
    result = compare(
        completion=lambda *_: {
            "responsiveness": responsiveness,
            "source_scope": source_scope,
            "relation": "equivalent",
        }
    )
    assert result["raw_relation"] == "equivalent"
    assert result["relation"] == "uncertain"
    assert result["independent_scope_agrees"] is False


@pytest.mark.parametrize(
    "mutation", ["question", "answer", "scope", "citation", "speech", "scope_approval"]
)
def test_comparison_rejects_unresponsive_or_tampered_source(mutation):
    answer, scope = source_answer(), scope_review()
    kwargs = {}
    if mutation == "question":
        answer["question"] = "Different?"
    elif mutation == "answer":
        scope["answer_quote"] = "Sean Shelby"
    elif mutation == "scope":
        scope["responsiveness"] = "does_not_answer"
    elif mutation == "citation":
        answer["citation"] = deepcopy(answer["citation"])
        answer["citation"]["text"] = "fabricated"
        scope["citation"] = answer["citation"]
    elif mutation == "scope_approval":
        scope["production_approved"] = True
    else:
        kwargs["delivered_units"] = ["Changed."]
    with pytest.raises(ValueError, match=r"responsive|citation"):
        compare(source_answer=answer, scope_review=scope, **kwargs)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"question": "Not a question"}, "relation question"),
        ({"proposed_answer": ""}, "proposed headline answer"),
        ({"delivered_units": []}, "delivered source speech"),
        ({"source_answer": {"question": QUESTION, "status": "unknown"}}, "responsive"),
    ],
)
def test_comparison_rejects_invalid_inputs(change, error):
    with pytest.raises(ValueError, match=error):
        compare(**change)


def test_comparison_rejects_empty_answer_quote_after_scope_match():
    answer, scope = source_answer(), scope_review()
    answer["answer_quote"] = ""
    scope["answer_quote"] = ""
    with pytest.raises(ValueError, match="exact cited source quote"):
        compare(source_answer=answer, scope_review=scope)


@pytest.mark.parametrize(
    "proposal",
    [
        {},
        {
            "responsiveness": "answers_question",
            "source_scope": "actual_event_or_state",
            "relation": "same",
        },
    ],
)
def test_comparison_rejects_invalid_model_contract(proposal):
    with pytest.raises(ValueError, match=r"missing or unknown|invalid relation"):
        compare(completion=lambda *_: proposal)
