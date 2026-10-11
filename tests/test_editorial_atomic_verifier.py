"""Atomic relation checks are source-first and diagnostic even on agreement."""

from __future__ import annotations

import pytest

from clipper import editorial_propositions
from clipper.editorial_atomic_verifier import verify_atomic_relation


@pytest.fixture(autouse=True)
def isolate_source_only_extraction(monkeypatch):
    original = editorial_propositions.extract_source_proposition

    def source_only_evidence(**kwargs):
        return original(
            **{key: val for key, val in kwargs.items() if key != "completion"},
            completion=lambda *_: {
                "narrative_role": "recounted_event",
                "attribution": "first_person_account",
                "polarity": "affirmed",
                "modality": "asserted",
                "timeframe": "past",
                "actor": "Bobby Green",
                "predicate": "paced",
            },
        )

    monkeypatch.setattr(editorial_propositions, "extract_source_proposition", source_only_evidence)


QUESTION = "Who paced at the fighter meeting?"
UNITS = ["Sean Shelby spoke.", "Bobby Green paced in the back."]


def answer(prompt, payload, schema, tokens):
    return {
        "status": "answered",
        "answer_quote": "Bobby Green",
        "first_unit": 1,
        "last_unit": 1,
    }


def scope(prompt, payload, schema, tokens):
    if "question_target" in schema:
        return {"question_target": "described_event_or_state"}
    return {
        "responsiveness": "answers_question",
        "evidence_first_unit": 1,
        "evidence_last_unit": 1,
        "evidence_quote": "Bobby Green paced",
    }


def relation(prompt, payload, schema, tokens):
    return {
        "relation": "equivalent",
        "claim_type": "event_or_state",
        "event_link": "same_event",
        "event_first_unit": 1,
        "event_last_unit": 1,
    }


def test_atomic_relation_withholds_candidate_until_source_is_scoped():
    calls = []

    def source_completion(prompt, payload, schema, tokens):
        calls.append(("source", payload))
        assert "proposed_answer" not in payload
        return answer(prompt, payload, schema, tokens)

    def scope_completion(prompt, payload, schema, tokens):
        calls.append(("scope", payload))
        assert "proposed_answer" not in payload
        return scope(prompt, payload, schema, tokens)

    def comparison_completion(prompt, payload, schema, tokens):
        calls.append(("comparison", payload))
        return relation(prompt, payload, schema, tokens)

    result = verify_atomic_relation(
        question=QUESTION,
        proposed_answer="Bobby Green",
        delivered_units=UNITS,
        source_completion=source_completion,
        scope_completion=scope_completion,
        comparison_completion=comparison_completion,
    )
    assert [name for name, _ in calls] == ["source", "scope", "scope", "comparison"]
    assert result["verdict"] == "supported_label_unqualified"
    assert result["claim_inventory_semantically_qualified"] is False
    assert result["source_entailment_qualified"] is False
    assert result["production_approved"] is False


def test_atomic_relation_abstains_without_source_answer_or_comparator_call():
    result = verify_atomic_relation(
        question=QUESTION,
        proposed_answer="Bobby Green",
        delivered_units=UNITS,
        source_completion=lambda *_: {
            "status": "unknown",
            "answer_quote": "",
            "first_unit": -1,
            "last_unit": -1,
        },
        scope_completion=lambda *_: (_ for _ in ()).throw(AssertionError("scope model called")),
        comparison_completion=lambda *_: (_ for _ in ()).throw(AssertionError("comparator called")),
    )
    assert result["verdict"] == "uncertain"
    assert result["answer_comparison"] is None


def test_atomic_relation_stops_on_nonresponsive_exact_quote():
    result = verify_atomic_relation(
        question=QUESTION,
        proposed_answer="Bobby Green",
        delivered_units=UNITS,
        source_completion=answer,
        scope_completion=lambda _p, _v, schema, _t: (
            {"question_target": "described_event_or_state"}
            if "question_target" in schema
            else {
                "responsiveness": "does_not_answer",
                "evidence_first_unit": -1,
                "evidence_last_unit": -1,
                "evidence_quote": "",
            }
        ),
        comparison_completion=lambda *_: (_ for _ in ()).throw(AssertionError("comparator called")),
    )
    assert result["verdict"] == "uncertain"
    assert result["answer_comparison"] is None


@pytest.mark.parametrize(
    "comparison_label,expected",
    [
        ("different", "unsupported_label_unqualified"),
        ("uncertain", "uncertain"),
    ],
)
def test_atomic_relation_does_not_promote_different_or_uncertain_answer(comparison_label, expected):
    result = verify_atomic_relation(
        question=QUESTION,
        proposed_answer="Sean Shelby",
        delivered_units=UNITS,
        source_completion=answer,
        scope_completion=scope,
        comparison_completion=lambda *_: {
            **relation(None, None, None, None),
            "relation": comparison_label,
        },
    )
    assert result["verdict"] == expected
    assert result["production_approved"] is False


def test_atomic_relation_rejects_empty_candidate_and_bad_source_contract():
    with pytest.raises(ValueError, match="proposed answer"):
        verify_atomic_relation(
            question=QUESTION,
            proposed_answer="",
            delivered_units=UNITS,
            source_completion=answer,
            scope_completion=scope,
            comparison_completion=relation,
        )
    with pytest.raises(ValueError, match="source answer has missing"):
        verify_atomic_relation(
            question=QUESTION,
            proposed_answer="Bobby Green",
            delivered_units=UNITS,
            source_completion=lambda *_: {},
            scope_completion=scope,
            comparison_completion=relation,
        )
