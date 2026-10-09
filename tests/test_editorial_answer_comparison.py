"""Candidate answers cannot bypass exact source citation or source-only scope."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clipper.editorial_answer_comparison import compare_claim_answer

QUESTION = "Who paced at the fighter meeting?"


def response(
    relation="different",
    *,
    source_role="recounted_event",
    claimed_role="recounted_event",
    event_link="same_event",
    unit=1,
):
    return {
        "relation": relation,
        "claimed_narrative_role": claimed_role,
        "source_narrative_role": source_role,
        "event_link": event_link,
        "event_first_unit": unit if event_link == "same_event" else -1,
        "event_last_unit": unit if event_link == "same_event" else -1,
    }


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
        "narrative_role": "recounted_event",
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
        "completion": lambda *_: response(),
        **changes,
    }
    return compare_claim_answer(**args)


def test_comparison_preserves_role_difference_and_never_approves():
    seen = []

    def completion(prompt, payload, schema, tokens):
        seen.append(payload)
        assert "actual-versus-conditional" in prompt
        assert schema["relation"]["enum"] == ["equivalent", "different", "uncertain"]
        assert tokens == 160
        assert "past firsthand experience" in prompt
        assert schema["event_link"]["enum"] == ["same_event", "different_event", "unresolved"]
        return response()

    result = compare(completion=completion)
    assert seen[0]["proposed_answer"] == "Sean Shelby"
    assert seen[0]["source_answer_quote"] == "Bobby Green"
    assert len(seen[0]["source_units"]) == 2
    assert result["relation"] == "different"
    assert result["production_approved"] is False


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
        {**response(), "relation": "same"},
        {**response(), "event_first_unit": 99},
        {**response(), "event_link": "unresolved"},
        {**response(), "claimed_narrative_role": "actual_television_show"},
        {**response(), "event_first_unit": True},
    ],
)
def test_comparison_rejects_invalid_model_contract(proposal):
    with pytest.raises(ValueError, match=r"missing or unknown|invalid|disagree"):
        compare(completion=lambda *_: proposal)


@pytest.mark.parametrize(
    "source_units,question,answer_quote,claimed_answer,source_role,claim_role,alignment,expected",
    [
        (
            [
                "Years ago I was pitching the startup to investors.",
                "My cofounder said, 'we could raise a million' but we didn't.",
            ],
            "Did the startup actually raise a million?",
            "we could raise a million",
            "raised a million",
            "conditional_or_hypothetical",
            "recounted_event",
            "different_event",
            "uncertain",
        ),
        (
            [
                "When I was a nurse, I worked nights in the hospital.",
                "My supervisor told me that we had to move faster.",
            ],
            "Where did the guest say they worked nights?",
            "worked nights in the hospital",
            "hospital",
            "recounted_event",
            "recounted_event",
            "same_event",
            "equivalent",
        ),
        (
            [
                "Back when I was touring I missed the train.",
                "My manager shouted, 'we are live on air now!'",
            ],
            "What did the manager shout?",
            "we are live on air now",
            "we are live on air now",
            "reported_utterance",
            "reported_utterance",
            "same_event",
            "equivalent",
        ),
        (
            [
                "I was working in a kitchen and burned the food.",
                "The head chef said, 'this is a Michelin starred restaurant'.",
            ],
            "Was the kitchen Michelin-starred when the food burned?",
            "this is a Michelin starred restaurant",
            "Michelin-starred",
            "reported_utterance",
            "recounted_event",
            "same_event",
            "uncertain",
        ),
        (
            ["We talked about whether we'd ever go to Mars.", "I said I might try it someday."],
            "Did the guest actually travel to Mars?",
            "might try it someday",
            "traveled to Mars",
            "conditional_or_hypothetical",
            "recounted_event",
            "same_event",
            "uncertain",
        ),
    ],
)
def test_podcast_context_distinguishes_narrated_events_and_embedded_dialogue(
    source_units,
    question,
    answer_quote,
    claimed_answer,
    source_role,
    claim_role,
    alignment,
    expected,
):
    citation = {"first_unit": 0, "last_unit": 1, "text": " ".join(source_units)}
    saved = {
        "question": question,
        "status": "answered",
        "answer_quote": answer_quote,
        "citation": citation,
    }
    scope = {
        "question": question,
        "responsiveness": "answers_question",
        "answer_quote": answer_quote,
        "citation": citation,
        "scope": "actual_event_or_state",
        "narrative_role": source_role,
        "diagnostic_only": True,
        "production_approved": False,
    }
    observed = []

    def answer(prompt, payload, schema, tokens):
        observed.append(payload)
        assert "past firsthand experience" in prompt
        assert payload["source_units"] == [
            {"id": i, "text": text} for i, text in enumerate(source_units)
        ]
        return response(
            "equivalent",
            source_role=source_role,
            claimed_role=claim_role,
            event_link=alignment,
            unit=0,
        )

    result = compare_claim_answer(
        question=question,
        proposed_answer=claimed_answer,
        source_answer=saved,
        scope_review=scope,
        delivered_units=source_units,
        completion=answer,
    )
    assert len(observed) == 1
    assert result["relation"] == expected
    assert result["production_approved"] is False
    assert result["narrative_evidence"]["relation_before_guard"] == "equivalent"
    if alignment == "same_event":
        assert result["narrative_evidence"]["event_source_span"]["text"] == source_units[0]
    else:
        assert result["narrative_evidence"]["event_source_span"] is None


def test_narrative_comparator_cannot_override_blind_source_role():
    source = scope_review()
    source["narrative_role"] = "reported_utterance"
    result = compare(
        proposed_answer="Bobby Green",
        scope_review=source,
        completion=lambda *_: response(
            "equivalent", claimed_role="recounted_event", source_role="recounted_event"
        ),
    )
    assert result["relation"] == "uncertain"
    assert result["narrative_evidence"]["blind_source_role"] == "reported_utterance"
