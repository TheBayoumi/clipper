"""Source-only proposition, question intent, and relation remain independent."""

from __future__ import annotations

import pytest

from clipper import editorial_propositions
from clipper.editorial_source_scope import assess_source_answer_scope

QUESTION = "What was the actual broadcast setting while Bobby paced?"
UNITS = [
    "Sean described the briefing years ago.",
    "The organizer said, 'We're on live TV, don't do this.' Bobby paced in the back.",
]


def answer(question=QUESTION, quote="We're on live TV", unit=1, units=UNITS):
    return {
        "question": question,
        "status": "answered",
        "answer_quote": quote,
        "citation": {"first_unit": unit, "last_unit": unit, "text": units[unit]},
    }


@pytest.fixture(autouse=True)
def isolate_source_extraction(monkeypatch):
    original = editorial_propositions.extract_source_proposition

    def source_only(**kwargs):
        quote = kwargs["answer_quote"].casefold()
        spoken = "live tv" in quote or "studio tonight" in quote
        negative = "don't make" in quote or "never earn" in quote
        return original(
            **{key: val for key, val in kwargs.items() if key != "completion"},
            completion=lambda *_: {
                "narrative_role": "reported_utterance" if spoken else "recounted_event",
                "attribution": "reported_speech" if spoken else "first_person_account",
                "timeframe": "past",
                "polarity": "negated" if negative else "affirmed",
                "modality": "asserted",
                "actor": "guest",
                "predicate": "account of the described event",
            },
        )

    monkeypatch.setattr(editorial_propositions, "extract_source_proposition", source_only)


def staged_completion(*, target="described_event_or_state", responsiveness="answers_question"):
    calls = []

    def fn(prompt, payload, schema, tokens):
        calls.append((payload, schema))
        if "question_target" in schema:
            assert payload.keys() == {"question"}
            return {"question_target": target}
        assert schema == {
            "responsiveness": {
                "type": "string",
                "enum": ["answers_question", "does_not_answer", "uncertain"],
            }
        }
        assert "source_proposition" in payload
        assert "proposed_answer" not in payload and "headline" not in payload
        return {"responsiveness": responsiveness}

    return fn, calls


def test_quoted_admonition_cannot_prove_actual_broadcast_even_if_model_approves():
    completion, calls = staged_completion(responsiveness="answers_question")
    scoped = assess_source_answer_scope(
        question=QUESTION,
        source_answer=answer(),
        delivered_units=UNITS,
        completion=completion,
    )
    assert len(calls) == 2
    assert scoped["responsiveness"] == "does_not_answer"
    assert scoped["scope"] == "quoted_instruction"
    assert scoped["narrative_role"] == "reported_utterance"
    assert scoped["source_proposition"]["attribution"] == "reported_speech"
    assert scoped["scope_correction"] == "reported_words_do_not_directly_establish_event"
    assert scoped["production_approved"] is False


def test_quoted_words_can_answer_a_question_about_what_was_said():
    question = "What did the organizer say?"
    completion, _ = staged_completion(target="reported_words_or_belief")
    scoped = assess_source_answer_scope(
        question=question,
        source_answer=answer(question=question),
        delivered_units=UNITS,
        completion=completion,
    )
    assert scoped["responsiveness"] == "answers_question"
    assert scoped["scope"] == "quoted_instruction"
    assert scoped["production_approved"] is False


def test_negative_firsthand_claim_is_responsive_to_income_questions():
    source = [
        "After 60 million views he asked me about income.",
        "You don't make anything off Instagram.",
    ]
    for question, target in (
        ("What does the speaker say Instagram earns?", "reported_words_or_belief"),
        ("Does the speaker earn money from those views?", "described_event_or_state"),
    ):
        completion, calls = staged_completion(target=target)
        result = assess_source_answer_scope(
            question=question,
            source_answer=answer(question=question, quote="You don't make anything", units=source),
            delivered_units=source,
            completion=completion,
        )
        assert len(calls) == 2
        assert result["responsiveness"] == "answers_question"
        assert result["scope"] == "negated_actual_state"
        assert result["source_proposition"]["polarity"] == "negated"
        assert result["production_approved"] is False


def test_model_cannot_upgrade_abstained_source_answer():
    result = assess_source_answer_scope(
        question=QUESTION,
        source_answer={"question": QUESTION, "status": "unknown"},
        delivered_units=UNITS,
        completion=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected model call")),
    )
    assert result["responsiveness"] == "uncertain"
    assert result["scope"] == "unknown"
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"question": "Different?"}, "does not match"),
        ({"answer_quote": "invented"}, "not exact delivered"),
        ({"citation": None}, "exact cited answer"),
        ({"citation": {"first_unit": 20, "last_unit": 20, "text": ""}}, "positions"),
        ({"citation": {"first_unit": 1, "last_unit": 1, "text": "fake"}}, "not exact"),
    ],
)
def test_tampered_source_never_reaches_models(mutation, error):
    with pytest.raises(ValueError, match=error):
        assess_source_answer_scope(
            question=QUESTION,
            source_answer={**answer(), **mutation},
            delivered_units=UNITS,
            completion=lambda *_: {},
        )


@pytest.mark.parametrize("target", ["future_guess", 123, None])
def test_question_only_classifier_rejects_invalid_type(target):
    calls = []

    def completion(prompt, payload, schema, tokens):
        calls.append(payload)
        return {"question_target": target}

    with pytest.raises(ValueError, match="question-only target classifier"):
        assess_source_answer_scope(
            question=QUESTION,
            source_answer=answer(),
            delivered_units=UNITS,
            completion=completion,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "judgement",
    [
        {},
        {"responsiveness": "somewhat"},
        {"responsiveness": "answers_question", "scope": "fake"},
    ],
)
def test_responsiveness_contract_is_not_the_source_role_schema(judgement):
    calls = []

    def completion(prompt, payload, schema, tokens):
        calls.append(payload)
        if "question_target" in schema:
            return {"question_target": "described_event_or_state"}
        return judgement

    with pytest.raises(ValueError, match="relation has invalid responsiveness"):
        assess_source_answer_scope(
            question=QUESTION,
            source_answer=answer(),
            delivered_units=UNITS,
            completion=completion,
        )
    assert len(calls) == 2


def test_unknown_question_intent_cannot_produce_positive_scope():
    completion, _ = staged_completion(target="unknown")
    result = assess_source_answer_scope(
        question=QUESTION,
        source_answer=answer(),
        delivered_units=UNITS,
        completion=completion,
    )
    assert result["responsiveness"] == "uncertain"
    assert result["scope_correction"] == "question_intent_unresolved"


def test_recounted_event_not_equated_with_quote_inside_story():
    source = [
        "Back when I was working at the station, the power went out during my shift.",
        "My boss yelled that we were broadcasting live, but the lights were out.",
    ]
    question = "When did the power go out?"
    completion, _ = staged_completion()
    result = assess_source_answer_scope(
        question=question,
        source_answer=answer(question=question, quote="power went out", unit=0, units=source),
        delivered_units=source,
        completion=completion,
    )
    assert result["narrative_role"] == "recounted_event"
    assert result["scope"] == "actual_event_or_state"
    assert result["responsiveness"] == "answers_question"
