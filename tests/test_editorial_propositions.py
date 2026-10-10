"""Source-only proposition extraction and immutable narrative provenance."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clipper.editorial_propositions import (
    extract_source_proposition,
    validate_source_proposition,
)


@pytest.mark.parametrize(
    "source,quote,role,attribution,polarity,modality",
    [
        (
            [
                "A guest remembered a tour from years ago.",
                "The manager yelled, 'We're on live TV' during the rehearsal.",
            ],
            "We're on live TV",
            "reported_utterance",
            "reported_speech",
            "affirmed",
            "asserted",
        ),
        (
            ["I had 60 million views on my account.", "You don't make anything off Instagram."],
            "You don't make anything off Instagram.",
            "present_event_or_state",
            "first_person_account",
            "negated",
            "asserted",
        ),
        (
            [
                "I started working the night shift in 2016.",
                "The hospital lost electricity that night.",
            ],
            "hospital lost electricity",
            "recounted_event",
            "first_person_account",
            "affirmed",
            "asserted",
        ),
        (
            ["If the channel reached 60 million views, I could make millions."],
            "I could make millions",
            "conditional_or_hypothetical",
            "general_claim",
            "affirmed",
            "conditional",
        ),
    ],
)
def test_extraction_uses_exact_source_only_with_source_identity(
    source,
    quote,
    role,
    attribution,
    polarity,
    modality,
):
    cited_id = 1 if len(source) > 1 else 0
    citation = {"first_unit": cited_id, "last_unit": cited_id, "text": source[cited_id]}
    if quote not in citation["text"]:
        citation = {"first_unit": 0, "last_unit": 0, "text": source[0]}
    payloads = []

    def completion(prompt, payload, schema, tokens):
        payloads.append(payload)
        assert "question" not in payload and "proposed_answer" not in payload
        assert "headline" not in payload and "source_units" in payload
        assert "scope" not in schema and "narrative_role" in schema
        return {
            "narrative_role": role,
            "attribution": attribution,
            "timeframe": "past",
            "polarity": polarity,
            "modality": modality,
            "actor": "guest",
            "predicate": "describes an experience",
        }

    evidence = extract_source_proposition(
        answer_quote=quote,
        citation=citation,
        delivered_units=source,
        completion=completion,
    )
    validate_source_proposition(
        evidence,
        answer_quote=quote,
        citation=citation,
        delivered_units=source,
    )
    assert len(payloads) == 1
    assert evidence["schema"] == "source_proposition_v1"
    assert len(evidence["anchor_id"]) == 64
    assert evidence["citation"] == citation
    assert evidence["production_approved"] is False
    assert evidence["diagnostic_only"] is True


def test_proposition_identity_is_bound_to_quote_positions_and_source_bytes():
    source = ["A guest described a festival.", "I lost my phone near the stage."]
    quote = "lost my phone"
    citation = {"first_unit": 1, "last_unit": 1, "text": source[1]}
    original = extract_source_proposition(
        answer_quote=quote,
        citation=citation,
        delivered_units=source,
        completion=lambda *_: {
            "narrative_role": "recounted_event",
            "attribution": "first_person_account",
            "timeframe": "past",
            "polarity": "affirmed",
            "modality": "asserted",
            "actor": "guest",
            "predicate": "lost phone",
        },
    )
    for field, replacement in (
        ("anchor_id", "bad"),
        ("answer_quote", "festival"),
        ("citation", {"first_unit": 0, "last_unit": 0, "text": source[0]}),
        ("schema", "source_proposition_v99"),
        ("production_approved", True),
    ):
        tampered = deepcopy(original)
        tampered[field] = replacement
        with pytest.raises(ValueError, match="identity or schema"):
            validate_source_proposition(
                tampered,
                answer_quote=quote,
                citation=citation,
                delivered_units=source,
            )
    with pytest.raises(ValueError, match="source citation"):
        validate_source_proposition(
            original,
            answer_quote=quote,
            citation=citation,
            delivered_units=["altered", "altered"],
        )


def test_source_model_cannot_claim_reported_speech_is_direct_event():
    source = ["The editor said, 'We filmed this live.'"]
    with pytest.raises(ValueError, match="attribution conflicts"):
        extract_source_proposition(
            answer_quote="We filmed this live",
            delivered_units=source,
            citation={"first_unit": 0, "last_unit": 0, "text": source[0]},
            completion=lambda *_: {
                "narrative_role": "recounted_event",
                "attribution": "reported_speech",
                "timeframe": "past",
                "polarity": "affirmed",
                "modality": "asserted",
                "actor": "editor",
                "predicate": "claimed to film live",
            },
        )


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {
            "narrative_role": "actual_event_or_state",
            "attribution": "unknown",
            "timeframe": "past",
            "polarity": "affirmed",
            "modality": "asserted",
            "actor": "guest",
            "predicate": "spoke",
        },
    ],
)
def test_invalid_extraction_fails_closed(bad):
    speech = ["I remember going to school."]
    with pytest.raises(ValueError, match="source proposition"):
        extract_source_proposition(
            answer_quote="going to school",
            citation={"first_unit": 0, "last_unit": 0, "text": speech[0]},
            delivered_units=speech,
            completion=lambda *_: bad,
        )
