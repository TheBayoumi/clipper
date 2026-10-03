import pytest

from clipper.editorial_headline import materialize_source_headline, propose_source_headline

UNITS = [
    "Because right now I got like 60 million views on my Instagram.",
    "I'm like not making a dime off of it.",
    "No, it just drives the podcast.",
]
SPANS = {
    "setup_quote": {"text": UNITS[0], "first_unit": 0, "last_unit": 0},
    "resolution_quote": {"text": UNITS[2], "first_unit": 2, "last_unit": 2},
}
PARTS = [
    {"unit_id": 0, "text": "60 million views on my Instagram"},
    {"unit_id": 2, "text": "No, it just drives the podcast."},
]


def test_materializes_business_contrast_from_exact_setup_and_payoff():
    result = materialize_source_headline(UNITS, SPANS, PARTS)
    assert result["headline"] == (
        "60 million views on my Instagram — No, it just drives the podcast."
    )
    assert result["source_bound"] is True
    assert result["contextually_verified"] is False
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "parts,error",
    [
        (
            [
                {"unit_id": 0, "text": "60 million views on my Instagram"},
                {"unit_id": 2, "text": "No, it makes millions from the podcast."},
            ],
            "verbatim",
        ),
        (
            [
                {"unit_id": 0, "text": "60 million views on my Instagram"},
                {"unit_id": 1, "text": "I'm like not making a dime off of it."},
            ],
            "outside",
        ),
        ([{"unit_id": True, "text": "60 million views on my Instagram"}, PARTS[1]], "outside"),
        (
            [
                {"unit_id": 0, "text": "60 million views on my Instagram"},
                {"unit_id": 2, "text": "it just drives the podcast."},
            ],
            "omits source scope",
        ),
    ],
)
def test_rejects_fabrication_wrong_role_or_dropped_scope(parts, error):
    with pytest.raises(ValueError, match=error):
        materialize_source_headline(UNITS, SPANS, parts)


def test_rejects_conditional_earnings_laundered_as_actual():
    units = [
        "If you had a podcast getting 60 million views, you'd be making millions of dollars.",
        "You got to transition to somewhere that's monetizable.",
    ]
    spans = {
        "setup_quote": {"text": units[0], "first_unit": 0, "last_unit": 0},
        "resolution_quote": {"text": units[1], "first_unit": 1, "last_unit": 1},
    }
    parts = [
        {"unit_id": 0, "text": "you'd be making millions of dollars"},
        {"unit_id": 1, "text": "transition to somewhere that's monetizable"},
    ]
    with pytest.raises(ValueError, match="omits source scope"):
        materialize_source_headline(units, spans, parts)


def test_literal_quote_cannot_self_certify_contextual_truth():
    units = [
        "We're on live TV, don't do this, Bobby Green was pacing in the back.",
        "Everybody was listening to Sean Shelby.",
    ]
    spans = {
        "setup_quote": {"text": units[0], "first_unit": 0, "last_unit": 0},
        "resolution_quote": {"text": units[1], "first_unit": 1, "last_unit": 1},
    }
    result = materialize_source_headline(
        units,
        spans,
        [
            {"unit_id": 0, "text": "We're on live TV, don't do this"},
            {"unit_id": 1, "text": "Everybody was listening to Sean Shelby."},
        ],
    )
    assert result["source_bound"] is True
    assert result["contextually_verified"] is False
    assert result["production_approved"] is False


def test_model_only_selects_excerpt_positions_and_cannot_author_headline():
    def completion(prompt, payload, schema, tokens):
        assert "verbatim" in prompt
        assert payload["delivered_units"][2]["text"] == UNITS[2]
        assert set(schema) == {"setup_quote", "resolution_quote"}
        assert tokens == 160
        return {"setup_quote": PARTS[0], "resolution_quote": PARTS[1]}

    result = propose_source_headline(UNITS, SPANS, completion)
    assert result["headline"].startswith("60 million views")
    assert result["production_approved"] is False


def test_model_cannot_approve_fabricated_or_extra_response_fields():
    def fabricated(*_):
        return {
            "setup_quote": PARTS[0],
            "resolution_quote": {"unit_id": 2, "text": "The podcast earned millions."},
        }

    with pytest.raises(ValueError, match="verbatim"):
        propose_source_headline(UNITS, SPANS, fabricated)

    def extra(*_):
        return {"setup_quote": PARTS[0], "resolution_quote": PARTS[1], "approved": True}

    with pytest.raises(ValueError, match="omitted a reviewed role"):
        propose_source_headline(UNITS, SPANS, extra)
