"""A missing payoff needs an auditable delivered obligation and excluded answer."""

from __future__ import annotations

import pytest

from clipper.editorial_boundary import propose_cut_obligation, validate_cut_obligation


def pointer(first=-1, last=-1):
    return {"first_unit": first, "last_unit": last}


def proposal(kind, pending=None, fulfillment=None):
    return {
        "kind": kind,
        "pending": pending or pointer(),
        "fulfillment": fulfillment or pointer(),
        "reason": "The source determines whether the cut is complete.",
    }


def test_complete_business_cut_does_not_need_excluded_tools_list():
    delivered = [
        "Because right now I got like 60 million views on my Instagram.",
        "I'm like not making a dime off of it.",
        "No, it just drives the podcast.",
        "Yeah.",
    ]
    result = validate_cut_obligation(
        delivered,
        ["It's like tools.", "Instagram, Facebook, TikTok."],
        2,
        proposal("none"),
    )
    assert result["cut_complete"] is True
    assert result["pending_source"] is None
    assert result["production_approved"] is False


def test_missing_contrast_needs_excluded_contrast_words():
    delivered = ["Because right now I got 60 million views.", "No, it drives the podcast."]
    after = ["It's like tools.", "Instagram, Facebook, TikTok."]
    with pytest.raises(ValueError, match="contrastive excluded speech"):
        validate_cut_obligation(
            delivered,
            after,
            1,
            proposal("contrast", pointer(1, 1), pointer(0, 1)),
        )


def test_excluded_nonpayment_is_a_cited_missing_contrast():
    delivered = ["Because right now I got 60 million views.", "My thing is going crazy."]
    after = ["I'm not making a dime off of it."]
    result = validate_cut_obligation(
        delivered,
        after,
        1,
        proposal("contrast", pointer(1, 1), pointer(0, 0)),
    )
    assert result["cut_complete"] is False
    assert result["fulfillment_source"]["text"] == after[0]


def test_unfinished_reporting_clause_needs_its_excluded_content():
    delivered = ["I have a video from 2019 where I'm like telling him."]
    after = ["You can see me on the Joe Rogan podcast."]
    result = validate_cut_obligation(
        delivered,
        after,
        0,
        proposal("clause", pointer(0, 0), pointer(0, 0)),
    )
    assert result["cut_complete"] is False
    with pytest.raises(ValueError, match="unfinished clause"):
        validate_cut_obligation(delivered, after, 0, proposal("none"))


def test_unfinished_question_requires_question_mark():
    delivered = ["How do you get paid?", "I get views on Instagram."]
    after = ["I don't make a dime from it."]
    result = validate_cut_obligation(
        delivered, after, 1, proposal("question", pointer(0, 0), pointer(0, 0))
    )
    assert result["pending_source"]["text"] == delivered[0]
    with pytest.raises(ValueError, match="explicit delivered question"):
        validate_cut_obligation(
            delivered, after, 1, proposal("question", pointer(1, 1), pointer(0, 0))
        )


@pytest.mark.parametrize(
    "bad,error",
    [
        (None, "missing or extra fields"),
        ({"kind": "none"}, "missing or extra fields"),
        (proposal("invented"), "kind or reason"),
        (proposal("clause", pointer(0, 0), pointer()), "both delivered and excluded"),
        (proposal("none", pointer(0, 0), pointer()), "cannot cite"),
        (proposal("clause", pointer(1, 1), pointer(0, 0)), "dangling delivered clause"),
        (proposal("contrast", pointer(0, 0), pointer(0, 0)), "final delivered point"),
        (proposal("contrast", {"first_unit": True, "last_unit": 1}, pointer(0, 0)), "integers"),
        (proposal("contrast", pointer(1, 2), pointer(0, 0)), "positions are invalid"),
        (proposal("contrast", {"first_unit": 1}, pointer(0, 0)), "exact source-unit"),
    ],
)
def test_invalid_obligations_fail_closed(bad, error):
    with pytest.raises(ValueError, match=error):
        validate_cut_obligation(
            ["This point is complete.", "Another point."], ["But not this."], 1, bad
        )


def test_model_cannot_use_uncertain_as_an_unsupported_complete_verdict():
    delivered = ["No, it just drives the podcast."]
    after = ["It's like tools."]

    def completion(prompt, payload, schema, tokens):
        assert "Topic similarity" in prompt
        assert payload["final_substantive_unit_id"] == 0
        assert schema["fulfillment"]["properties"]["first_unit"]["enum"] == [-1, 0]
        assert schema["kind"]["enum"] == ["none", "question", "contrast"]
        assert tokens == 192
        return proposal("uncertain")

    with pytest.raises(ValueError, match="outside the constrained contract"):
        propose_cut_obligation(delivered, after, 0, completion)
    result = validate_cut_obligation(delivered, after, 0, proposal("uncertain"))
    assert result["cut_complete"] is False
    assert result["production_approved"] is False


def test_dangling_final_clause_is_a_deterministic_veto_without_model_call():
    delivered = ["I have a video where I'm like telling him."]
    after = ["Like, hey, you can see me on the podcast."]

    def forbidden(*_):
        raise AssertionError("model inference was unnecessary")

    result = propose_cut_obligation(delivered, after, 0, forbidden)
    assert result["kind"] == "clause"
    assert result["pending_source"]["text"] == delivered[0]
    assert result["fulfillment_source"] is None
    assert result["decision_origin"] == "python_syntax_v1"
    assert result["cut_complete"] is False
    assert result["production_approved"] is False


def test_cut_obligation_rejects_missing_source_context():
    with pytest.raises(ValueError, match="exact delivered and excluded speech"):
        validate_cut_obligation([], ["A continuation."], 0, proposal("none"))
    with pytest.raises(ValueError, match="exact delivered and excluded speech"):
        propose_cut_obligation([], ["A continuation."], 0, lambda *_: proposal("none"))


def test_constrained_model_can_choose_none_without_becoming_production_approval():
    result = propose_cut_obligation(
        ["No, it just drives the podcast."],
        ["It's like tools."],
        0,
        lambda *_: proposal("none"),
    )
    assert result["cut_complete"] is True
    assert result["production_approved"] is False
