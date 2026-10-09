"""Explicit questions are inventoried before judging delivered/excluded answers."""

from __future__ import annotations

import pytest

from clipper.editorial_question_state import audit_explicit_question_state


def response(status, first=-1, last=-1):
    return {"status": status, "first_unit": first, "last_unit": last}


def test_complete_cut_without_explicit_question_does_not_imply_approval():
    result = audit_explicit_question_state(
        ["No, it just drives the podcast."],
        ["It's like tools."],
        lambda *_: (_ for _ in ()).throw(AssertionError("no explicit question")),
    )
    assert result["question_inventory"] == []
    assert result["missing_answer_question_ids"] == []
    assert result["production_approved"] is False


def test_delivered_unanswered_question_with_excluded_answer_is_recorded():
    delivered = [
        "Do you get paid for views?",
        "It's all about sponsors.",
        "I got a lot of views.",
    ]
    excluded = ["I'm not making a dime off it."]
    calls = []

    def completion(prompt, payload, schema, tokens):
        calls.append(payload)
        assert "ANSWER" in prompt
        assert tokens == 64
        assert schema["first_unit"]["enum"] == (
            [-1, 1, 2] if payload["candidate_region"] == "delivered" else [-1, 0]
        )
        return (
            response("unanswered")
            if payload["candidate_region"] == "delivered"
            else response("answered", 0, 0)
        )

    result = audit_explicit_question_state(delivered, excluded, completion)
    assert result["missing_answer_question_ids"] == [0]
    assert result["question_inventory"][0]["excluded"]["answer_source"]["text"] == excluded[0]
    assert [call["candidate_region"] for call in calls] == ["delivered", "excluded"]
    assert all("candidate_units" in call for call in calls)
    assert result["production_approved"] is False


def test_delivered_answer_skips_excluded_model_call():
    calls = []

    def completion(_prompt, payload, _schema, _tokens):
        calls.append(payload["candidate_region"])
        return response("answered", 1, 1)

    result = audit_explicit_question_state(
        ["Where does it go?", "It drives the podcast."], ["Another topic."], completion
    )
    assert calls == ["delivered"]
    assert result["missing_answer_question_ids"] == []


def test_every_explicit_question_is_inventoried_and_uncertainty_is_not_approval():
    seen = []

    def completion(_prompt, payload, _schema, _tokens):
        key = (payload["question"]["id"], payload["candidate_region"])
        seen.append(key)
        if key == (0, "delivered"):
            return response("answered", 1, 1)
        if key == (2, "delivered"):
            return response("uncertain")
        raise AssertionError("only unanswered delivered questions need excluded inference")

    result = audit_explicit_question_state(
        ["What happened?", "He left.", "Why did he leave?", "It was late."],
        ["He had an appointment."],
        completion,
    )
    assert seen == [(0, "delivered"), (2, "delivered")]
    assert [item["question_source"]["unit_id"] for item in result["question_inventory"]] == [
        0,
        2,
    ]
    assert result["missing_answer_question_ids"] == []
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "bad,error",
    [
        (response("answered"), "needs a cited"),
        (response("unanswered", 1, 1), "cannot cite"),
        (response("answered", 0, 0), "needs a cited"),
        ({"status": "answered"}, "missing or extra"),
        (response("invented"), "invalid status"),
        (response("answered", True, 1), "invalid status"),
    ],
)
def test_invalid_delivered_answer_contract_fails_closed(bad, error):
    with pytest.raises(ValueError, match=error):
        audit_explicit_question_state(
            ["Where does it go?", "It drives the podcast."],
            ["Another topic."],
            lambda *_: bad,
        )


def test_question_state_requires_real_source_regions():
    with pytest.raises(ValueError, match="delivered and excluded"):
        audit_explicit_question_state([], ["After."], lambda *_: response("unanswered"))
