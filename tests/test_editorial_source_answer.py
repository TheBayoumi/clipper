"""Blind source answers must bind to exact delivered speech and never approve."""

from __future__ import annotations

import pytest

from clipper.editorial_source_answer import answer_source_question


def result(status="answered", quote="Bobby Green was pacing", first=0, last=0):
    return {
        "status": status,
        "answer_quote": quote,
        "first_unit": first,
        "last_unit": last,
    }


def test_source_question_is_blind_to_headline_answer_and_binds_exact_quote():
    seen = []

    def completion(prompt, payload, schema, tokens):
        seen.append(payload)
        assert "people merely mentioned" in prompt
        assert schema["first_unit"]["enum"] == [-1, 0, 1]
        assert tokens == 112
        return result()

    answer = answer_source_question(
        "Who paced at the meeting?",
        ["Bobby Green was pacing in the back.", "Others listened to Sean Shelby."],
        completion,
    )
    assert set(seen[0]) == {"question", "source_units"}
    assert answer["answer_quote"] == "Bobby Green was pacing"
    assert answer["citation"]["text"] == "Bobby Green was pacing in the back."
    assert answer["answer_semantically_verified"] is False
    assert answer["production_approved"] is False


def test_unknown_and_uncertain_source_answers_abstain_without_citation():
    for status in ("unknown", "uncertain"):
        answer = answer_source_question(
            "Who was the opponent?",
            ["Bobby was in the back."],
            lambda *_, status=status: result(status, "", -1, -1),
        )
        assert answer["status"] == status
        assert answer["citation"] is None
        assert answer["production_approved"] is False


@pytest.mark.parametrize(
    "bad,error",
    [
        (None, "missing or unknown"),
        ({"status": "answered"}, "missing or unknown"),
        (result("invented"), "invalid labels"),
        (result(first=True), "invalid labels"),
        (result(first=1, last=0), "focused source range"),
        (result(quote="Sean Shelby paced"), "not exact cited speech"),
        (result("unknown", "Bobby Green", -1, -1), "cannot cite"),
        (result("uncertain", "", 0, 0), "cannot cite"),
    ],
)
def test_invalid_source_answers_fail_closed(bad, error):
    with pytest.raises(ValueError, match=error):
        answer_source_question("Who paced?", ["Bobby Green was pacing."], lambda *_: bad)


def test_source_answer_requires_question_and_source():
    with pytest.raises(ValueError, match="requires a question"):
        answer_source_question("Who paced", ["Bobby paced."], lambda *_: result())
    with pytest.raises(ValueError, match="delivered speech"):
        answer_source_question("Who paced?", [], lambda *_: result())
