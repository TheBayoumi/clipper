"""A headline-only relation inventory binds words but never proves truth."""

from __future__ import annotations

import pytest

from clipper.editorial_claim_inventory import inventory_headline_relations

HEADLINE = "Bobby Green paced during the fighter meeting"


def relation(
    kind="actor_action",
    question="Who paced during the fighter meeting?",
    af=0,
    al=1,
    cf=0,
    cl=6,
):
    return {
        "kind": kind,
        "question": question,
        "answer_first_word": af,
        "answer_last_word": al,
        "claim_first_word": cf,
        "claim_last_word": cl,
    }


def test_inventory_sees_only_headline_and_binds_answer_to_exact_words():
    seen = []

    def completion(prompt, payload, schema, tokens):
        seen.append((prompt, payload, schema, tokens))
        return {"relations": [relation()]}

    result = inventory_headline_relations(HEADLINE, completion)
    assert set(seen[0][1]) == {"headline", "word_index"}
    assert seen[0][1]["word_index"][0] == {"id": 0, "text": "Bobby"}
    assert seen[0][3] == 416
    assert result["relations"][0]["answer"] == "Bobby Green"
    assert result["uncovered_headline_words"] == []
    assert result["inventory_semantically_qualified"] is False
    assert result["production_approved"] is False


def test_inventory_reports_omitted_headline_words_without_approving():
    result = inventory_headline_relations(HEADLINE, lambda *_: {"relations": [relation(cl=2)]})
    assert result["uncovered_headline_words"] == ["during", "the", "fighter", "meeting"]
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "bad,error",
    [
        ({}, "missing or unknown"),
        ({"relations": []}, "needs 1-10"),
        ({"relations": [relation(af=True)]}, "not bound"),
        ({"relations": [relation(af=3, al=4, cf=5)]}, "not bound"),
        ({"relations": [relation(question="Who paced")]}, "not bound"),
        ({"relations": [relation(), relation()]}, "repeats"),
        ({"relations": [{**relation(), "source": "invented"}]}, "missing or unknown"),
    ],
)
def test_invalid_inventory_fails_closed(bad, error):
    with pytest.raises(ValueError, match=error):
        inventory_headline_relations(HEADLINE, lambda *_: bad)


def test_inventory_rejects_invalid_headline():
    with pytest.raises(ValueError, match="4-14 word"):
        inventory_headline_relations("Bobby paced", lambda *_: {"relations": [relation()]})
