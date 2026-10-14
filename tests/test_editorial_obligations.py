"""Parser-owned obligations cannot be omitted or promoted to factual approval."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clipper.editorial_obligations import (
    build_syntax_obligations,
    validate_obligation_reviews,
)

HEADLINE = "Bobby was the speaker's opponent"
SOURCE = ["Bobby stood near the speaker.", "Nobody identified the opponent."]


def inventory():
    return {
        "headline": HEADLINE,
        "frames": [
            {
                "predicate_token": 1,
                "predicate": "was",
                "predicate_first_char": 6,
                "predicate_last_char": 9,
                "roles": [
                    {
                        "role": "subject",
                        "head_token": 0,
                        "first_char": 0,
                        "last_char": 5,
                        "text": "Bobby",
                    },
                    {
                        "role": "argument",
                        "head_token": 4,
                        "first_char": 10,
                        "last_char": len(HEADLINE),
                        "text": "the speaker's opponent",
                    },
                ],
            }
        ],
        "parse_warnings": [],
        "uncovered_content_tokens": [],
        "source_entailment_checked": False,
        "production_approved": False,
    }


def reviews(manifest):
    return [
        {
            "obligation_id": item["obligation_id"],
            "verdict": "supported",
            "first_unit": 0,
            "last_unit": 1,
        }
        for item in manifest["obligations"]
    ]


def test_parser_obligations_include_copular_relationship_role_and_cannot_approve():
    manifest = build_syntax_obligations(inventory())
    assert [(item["obligation_id"], item["text"]) for item in manifest["obligations"]] == [
        ("predicate:1", "was"),
        ("role:1:0", "Bobby"),
        ("role:1:4", "the speaker's opponent"),
    ]
    assert manifest["ready_for_source_review"] is True
    result = validate_obligation_reviews(manifest, reviews(manifest), SOURCE)
    assert result["all_recorded_obligations_labeled_supported"] is True
    assert result["claim_inventory_semantically_qualified"] is False
    assert result["source_entailment_qualified"] is False
    assert result["production_approved"] is False
    assert result["reviews"][0]["source_text"] == " ".join(SOURCE)


@pytest.mark.parametrize(
    "mutation", ["predicate", "role", "duplicate", "missing_frame", "bad_frame", "bad_role"]
)
def test_obligation_manifest_rejects_tampered_or_duplicate_parser_data(mutation):
    data = inventory()
    if mutation == "predicate":
        data["frames"][0]["predicate_first_char"] = 0
    elif mutation == "role":
        data["frames"][0]["roles"][1]["text"] = "fighter"
    elif mutation == "duplicate":
        data["frames"][0]["roles"].append(deepcopy(data["frames"][0]["roles"][0]))
    elif mutation == "missing_frame":
        data["frames"] = []
    elif mutation == "bad_frame":
        data["frames"][0] = "not a frame"
    else:
        data["frames"][0]["roles"][0] = "not a role"
    with pytest.raises(
        ValueError,
        match=r"differs|duplicate|parser inventory|invalid frame|invalid role",
    ):
        build_syntax_obligations(data)


def test_obligation_manifest_preserves_parser_abstention_signals():
    data = inventory()
    data["parse_warnings"] = [{"reason": "subject_not_explicit_in_clause"}]
    manifest = build_syntax_obligations(data)
    assert manifest["ready_for_source_review"] is False
    with pytest.raises(ValueError, match="warning-free"):
        validate_obligation_reviews(manifest, reviews(manifest), SOURCE)
    data = inventory()
    data["uncovered_content_tokens"] = [{"text": "opponent"}]
    assert build_syntax_obligations(data)["ready_for_source_review"] is False


@pytest.mark.parametrize(
    "mutation", ["omitted", "duplicate", "fabricated", "uncited", "range", "label"]
)
def test_obligation_review_rejects_missing_or_invalid_records(mutation):
    manifest = build_syntax_obligations(inventory())
    records = reviews(manifest)
    if mutation == "omitted":
        records.pop()
    elif mutation == "duplicate":
        records[-1]["obligation_id"] = records[0]["obligation_id"]
    elif mutation == "fabricated":
        records[-1]["obligation_id"] = "role:1:99"
    elif mutation == "uncited":
        records[-1]["first_unit"] = records[-1]["last_unit"] = -1
    elif mutation == "range":
        records[-1]["last_unit"] = 99
    else:
        records[-1]["verdict"] = "invented"
    with pytest.raises(ValueError, match=r"omitted or added|invalid ID"):
        validate_obligation_reviews(manifest, records, SOURCE)


def test_uncertain_obligation_may_abstain_without_a_citation():
    manifest = build_syntax_obligations(inventory())
    records = reviews(manifest)
    records[-1].update(verdict="uncertain", first_unit=-1, last_unit=-1)
    result = validate_obligation_reviews(manifest, records, SOURCE)
    assert result["all_recorded_obligations_labeled_supported"] is False
    assert result["reviews"][-1]["source_text"] == ""


def test_obligation_review_rejects_missing_speech_duplicate_manifest_and_extra_field():
    manifest = build_syntax_obligations(inventory())
    with pytest.raises(ValueError, match="delivered source speech"):
        validate_obligation_reviews(manifest, reviews(manifest), [])
    duplicate_manifest = deepcopy(manifest)
    duplicate_manifest["obligations"][-1]["obligation_id"] = duplicate_manifest["obligations"][0][
        "obligation_id"
    ]
    with pytest.raises(ValueError, match="repeats an ID"):
        validate_obligation_reviews(duplicate_manifest, reviews(duplicate_manifest), SOURCE)
    records = reviews(manifest)
    records[0]["extra"] = True
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_obligation_reviews(manifest, records, SOURCE)
