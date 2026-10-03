"""A cited claim record is auditable, but never a publication approval."""

from __future__ import annotations

from copy import deepcopy

import pytest

from clipper.editorial_review import (
    create_claim_review_packet,
    validate_claim_review,
    validate_claim_review_for_packet,
)

VIDEO = "_kDrxucOx9g"
SOURCE = "a" * 64
TRANSCRIPT = "b" * 64
UNITS = [
    "I had a podcast with Mighty Mouse.",
    "He says I spar too much.",
    "I am going to reassess in the future.",
]
HEADLINE = "The speaker plans to reassess training after Mighty Mouse's advice"


def assessment(status="future_plan", verdict="supported", first=0, last=2, reporting="no_report"):
    return {
        "verdict": verdict,
        "claimed_reporting_status": reporting,
        "source_reporting_status": reporting,
        "claimed_event_status": status,
        "source_event_status": status,
        "evidence": {"first_unit": first, "last_unit": last},
        "reason": "The original speech establishes this claim in context.",
    }


def record(kind="human"):
    return {
        "schema": "clipper-headline-claim-review-v1",
        "source_video_id": VIDEO,
        "source_sha256": SOURCE,
        "transcript_sha256": TRANSCRIPT,
        "headline": HEADLINE,
        "reviewer": {"kind": kind, "identifier": "reviewer-1"},
        "whole_headline": assessment(),
        "claims": [
            {
                "kind": "event_modality",
                "first_char": 0,
                "last_char": len(HEADLINE),
                "assessment": assessment(),
            },
            {
                "kind": "attribution",
                "first_char": HEADLINE.index("Mighty Mouse"),
                "last_char": len(HEADLINE),
                "assessment": assessment(
                    "actual_event", first=0, last=1, reporting="actual_report"
                ),
            },
        ],
    }


def validate(value, **changes):
    arguments = {
        "source_units": UNITS,
        "source_video_id": VIDEO,
        "source_sha256": SOURCE,
        "transcript_sha256": TRANSCRIPT,
        "expected_headline": value.get("headline", HEADLINE),
        **changes,
    }
    return validate_claim_review(value, **arguments)


def packet(spans=None, **changes):
    arguments = {
        "headline": HEADLINE,
        "selected_units": UNITS,
        "excluded_before": ["Earlier unrelated speech."],
        "excluded_after": ["Later unrelated speech."],
        "reviewed_spans": spans
        if spans is not None
        else {
            "setup_quote": {"text": "podcast with Mighty Mouse", "first_unit": 0, "last_unit": 0},
            "resolution_quote": {
                "text": "going to reassess in the future",
                "first_unit": 2,
                "last_unit": 2,
            },
        },
        "source_video_id": VIDEO,
        "source_sha256": SOURCE,
        "transcript_sha256": TRANSCRIPT,
        **changes,
    }
    return create_claim_review_packet(**arguments)


def test_packet_retains_delivered_source_and_marks_excluded_context_as_non_evidence():
    value = packet()
    assert value["selected_source_units"][2]["text"] == UNITS[2]
    assert value["excluded_after_context_only"] == ["Later unrelated speech."]
    assert value["required_review"]["atomic_claims"] == "unreviewed"
    assert value["production_approved"] is False


def test_packet_rejects_fabricated_or_mispositioned_setup_and_payoff():
    for span in (
        {"text": "Mighty Mouse advised more sparring", "first_unit": 0, "last_unit": 0},
        {"text": "podcast with Mighty Mouse", "first_unit": 1, "last_unit": 1},
    ):
        with pytest.raises(ValueError, match="absent"):
            packet(
                {
                    "setup_quote": span,
                    "resolution_quote": {
                        "text": "going to reassess in the future",
                        "first_unit": 2,
                        "last_unit": 2,
                    },
                }
            )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"selected_units": []}, "delivered speech"),
        ({"source_video_id": "wrong"}, "source identity"),
        ({"excluded_after": [""]}, "delivered speech"),
    ],
)
def test_packet_fails_closed_on_missing_source_or_context(changes, error):
    with pytest.raises(ValueError, match=error):
        packet(**changes)


@pytest.mark.parametrize(
    "spans,error",
    [
        ({"setup_quote": {"text": "x", "first_unit": 0, "last_unit": 0}}, "setup and resolution"),
        (
            {
                "setup_quote": {"text": "x", "first_unit": 0},
                "resolution_quote": {"text": "future", "first_unit": 2, "last_unit": 2},
            },
            "invalid reviewed",
        ),
        (
            {
                "setup_quote": {"text": "", "first_unit": 0, "last_unit": 0},
                "resolution_quote": {"text": "future", "first_unit": 2, "last_unit": 2},
            },
            "empty reviewed",
        ),
    ],
)
def test_packet_rejects_incomplete_reviewed_spans(spans, error):
    with pytest.raises(ValueError, match=error):
        packet(spans)


def test_review_handoff_binds_packet_to_independently_delivered_speech():
    result = validate_claim_review_for_packet(
        packet(),
        record("automated"),
        delivered_source_units=UNITS,
        source_video_id=VIDEO,
        source_sha256=SOURCE,
        transcript_sha256=TRANSCRIPT,
    )
    assert len(result["packet_sha256"]) == 64
    assert result["all_claims_supported"] is True
    assert result["review_status"] == "automated_unqualified"
    assert result["production_approved"] is False


@pytest.mark.parametrize(
    "change,error",
    [
        ("speech", "delivered speech differs"),
        ("headline", "differs from the draft"),
        ("source", "source identity"),
        ("span", "absent from cited"),
        ("approval", "cannot self-approve"),
        ("extra_field", "missing or unknown fields"),
        ("missing_delivered", "independently verified delivered speech"),
        ("missing_headline", "exact draft headline"),
        ("requirements", "invalid review requirement"),
        ("excluded", "invalid excluded context"),
        ("span_shape", "reviewed setup and resolution"),
        ("span_fields", "invalid reviewed source span"),
    ],
)
def test_review_handoff_rejects_mutated_packet_or_record(change, error):
    draft, review = packet(), record()
    delivered = UNITS
    if change == "speech":
        draft["selected_source_units"][1]["text"] = "A fabricated line."
    elif change == "headline":
        review["headline"] = "A different headline"
    elif change == "source":
        draft["source_sha256"] = "c" * 64
    elif change == "span":
        draft["reviewed_spans"]["resolution_quote"]["text"] = "A fabricated payoff"
    elif change == "extra_field":
        draft["unexpected"] = True
    elif change == "missing_delivered":
        delivered = []
    elif change == "missing_headline":
        draft["headline"] = ""
    elif change == "requirements":
        draft["required_review"]["atomic_claims"] = "approved"
    elif change == "excluded":
        draft["excluded_before_context_only"] = [""]
    elif change == "span_shape":
        draft["reviewed_spans"].pop("resolution_quote")
    elif change == "span_fields":
        draft["reviewed_spans"]["setup_quote"]["unexpected"] = True
    else:
        draft["production_approved"] = True
    with pytest.raises(ValueError, match=error):
        validate_claim_review_for_packet(
            draft,
            review,
            delivered_source_units=delivered,
            source_video_id=VIDEO,
            source_sha256=SOURCE,
            transcript_sha256=TRANSCRIPT,
        )


def test_human_attestation_cites_python_owned_context_without_publication_approval():
    result = validate(record())
    assert result["all_claims_supported"] is True
    assert result["review_status"] == "human_attested_not_publication_approved"
    assert result["production_approved"] is False
    assert result["claims"][1]["assessment"]["source_text"] == " ".join(UNITS[:2])
    assert result["claims"][1]["assessment"]["context"]["text"] == " ".join(UNITS)


def test_automated_labels_cannot_attest_even_when_every_claim_says_supported():
    result = validate(record("automated"))
    assert result["all_claims_supported"] is True
    assert result["review_status"] == "automated_unqualified"
    assert result["production_approved"] is False


def test_modality_conflict_cannot_be_marked_supported():
    value = record()
    value["whole_headline"]["source_event_status"] = "actual_event"
    with pytest.raises(ValueError, match="mismatched"):
        validate(value)


def test_reporting_act_and_embedded_possibility_are_independent():
    value = record()
    value["headline"] = "The speaker says a great fighter can be cut"
    value["whole_headline"] = assessment("hypothetical", reporting="actual_report")
    value["claims"] = [
        {
            "kind": "event_modality",
            "first_char": 0,
            "last_char": len(value["headline"]),
            "assessment": assessment("hypothetical", reporting="actual_report"),
        },
        {
            "kind": "attribution",
            "first_char": 4,
            "last_char": 16,
            "assessment": assessment("actual_event", reporting="actual_report"),
        },
    ]
    assert validate(value)["all_claims_supported"]
    value["whole_headline"]["source_reporting_status"] = "no_report"
    with pytest.raises(ValueError, match="reporting"):
        validate(value)


def test_reported_state_has_no_embedded_event_but_requires_real_reporting_act():
    value = record()
    value["headline"] = "The speaker calls UFC fighters independent contractors"
    value["whole_headline"] = assessment(
        "not_applicable", first=0, last=0, reporting="actual_report"
    )
    value["claims"] = [
        {
            "kind": "attribution",
            "first_char": 0,
            "last_char": len(value["headline"]),
            "assessment": assessment("not_applicable", first=0, last=0, reporting="actual_report"),
        }
    ]
    source_units = ["You guys are independent contractors."]
    assert validate(value, source_units=source_units)["all_claims_supported"]
    value["whole_headline"]["source_reporting_status"] = "no_report"
    with pytest.raises(ValueError, match="mismatched"):
        validate(value, source_units=source_units)
    value["whole_headline"]["source_reporting_status"] = "actual_report"
    value["whole_headline"]["claimed_reporting_status"] = "no_report"
    with pytest.raises(ValueError, match="mismatched"):
        validate(value, source_units=source_units)


def test_missing_attribution_or_condition_words_prevents_attestation():
    value = record()
    value["claims"] = [
        {
            "kind": "event_modality",
            "first_char": 0,
            "last_char": HEADLINE.index(" after"),
            "assessment": assessment(),
        }
    ]
    result = validate(value)
    assert result["review_status"] == "human_review_incomplete"
    assert "Mighty" in result["uncovered_headline_words"]


def test_source_identity_and_evidence_positions_are_not_model_editable():
    for change, error in (
        ({"source_sha256": "c" * 64}, "identity"),
        ({"transcript_sha256": "c" * 64}, "identity"),
    ):
        value = {**record(), **change}
        with pytest.raises(ValueError, match=error):
            validate(value)
    value = deepcopy(record())
    value["claims"][0]["assessment"]["evidence"]["first_unit"] = True
    with pytest.raises(ValueError, match="range"):
        validate(value)
    value = deepcopy(record())
    value["claims"][0]["assessment"]["evidence"]["last_unit"] = 99
    with pytest.raises(ValueError, match="range"):
        validate(value)


def test_attestation_cannot_approve_a_different_rendered_headline():
    value = record()
    value["headline"] = "A safer but different headline"
    with pytest.raises(ValueError, match="differs from the draft"):
        validate(value, expected_headline=HEADLINE)


def test_whole_headline_veto_and_claim_veto_both_prevent_attestation():
    value = record()
    value["whole_headline"]["verdict"] = "uncertain"
    assert not validate(value)["all_claims_supported"]
    value = record()
    value["claims"][1]["assessment"]["verdict"] = "unsupported"
    assert not validate(value)["all_claims_supported"]


def test_review_record_requires_authoritative_source_and_complete_shape():
    with pytest.raises(ValueError, match="source provenance"):
        validate(record(), source_units=[])
    value = record()
    value["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        validate(value)
    value = record()
    value["headline"] = ""
    with pytest.raises(ValueError, match="factual headline"):
        validate(value)
    value = record()
    value["reviewer"]["identifier"] = ""
    with pytest.raises(ValueError, match="identified reviewer"):
        validate(value)
    value = record()
    value["whole_headline"].pop("reason")
    with pytest.raises(ValueError, match="assertion contract"):
        validate(value)
    value = record()
    value["whole_headline"]["verdict"] = "approved"
    with pytest.raises(ValueError, match="assertion labels"):
        validate(value)
    value = record()
    value["claims"] = []
    with pytest.raises(ValueError, match="atomic claim"):
        validate(value)
    value = record()
    value["claims"][0]["extra"] = True
    with pytest.raises(ValueError, match="invalid fields"):
        validate(value)
    value = record()
    value["claims"][0]["last_char"] = len(HEADLINE) + 1
    with pytest.raises(ValueError, match="headline words"):
        validate(value)


def test_unfocused_source_citation_is_not_a_valid_claim_proof():
    value = record()
    value["whole_headline"]["evidence"] = {"first_unit": 0, "last_unit": 12}
    with pytest.raises(ValueError, match="unfocused"):
        validate(value, source_units=UNITS + ["Additional speech."] * 10)


def test_source_range_requires_exact_position_fields():
    value = record()
    value["claims"][0]["assessment"]["evidence"]["extra"] = "model copied quote"
    with pytest.raises(ValueError, match="exact source-unit range"):
        validate(value)
