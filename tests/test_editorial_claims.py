"""Contract checks for the experimental claim-level factual audit."""

from __future__ import annotations

import pytest

from clipper.editorial_claims import audit_headline_claims


class Reviewer:
    def __init__(self, claims: list[dict], judgments: list[dict]) -> None:
        self.responses = iter([{"claims": claims}, *judgments])
        self.payloads: list[dict] = []

    def _review_completion(self, prompt, payload, properties, tokens):
        self.payloads.append(payload)
        return next(self.responses)


def judgment(verdict="supported", scope="actual", first=0, last=0):
    return {
        "verdict": verdict,
        "source_scope": scope,
        "first_unit": first,
        "last_unit": last,
        "reason": "The cited speech was interpreted in context.",
    }


def test_atomic_claims_are_audited_separately_and_all_must_pass():
    headline = "Host earned millions from the podcast"
    reviewer = Reviewer(
        [
            {"text": "Host earned millions", "claim_scope": "actual"},
            {"text": "from the podcast", "claim_scope": "actual"},
        ],
        [judgment(), judgment("unsupported", "conditional"), judgment()],
    )
    result = audit_headline_claims(reviewer, headline, ["A podcast could make millions."])
    assert result["verdict"] == "unsupported"
    assert [item["verdict"] for item in result["claims"]] == [
        "supported",
        "unsupported",
        "supported",
    ]
    assert reviewer.payloads[0] == {"headline": headline}
    assert reviewer.payloads[1]["source_units"] == [
        {"id": 0, "text": "A podcast could make millions."}
    ]


def test_omitted_subclaim_words_remain_in_mandatory_whole_headline_check():
    reviewer = Reviewer(
        [{"text": "Host", "claim_scope": "actual"}],
        [judgment("unsupported"), judgment()],
    )
    result = audit_headline_claims(reviewer, "Host earned millions", ["Host spoke."])
    assert result["verdict"] == "unsupported"
    assert result["claims"][0]["claim"] == "Host earned millions"
    assert result["subclaim_word_coverage"] < 1


def test_supported_claim_requires_valid_source_range_and_known_scope():
    claims = [{"text": "Host earned millions", "claim_scope": "actual"}]
    for bad in (judgment(first=-1, last=-1), judgment(scope="unknown"), judgment(first=9)):
        with pytest.raises(RuntimeError, match=r"evidence|scope"):
            audit_headline_claims(Reviewer(claims, [bad]), "Host earned millions", ["Maybe."])


def test_uncertain_claim_prevents_approval():
    claims = [{"text": "Host earned millions", "claim_scope": "actual"}]
    result = audit_headline_claims(
        Reviewer(claims, [judgment("uncertain", "unknown", -1, -1)]),
        "Host earned millions",
        ["Maybe."],
    )
    assert result["verdict"] == "uncertain"


def test_quoted_instruction_cannot_support_actual_event():
    claims = [{"text": "on live TV", "claim_scope": "actual"}]
    result = audit_headline_claims(
        Reviewer(claims, [judgment(), judgment("supported", "quoted_instruction")]),
        "The meeting was on live TV",
        ["They told us, 'We are on live TV.'"],
    )
    assert result["verdict"] == "uncertain"


def test_whole_headline_yes_cannot_override_incomplete_claim_coverage():
    reviewer = Reviewer(
        [{"text": "Host", "claim_scope": "actual"}],
        [judgment(), judgment()],
    )
    result = audit_headline_claims(reviewer, "Host earned millions", ["Host spoke."])
    assert result["verdict"] == "uncertain"
    assert result["claim_text_coverage_complete"] is False
    assert [word["text"] for word in result["uncovered_headline_words"]] == ["earned", "millions"]
    assert result["production_approved"] is False


def test_fragment_judgment_retains_reporting_and_negation_context():
    headline = "Guest denies that the host earned millions"
    reviewer = Reviewer(
        [
            {"text": "Guest denies that", "claim_scope": "actual"},
            {"text": "the host earned millions", "claim_scope": "reported"},
        ],
        [judgment(), judgment(), judgment(scope="reported")],
    )
    result = audit_headline_claims(reviewer, headline, ["I deny that the host earned millions."])
    for payload in reviewer.payloads[1:]:
        assert payload["headline_context"] == headline
        span = payload["headline_span"]
        assert headline[span["first_char"] : span["last_char"]] == payload["claim"]
    assert result["claim_text_coverage_complete"] is True
    assert result["claim_inventory_semantically_qualified"] is False


def test_repeated_fragment_cannot_silently_bind_to_first_occurrence():
    reviewer = Reviewer([{"text": "Host", "claim_scope": "actual"}], [])
    with pytest.raises(RuntimeError, match="ambiguous repeated"):
        audit_headline_claims(reviewer, "Host disputes what Host said", ["A disputed statement."])
    assert len(reviewer.payloads) == 1
