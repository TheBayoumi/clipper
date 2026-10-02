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
