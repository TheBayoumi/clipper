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
            {"text": "Host earned millions", "start": 0, "end": 20, "claim_scope": "actual"},
            {"text": "from the podcast", "start": 21, "end": 37, "claim_scope": "actual"},
        ],
        [judgment("unsupported", "conditional"), judgment()],
    )
    result = audit_headline_claims(reviewer, headline, ["A podcast could make millions."])
    assert result["verdict"] == "unsupported"
    assert [item["verdict"] for item in result["claims"]] == ["unsupported", "supported"]
    assert reviewer.payloads[0] == {"headline": headline}
    assert reviewer.payloads[1]["source_units"] == [
        {"id": 0, "text": "A podcast could make millions."}
    ]


def test_omitted_headline_words_fail_closed():
    reviewer = Reviewer([{"text": "Host", "start": 0, "end": 4, "claim_scope": "actual"}], [])
    with pytest.raises(RuntimeError, match="omitted headline words"):
        audit_headline_claims(reviewer, "Host earned millions", ["Host spoke."])


def test_supported_claim_requires_valid_source_range_and_known_scope():
    claims = [{"text": "Host earned millions", "start": 0, "end": 20, "claim_scope": "actual"}]
    for bad in (judgment(first=-1, last=-1), judgment(scope="unknown"), judgment(first=9)):
        with pytest.raises(RuntimeError, match=r"evidence|scope"):
            audit_headline_claims(Reviewer(claims, [bad]), "Host earned millions", ["Maybe."])


def test_uncertain_claim_prevents_approval():
    claims = [{"text": "Host earned millions", "start": 0, "end": 20, "claim_scope": "actual"}]
    result = audit_headline_claims(
        Reviewer(claims, [judgment("uncertain", "unknown", -1, -1)]),
        "Host earned millions",
        ["Maybe."],
    )
    assert result["verdict"] == "uncertain"


def test_quoted_instruction_cannot_support_actual_event():
    claims = [
        {"text": "The meeting was on live TV", "start": 0, "end": 26, "claim_scope": "actual"}
    ]
    result = audit_headline_claims(
        Reviewer(claims, [judgment("supported", "quoted_instruction")]),
        "The meeting was on live TV",
        ["They told us, 'We are on live TV.'"],
    )
    assert result["verdict"] == "uncertain"
