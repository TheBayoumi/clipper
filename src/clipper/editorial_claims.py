"""Experimental, source-positioned headline audit; not a publication gate.

The model proposes claims and judgments. Python owns coverage, evidence identity,
and fail-closed verdict aggregation. A valid pointer proves provenance, not truth.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

_WORD = re.compile(r"\w+", re.UNICODE)
_STATES = ("supported", "unsupported", "uncertain")
_SCOPES = ("actual", "reported", "conditional", "quoted_instruction", "unknown")


class ClaimReviewer(Protocol):
    def _review_completion(
        self, prompt: str, payload: dict[str, Any], properties: dict[str, Any], tokens: int
    ) -> dict[str, Any]: ...


def audit_headline_claims(
    reviewer: ClaimReviewer, headline: str, source_units: list[str]
) -> dict[str, Any]:
    """Decompose first, then assess each claim against original delivered speech.

    This is intentionally separate from the current production factual gate until
    real and held-out controls demonstrate adequate recall and precision.
    """
    if not headline.strip() or not source_units or any(not unit.strip() for unit in source_units):
        raise ValueError("claim audit requires a headline and nonempty source units")
    claims_result = reviewer._review_completion(
        "Split the headline into its smallest independently checkable factual claims. "
        "Copy each claim as an exact contiguous substring of the headline and give its "
        "zero-based start and exclusive end character offsets. Include attribution, "
        "actual-versus-conditional outcomes, roles, quantities, and settings as claims. "
        "Classify the scope each claim asserts: actual, reported, conditional, "
        "quoted_instruction, or unknown. "
        "Do not inspect source speech in this step. Return output_schema JSON.",
        {"headline": headline},
        {
            "claims": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "claim_scope": {"type": "string", "enum": list(_SCOPES)},
                    },
                    "required": ["text", "start", "end", "claim_scope"],
                    "additionalProperties": False,
                },
            }
        },
        384,
    )["claims"]
    if not isinstance(claims_result, list) or not 1 <= len(claims_result) <= 8:
        raise RuntimeError("claim extraction returned an invalid claim count")
    coverage = [False] * len(headline)
    claims: list[str] = []
    for item in claims_result:
        if not isinstance(item, dict) or set(item) != {"text", "start", "end", "claim_scope"}:
            raise RuntimeError("claim extraction returned an invalid span")
        start, end, value = item["start"], item["end"], item["text"]
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(value, str)
            or not 0 <= start < end <= len(headline)
            or headline[start:end] != value
            or not _WORD.search(value)
            or item["claim_scope"] not in _SCOPES
        ):
            raise RuntimeError("claim extraction did not copy a headline span")
        for index in range(start, end):
            coverage[index] = True
        claims.append(value)
    if any(not all(coverage[match.start() : match.end()]) for match in _WORD.finditer(headline)):
        raise RuntimeError("claim extraction omitted headline words")

    reviews = []
    numbered_source = [{"id": index, "text": unit} for index, unit in enumerate(source_units)]
    for claim, extracted in zip(claims, claims_result, strict=True):
        judgment = reviewer._review_completion(
            "Assess this one headline claim against the delivered source speech. "
            "The cited source range is evidence to interpret, not proof of entailment. "
            "Distinguish what happened from a quoted instruction, reported assertion, "
            "hypothetical possibility, or negated outcome. A source mention of a person "
            "does not establish the relationship assigned in the claim. Mark supported "
            "only when the complete claim, including scope and attribution, is entailed. "
            "Mark uncertain when speech is ambiguous or incomplete. Identify the smallest "
            "contiguous source-unit range necessary; use -1/-1 if none exists. "
            "Return output_schema JSON.",
            {
                "claim": claim,
                "claimed_scope": extracted["claim_scope"],
                "source_units": numbered_source,
            },
            {
                "verdict": {"type": "string", "enum": list(_STATES)},
                "source_scope": {"type": "string", "enum": list(_SCOPES)},
                "first_unit": {"type": "integer", "minimum": -1},
                "last_unit": {"type": "integer", "minimum": -1},
                "reason": {"type": "string", "minLength": 1},
            },
            256,
        )
        if set(judgment) != {"verdict", "source_scope", "first_unit", "last_unit", "reason"}:
            raise RuntimeError("claim judgment returned an invalid contract")
        verdict, scope = judgment["verdict"], judgment["source_scope"]
        first, last = judgment["first_unit"], judgment["last_unit"]
        if (
            verdict not in _STATES
            or scope not in _SCOPES
            or type(first) is not int
            or type(last) is not int
            or not isinstance(judgment["reason"], str)
            or not judgment["reason"].strip()
            or not (first == last == -1 or 0 <= first <= last < len(source_units))
        ):
            raise RuntimeError("claim judgment has invalid evidence or labels")
        if verdict == "supported" and (first == -1 or scope == "unknown"):
            raise RuntimeError("supported claim lacks cited source evidence and scope")
        if verdict == "supported" and scope != extracted["claim_scope"]:
            verdict = "uncertain"
        reviews.append(
            {
                "claim": claim,
                **judgment,
                "verdict": verdict,
                "claimed_scope": extracted["claim_scope"],
                "source_text": " ".join(source_units[first : last + 1]) if first >= 0 else "",
            }
        )
    verdict = (
        "supported"
        if all(item["verdict"] == "supported" for item in reviews)
        else "unsupported"
        if any(item["verdict"] == "unsupported" for item in reviews)
        else "uncertain"
    )
    return {"verdict": verdict, "claims": reviews, "experimental": True}
