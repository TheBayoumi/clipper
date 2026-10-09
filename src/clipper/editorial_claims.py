"""Experimental, source-positioned headline audit; not a publication gate.

The model proposes subclaims and judgments. Python retains the whole headline,
checks copied subclaims, and owns evidence identity and verdict aggregation.
A valid pointer proves provenance, not truth.
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
        "Copy each claim as an exact contiguous substring of the headline. "
        "Include attribution, "
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
                        "claim_scope": {"type": "string", "enum": list(_SCOPES)},
                    },
                    "required": ["text", "claim_scope"],
                    "additionalProperties": False,
                },
            }
        },
        384,
    )["claims"]
    if not isinstance(claims_result, list) or not 1 <= len(claims_result) <= 8:
        raise RuntimeError("claim extraction returned an invalid claim count")
    coverage = [False] * len(headline)
    claims: list[dict[str, Any]] = [
        {
            "text": headline,
            "claim_scope": "whole_headline",
            "first_char": 0,
            "last_char": len(headline),
        }
    ]
    for item in claims_result:
        if not isinstance(item, dict) or set(item) != {"text", "claim_scope"}:
            raise RuntimeError("claim extraction returned an invalid span")
        value = item["text"]
        if (
            not isinstance(value, str)
            or not _WORD.search(value)
            or item["claim_scope"] not in _SCOPES
            or value not in headline
        ):
            raise RuntimeError("claim extraction did not copy a headline span")
        start = headline.index(value)
        if headline.find(value, start + 1) != -1:
            raise RuntimeError("claim extraction has an ambiguous repeated headline span")
        for index in range(start, start + len(value)):
            coverage[index] = True
        if value != headline:
            claims.append({**item, "first_char": start, "last_char": start + len(value)})

    words = list(_WORD.finditer(headline))
    uncovered_words = [
        {"text": match.group(), "first_char": match.start(), "last_char": match.end()}
        for match in words
        if not all(coverage[match.start() : match.end()])
    ]

    reviews = []
    numbered_source = [{"id": index, "text": unit} for index, unit in enumerate(source_units)]
    for extracted in claims:
        claim = extracted["text"]
        judgment = reviewer._review_completion(
            "Assess this one headline claim against the delivered source speech. "
            "Use headline_context to preserve the target span's governing subject, "
            "attribution, conditions and negation. Headline context is the assertion "
            "being tested, never source evidence. Do not turn a fragment into an "
            "independent assertion by dropping that context. "
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
                "headline_context": headline,
                "headline_span": {
                    "first_char": extracted["first_char"],
                    "last_char": extracted["last_char"],
                },
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
        if (
            verdict == "supported"
            and extracted["claim_scope"] != "whole_headline"
            and scope != extracted["claim_scope"]
        ):
            verdict = "uncertain"
        reviews.append(
            {
                "claim": claim,
                **judgment,
                "verdict": verdict,
                "claimed_scope": extracted["claim_scope"],
                "source_text": " ".join(source_units[first : last + 1]) if first >= 0 else "",
                "headline_span": {
                    "first_char": extracted["first_char"],
                    "last_char": extracted["last_char"],
                },
            }
        )
    verdict = (
        "supported"
        if all(item["verdict"] == "supported" for item in reviews)
        else "unsupported"
        if any(item["verdict"] == "unsupported" for item in reviews)
        else "uncertain"
    )
    if verdict == "supported" and uncovered_words:
        verdict = "uncertain"
    return {
        "verdict": verdict,
        "claims": reviews,
        "subclaim_word_coverage": (len(words) - len(uncovered_words)) / len(words),
        "uncovered_headline_words": uncovered_words,
        "claim_text_coverage_complete": not uncovered_words,
        "claim_inventory_semantically_qualified": False,
        "production_approved": False,
        "experimental": True,
    }
