"""Experimental source-first QA audit for complete headline propositions.

The source-answer call never receives the headline or its claimed answer. Python
checks exact text provenance, coverage and modality before aggregating. Model
judgments are still fallible and are not a publication approval.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

_WORD = re.compile(r"\w+", re.UNICODE)
_SCOPES = ("actual_event", "future_plan", "hypothetical", "reported_speech", "unknown")
_COMPARISONS = ("equivalent", "different", "uncertain")


class QAReviewer(Protocol):
    def _review_completion(
        self, prompt: str, payload: dict[str, Any], properties: dict[str, Any], tokens: int
    ) -> dict[str, Any]: ...


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def audit_source_qa(reviewer: QAReviewer, headline: str, source_units: list[str]) -> dict[str, Any]:
    """Assess headline relations using blind source answers and cited spans."""
    if not headline.strip() or not source_units or any(not unit.strip() for unit in source_units):
        raise ValueError("source QA requires a headline and nonempty source speech")
    extracted = reviewer._review_completion(
        "Turn every factual predicate-argument relation in the headline into a neutral, "
        "open question and a short claimed answer. Include actors, actions, objects, "
        "roles, attribution, numbers, temporal status, conditions and consequences. "
        "Each claim_text must be an exact contiguous substring of the headline, and "
        "claimed_answer must be an exact substring of claim_text. A question must ask "
        "for the relation without supplying the claimed answer. Classify whether the "
        "claim describes an actual event, future plan, hypothetical, or reported speech. "
        "Do not inspect source speech. Return output_schema JSON.",
        {"headline": headline},
        {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_text": {"type": "string"},
                        "question": {"type": "string"},
                        "claimed_answer": {"type": "string"},
                        "claimed_scope": {"type": "string", "enum": list(_SCOPES[:-1])},
                    },
                    "required": [
                        "claim_text",
                        "question",
                        "claimed_answer",
                        "claimed_scope",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        512,
    )["questions"]
    if not isinstance(extracted, list) or not 1 <= len(extracted) <= 8:
        raise RuntimeError("source QA extracted an invalid question count")
    coverage = [False] * len(headline)
    checked: list[dict[str, Any]] = []
    for item in extracted:
        if not isinstance(item, dict) or set(item) != {
            "claim_text",
            "question",
            "claimed_answer",
            "claimed_scope",
        }:
            raise RuntimeError("source QA extracted an invalid proposition")
        claim = item["claim_text"]
        answer = item["claimed_answer"]
        question = item["question"]
        if (
            not isinstance(claim, str)
            or not isinstance(answer, str)
            or not isinstance(question, str)
            or not _WORD.search(claim)
            or not _WORD.search(answer)
            or "?" not in question
            or claim not in headline
            or answer not in claim
            or item["claimed_scope"] not in _SCOPES[:-1]
        ):
            raise RuntimeError("source QA proposition is not grounded in headline text")
        start = headline.index(claim)
        for index in range(start, start + len(claim)):
            coverage[index] = True
        checked.append(item)
    uncovered = [
        match.group()
        for match in _WORD.finditer(headline)
        if not all(coverage[match.start() : match.end()])
    ]
    source = [{"id": index, "text": unit} for index, unit in enumerate(source_units)]
    results = []
    for item in checked:
        source_answer = reviewer._review_completion(
            "Answer the neutral question from delivered source_units only. You have not "
            "seen the headline or its proposed answer. Copy the shortest exact source "
            "words that answer it, including any 'if', 'can', future-plan or quoted-speech "
            "scope needed for interpretation. Cite one contiguous unit range. If speech "
            "does not establish an answer, return unknown and positions -1/-1. Do not "
            "infer that a rule or possible outcome actually happened. Return output_schema JSON.",
            {"question": item["question"], "source_units": source},
            {
                "answer_quote": {"type": "string"},
                "source_scope": {"type": "string", "enum": list(_SCOPES)},
                "first_unit": {"type": "integer", "minimum": -1},
                "last_unit": {"type": "integer", "minimum": -1},
            },
            256,
        )
        if set(source_answer) != {"answer_quote", "source_scope", "first_unit", "last_unit"}:
            raise RuntimeError("source QA answer has invalid fields")
        quote = source_answer["answer_quote"]
        scope = source_answer["source_scope"]
        first, last = source_answer["first_unit"], source_answer["last_unit"]
        if (
            not isinstance(quote, str)
            or scope not in _SCOPES
            or type(first) is not int
            or type(last) is not int
            or not (first == last == -1 or 0 <= first <= last < len(source_units))
        ):
            raise RuntimeError("source QA answer has invalid evidence positions")
        if (first == -1) != (scope == "unknown"):
            raise RuntimeError("unknown source answer and absent evidence must agree")
        cited = " ".join(source_units[first : last + 1]) if first >= 0 else ""
        if first >= 0 and (not quote.strip() or _normalized(quote) not in _normalized(cited)):
            raise RuntimeError("source QA answer is not an exact cited source passage")
        if first == -1 and quote.strip().casefold() != "unknown":
            raise RuntimeError("absent source answer must be unknown")
        comparison: dict[str, Any] = {"relation": "uncertain", "reason": "No source answer."}
        if first >= 0 and scope == item["claimed_scope"]:
            comparison = reviewer._review_completion(
                "Compare these two short answers to the same neutral question. Mark "
                "equivalent only when they assert the same actors, roles, action, polarity, "
                "amount and temporal/conditional meaning. Mark different for a conflict "
                "and uncertain for insufficient detail. The cited source answer, not the "
                "claimed answer, is authoritative. Return output_schema JSON.",
                {
                    "question": item["question"],
                    "claimed_answer": answer,
                    "source_answer": quote,
                },
                {
                    "relation": {"type": "string", "enum": list(_COMPARISONS)},
                    "reason": {"type": "string", "minLength": 1},
                },
                128,
            )
            if (
                set(comparison) != {"relation", "reason"}
                or comparison["relation"] not in _COMPARISONS
                or not isinstance(comparison["reason"], str)
                or not comparison["reason"].strip()
            ):
                raise RuntimeError("source QA comparison has invalid fields")
        results.append(
            {
                **item,
                "source_answer": quote,
                "source_scope": scope,
                "source_span": {"first_unit": first, "last_unit": last, "text": cited}
                if first >= 0
                else None,
                **comparison,
            }
        )
    verdict = (
        "supported"
        if not uncovered and all(row["relation"] == "equivalent" for row in results)
        else "unsupported"
        if any(row["relation"] == "different" for row in results)
        else "uncertain"
    )
    return {
        "verdict": verdict,
        "questions": results,
        "uncovered_headline_words": uncovered,
        "experimental": True,
    }
