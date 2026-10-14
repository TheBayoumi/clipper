"""Source-first, claim-blind answers for atomic editorial questions.

The answerer sees a question and immutable delivered speech, never the headline's
proposed answer. Exact quotation and position checks prevent fabricated evidence;
they do not establish that a quotation semantically answers the question.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

Completion = Callable[[str, dict[str, Any], dict[str, Any], int], dict[str, Any]]
_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_STATES = {"answered", "unknown", "uncertain"}


def answer_source_question(
    question: str,
    delivered_units: Sequence[str],
    completion: Completion,
) -> dict[str, Any]:
    """Bind a source-only answer to exact delivered speech, or abstain."""
    if (
        not isinstance(question, str)
        or not question.strip().endswith("?")
        or not delivered_units
        or any(not isinstance(unit, str) or not unit.strip() for unit in delivered_units)
    ):
        raise ValueError("source answer requires a question and delivered speech")
    positions = [-1, *range(len(delivered_units))]
    proposal = completion(
        "Answer the question from delivered source speech only. Do not see or infer any "
        "headline's proposed answer. Distinguish who acted from people merely mentioned, "
        "actual settings from quoted instructions, actual outcomes from conditionals, "
        "and reported opinions from events. If the source does not establish an answer, "
        "choose unknown; if antecedents or scope are ambiguous, choose uncertain. "
        "For answered, copy the shortest exact continuous source words retaining actors, "
        "negation and conditions, and cite the containing unit range. For unknown or "
        "uncertain, use an empty quote and -1 positions. Return output_schema JSON.",
        {
            "question": question,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {
            "status": {"type": "string", "enum": sorted(_STATES)},
            "answer_quote": {"type": "string"},
            "first_unit": {"type": "integer", "enum": positions},
            "last_unit": {"type": "integer", "enum": positions},
        },
        112,
    )
    if not isinstance(proposal, dict) or set(proposal) != {
        "status",
        "answer_quote",
        "first_unit",
        "last_unit",
    }:
        raise ValueError("source answer has missing or unknown fields")
    status, quote, first, last = (
        proposal["status"],
        proposal["answer_quote"],
        proposal["first_unit"],
        proposal["last_unit"],
    )
    if (
        not isinstance(status, str)
        or status not in _STATES
        or not isinstance(quote, str)
        or type(first) is not int
        or type(last) is not int
    ):
        raise ValueError("source answer has invalid labels or positions")
    if status != "answered":
        if quote or first != -1 or last != -1:
            raise ValueError("unanswered source question cannot cite an answer")
        citation = None
    else:
        if not 0 <= first <= last < len(delivered_units) or last - first > 11:
            raise ValueError("answered source question needs a focused source range")
        source_text = " ".join(delivered_units[first : last + 1])
        if not 1 <= len(_WORD.findall(quote)) <= 24 or quote not in source_text:
            raise ValueError("source answer quote is not exact cited speech")
        citation = {"first_unit": first, "last_unit": last, "text": source_text}
    return {
        "question": question,
        "status": status,
        "answer_quote": quote,
        "citation": citation,
        "answer_semantically_verified": False,
        "diagnostic_only": True,
        "production_approved": False,
    }
