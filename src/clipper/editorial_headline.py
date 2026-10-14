"""Deterministic materialization of source-quoted editorial headlines.

This is a constraint on generation, not a semantic truth certificate. In
particular, an exact quotation can still be misleading outside its context.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_SCOPE_WORDS = {
    "if",
    "unless",
    "not",
    "no",
    "never",
    "only",
    "just",
    "would",
    "could",
    "might",
    "may",
    "should",
    "don't",
    "doesn't",
    "didn't",
    "wasn't",
    "weren't",
    "can't",
    "couldn't",
    "wouldn't",
    "won't",
    "said",
    "says",
    "told",
    "asked",
}

HeadlineCompletion = Callable[[str, dict[str, Any], dict[str, Any], int], dict[str, Any]]


def _source_excerpt(unit: str, excerpt: str) -> tuple[int, int]:
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError("source headline excerpt must be nonempty")
    start = unit.find(excerpt)
    if start < 0 or unit.find(excerpt, start + 1) >= 0:
        raise ValueError("source headline excerpt must occur once, verbatim, in its unit")
    end = start + len(excerpt)
    if (start and unit[start - 1].isalnum()) or (end < len(unit) and unit[end].isalnum()):
        raise ValueError("source headline excerpt must end on word boundaries")
    return start, end


def materialize_source_headline(
    source_units: Sequence[str],
    reviewed_spans: dict[str, Any],
    excerpts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Copy two reviewed setup/payoff fragments without model-written connectors.

    This intentionally abstains when a clipped fragment drops a negation,
    condition, modality or reporting word from its containing source unit.
    A separate contextual review and empirical qualification are still needed.
    """
    if (
        not source_units
        or any(not isinstance(unit, str) or not unit.strip() for unit in source_units)
        or not isinstance(reviewed_spans, dict)
        or set(reviewed_spans) != {"setup_quote", "resolution_quote"}
        or not isinstance(excerpts, (tuple, list))
        or len(excerpts) != 2
    ):
        raise ValueError("source headline needs delivered units and two reviewed roles")
    result: list[dict[str, Any]] = []
    for part, role in zip(excerpts, ("setup_quote", "resolution_quote"), strict=True):
        span = reviewed_spans[role]
        if (
            not isinstance(span, dict)
            or set(span) != {"text", "first_unit", "last_unit"}
            or type(span["first_unit"]) is not int
            or type(span["last_unit"]) is not int
            or not 0 <= span["first_unit"] <= span["last_unit"] < len(source_units)
            or span["text"] != " ".join(source_units[span["first_unit"] : span["last_unit"] + 1])
        ):
            raise ValueError("source headline reviewed role is not canonical speech")
        if not isinstance(part, dict) or set(part) != {"unit_id", "text"}:
            raise ValueError("source headline excerpt has invalid fields")
        unit_id, excerpt = part["unit_id"], part["text"]
        if type(unit_id) is not int or not span["first_unit"] <= unit_id <= span["last_unit"]:
            raise ValueError("source headline excerpt is outside its reviewed role")
        unit = source_units[unit_id]
        start, end = _source_excerpt(unit, excerpt)
        if not 3 <= len(_WORD.findall(excerpt)) <= 11:
            raise ValueError("source headline excerpt must contain 3-11 source words")
        omitted = unit[:start] + " " + unit[end:]
        omitted_scope = {word.casefold() for word in _WORD.findall(omitted)} & _SCOPE_WORDS
        if omitted_scope:
            raise ValueError(
                "source headline excerpt omits source scope: " + ", ".join(sorted(omitted_scope))
            )
        result.append(
            {
                "role": role,
                "unit_id": unit_id,
                "source_text": excerpt,
                "start_char": start,
                "end_char": end,
            }
        )
    if result[0]["unit_id"] > result[1]["unit_id"]:
        raise ValueError("source headline reverses reviewed setup and payoff")
    headline = f"{result[0]['source_text']} — {result[1]['source_text']}"
    if not 4 <= len(_WORD.findall(headline)) <= 14:
        raise ValueError("source headline exceeds the 4-14 word hook limit")
    return {
        "headline": headline,
        "source_excerpts": result,
        "source_bound": True,
        "contextually_verified": False,
        "production_approved": False,
    }


def propose_source_headline(
    source_units: Sequence[str],
    reviewed_spans: dict[str, Any],
    completion: HeadlineCompletion,
) -> dict[str, Any]:
    """Ask a model for positions and copied words; Python authors the headline.

    The proposal is intentionally unapproved. A contextual relation/scope
    verifier and an editorial quality check must be qualified separately.
    """
    properties = {
        role: {
            "type": "object",
            "properties": {
                "unit_id": {"type": "integer", "enum": list(range(len(source_units)))},
                "text": {"type": "string"},
            },
            "required": ["unit_id", "text"],
            "additionalProperties": False,
        }
        for role in ("setup_quote", "resolution_quote")
    }
    proposal = completion(
        "Choose one exact, contiguous 3-11 word excerpt from the reviewed setup "
        "and one from the reviewed resolution. Preserve the speaker's factual "
        "scope: do not drop any negation, conditional, modality or reporting "
        "language. Together they must convey the central contrast or payoff "
        "in 4-14 words. Return only source unit IDs and verbatim source words; "
        "never write a paraphrase or connective. Return output_schema JSON.",
        {
            "delivered_units": [
                {"id": index, "text": text} for index, text in enumerate(source_units)
            ],
            "reviewed_spans": reviewed_spans,
        },
        properties,
        160,
    )
    if not isinstance(proposal, dict) or set(proposal) != set(properties):
        raise ValueError("source headline proposal omitted a reviewed role")
    return materialize_source_headline(
        source_units,
        reviewed_spans,
        [proposal["setup_quote"], proposal["resolution_quote"]],
    )
