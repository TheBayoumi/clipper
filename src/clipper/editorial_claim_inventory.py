"""Headline-only predicate/argument inventory for factual review diagnostics.

The model cannot use source speech to decide which headline relations exist.
Python binds its proposed answers and claim ranges to headline words. This
checks provenance and coverage, not semantic completeness or source truth.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

Completion = Callable[[str, dict[str, Any], dict[str, Any], int], dict[str, Any]]
_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_KINDS = (
    "actor_action",
    "attribution",
    "event_modality",
    "condition",
    "relationship_role",
    "quantity_outcome",
    "setting_time",
)


def inventory_headline_relations(headline: str, completion: Completion) -> dict[str, Any]:
    """Propose source-blind atomic relation questions for an exact headline."""
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("claim inventory requires a headline")
    words = list(_WORD.finditer(headline))
    if not 4 <= len(words) <= 14:
        raise ValueError("claim inventory requires a 4-14 word headline")
    proposal = completion(
        "Inventory every independently checkable factual relation asserted by the "
        "headline. See only headline and word_index; no transcript or candidate source "
        "evidence is available. Each relation has one factual question, one exact "
        "answer word span from the headline, and the smallest headline word range "
        "expressing that relation. Include actor/action, attribution, relationship "
        "roles, settings, quantities, negation, conditional versus actual outcomes "
        "and reported versus enacted events when asserted. Do not silently omit a "
        "relation. Questions must ask what the headline claims, not whether it is "
        "true. Return output_schema JSON without explanations.",
        {
            "headline": headline,
            "word_index": [{"id": i, "text": word.group()} for i, word in enumerate(words)],
        },
        {
            "relations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(_KINDS)},
                        "question": {"type": "string"},
                        "answer_first_word": {"type": "integer", "minimum": 0},
                        "answer_last_word": {"type": "integer", "minimum": 0},
                        "claim_first_word": {"type": "integer", "minimum": 0},
                        "claim_last_word": {"type": "integer", "minimum": 0},
                    },
                    "required": [
                        "kind",
                        "question",
                        "answer_first_word",
                        "answer_last_word",
                        "claim_first_word",
                        "claim_last_word",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        416,
    )
    if not isinstance(proposal, dict) or set(proposal) != {"relations"}:
        raise ValueError("claim inventory has missing or unknown fields")
    raw_relations = proposal["relations"]
    if not isinstance(raw_relations, list) or not 1 <= len(raw_relations) <= 10:
        raise ValueError("claim inventory needs 1-10 relations")
    required = {
        "kind",
        "question",
        "answer_first_word",
        "answer_last_word",
        "claim_first_word",
        "claim_last_word",
    }
    covered = [False] * len(words)
    seen: set[tuple[str, str, int, int]] = set()
    relations: list[dict[str, Any]] = []
    for raw in raw_relations:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("claim inventory relation has missing or unknown fields")
        kind, question = raw["kind"], raw["question"]
        af, al, cf, cl = (
            raw["answer_first_word"],
            raw["answer_last_word"],
            raw["claim_first_word"],
            raw["claim_last_word"],
        )
        if (
            kind not in _KINDS
            or not isinstance(question, str)
            or not question.strip().endswith("?")
            or any(type(index) is not int for index in (af, al, cf, cl))
            or not 0 <= cf <= af <= al <= cl < len(words)
        ):
            raise ValueError("claim inventory relation is not bound to headline words")
        key = (kind, question.casefold().strip(), af, al)
        if key in seen:
            raise ValueError("claim inventory repeats an identical relation")
        seen.add(key)
        for index in range(cf, cl + 1):
            covered[index] = True
        relations.append(
            {
                **raw,
                "answer": headline[words[af].start() : words[al].end()],
                "claim_text": headline[words[cf].start() : words[cl].end()],
            }
        )
    uncovered = [word.group() for index, word in enumerate(words) if not covered[index]]
    return {
        "headline": headline,
        "relations": relations,
        "uncovered_headline_words": uncovered,
        "inventory_semantically_qualified": False,
        "diagnostic_only": True,
        "production_approved": False,
    }
