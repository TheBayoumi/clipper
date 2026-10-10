"""Provenance-bound narrative proposition extracted independently of a headline.

The extractor sees only verbatim speech and a pinned answer quote. It does not
receive the question or candidate headline, preventing them from rebinding the
source's speech act or its event anchor. All labels are model judgments and
never constitute external truth or production approval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from clipper.editorial_source_scope import NARRATIVE_ROLES, Completion

ATTRIBUTIONS = (
    "first_person_account",
    "third_person_account",
    "reported_speech",
    "reported_belief",
    "general_claim",
    "unknown",
)
TIMEFRAMES = ("past", "present", "future", "unspecified")
POLARITIES = ("affirmed", "negated", "mixed", "unknown")
MODALITIES = ("asserted", "conditional", "hypothetical", "unknown")


def validate_source_proposition(
    proposition: object,
    *,
    answer_quote: str,
    citation: dict[str, Any],
    delivered_units: Sequence[str],
) -> None:
    """Verify immutable source bytes and schema, never infer semantic truth."""
    first, last = citation.get("first_unit"), citation.get("last_unit")
    if (
        type(first) is not int
        or type(last) is not int
        or not 0 <= first <= last < len(delivered_units)
        or citation.get("text") != " ".join(delivered_units[first : last + 1])
        or not isinstance(answer_quote, str)
        or not answer_quote
        or answer_quote not in citation["text"]
    ):
        raise ValueError("proposition source citation does not match delivered speech")
    anchor = {
        "first_unit": first,
        "last_unit": last,
        "text": citation["text"],
        "quote": answer_quote,
    }
    expected_id = hashlib.sha256(
        json.dumps(anchor, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    fields = {
        "narrative_role": NARRATIVE_ROLES,
        "attribution": ATTRIBUTIONS,
        "timeframe": TIMEFRAMES,
        "polarity": POLARITIES,
        "modality": MODALITIES,
    }
    required = {
        "schema",
        "anchor_id",
        "citation",
        "answer_quote",
        *fields,
        "actor",
        "predicate",
        "diagnostic_only",
        "production_approved",
    }
    if (
        not isinstance(proposition, dict)
        or set(proposition) != required
        or proposition.get("schema") != "source_proposition_v1"
        or proposition.get("anchor_id") != expected_id
        or proposition.get("citation") != citation
        or proposition.get("answer_quote") != answer_quote
        or proposition.get("diagnostic_only") is not True
        or proposition.get("production_approved") is not False
    ):
        raise ValueError("source proposition identity or schema mismatch")
    if any(proposition[name] not in choices for name, choices in fields.items()):
        raise ValueError("source proposition label contract mismatch")
    if any(
        not isinstance(proposition[name], str) or len(proposition[name]) > 160
        for name in ("actor", "predicate")
    ):
        raise ValueError("source proposition actor/predicate invalid")
    if (
        proposition["attribution"] == "reported_speech"
        and proposition["narrative_role"] != "reported_utterance"
    ) or (
        proposition["attribution"] == "reported_belief"
        and proposition["narrative_role"] != "reported_belief"
    ):
        raise ValueError("source proposition attribution conflicts with speech act")


def extract_source_proposition(
    *,
    answer_quote: str,
    citation: dict[str, Any],
    delivered_units: Sequence[str],
    completion: Completion,
) -> dict[str, Any]:
    """Describe the cited source as speech, not as a verified real-world event."""
    first, last = citation.get("first_unit"), citation.get("last_unit")
    if (
        type(first) is not int
        or type(last) is not int
        or not 0 <= first <= last < len(delivered_units)
        or citation.get("text") != " ".join(delivered_units[first : last + 1])
        or not answer_quote
        or answer_quote not in citation["text"]
    ):
        raise ValueError("proposition extraction requires exact source quotation provenance")
    proposed = completion(
        "Extract the proposition expressed by answer_quote, using surrounding "
        "source_units ONLY for pronouns, speaker attribution, speech-act boundaries "
        "and time context. Do NOT answer a question; none is provided. Do not infer "
        "a headline. An interviewee's story about an earlier experience is a past "
        "source account; dialogue REPEATED inside that story is reported_speech, "
        "not direct evidence that what its words claim actually happened. A quote "
        "starting 'we're live' after 'he told us' remains reported_speech. In "
        "contrast, 'I made nothing from the views' is an asserted first_person_account "
        "with negated polarity; 'you don't make anything from views' can be the "
        "speaker's general statement about their own income in context, but should "
        "not automatically be extrapolated to everyone. Track conditional/hypothetical "
        "and negation without dropping them. `actor` and `predicate` are concise "
        "source-grounded strings, or empty strings if unclear; do not invent names. "
        "Classify the speech act of the ANCHORED words rather than a neighboring "
        "event in the same anecdote. Return output_schema JSON only.",
        {
            "answer_quote": answer_quote,
            "cited_first_unit": first,
            "cited_last_unit": last,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {
            "narrative_role": {"type": "string", "enum": list(NARRATIVE_ROLES)},
            "attribution": {"type": "string", "enum": list(ATTRIBUTIONS)},
            "timeframe": {"type": "string", "enum": list(TIMEFRAMES)},
            "polarity": {"type": "string", "enum": list(POLARITIES)},
            "modality": {"type": "string", "enum": list(MODALITIES)},
            "actor": {"type": "string"},
            "predicate": {"type": "string"},
        },
        200,
    )
    fields = {
        "narrative_role": NARRATIVE_ROLES,
        "attribution": ATTRIBUTIONS,
        "timeframe": TIMEFRAMES,
        "polarity": POLARITIES,
        "modality": MODALITIES,
    }
    if not isinstance(proposed, dict) or set(proposed) != {*fields, "actor", "predicate"}:
        raise ValueError("source proposition fields differ from the versioned contract")
    if any(proposed[name] not in choices for name, choices in fields.items()):
        raise ValueError("source proposition has invalid narrative labels")
    if any(
        not isinstance(proposed[name], str) or len(proposed[name]) > 160
        for name in ("actor", "predicate")
    ):
        raise ValueError("source proposition actor/predicate are invalid")
    anchor = {
        "first_unit": first,
        "last_unit": last,
        "text": citation["text"],
        "quote": answer_quote,
    }
    # An anchor ID identifies source bytes and positions, not an imagined event;
    # no cross-scene identity is asserted or inferred from adjacency alone.
    anchor_id = hashlib.sha256(
        json.dumps(anchor, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    evidence = {
        "schema": "source_proposition_v1",
        "anchor_id": anchor_id,
        "citation": dict(citation),
        "answer_quote": answer_quote,
        **proposed,
        "diagnostic_only": True,
        "production_approved": False,
    }
    validate_source_proposition(
        evidence, answer_quote=answer_quote, citation=citation, delivered_units=delivered_units
    )
    return evidence
