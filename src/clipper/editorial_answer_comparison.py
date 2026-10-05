"""Compare a headline answer with a cited source answer, after source-only review.

This is a diagnostic model judgment, not a qualified entailment decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from clipper.editorial_source_scope import Completion

_RELATIONS = ("equivalent", "different", "uncertain")
_RESPONSIVENESS = ("answers_question", "does_not_answer", "uncertain")
_SOURCE_SCOPES = {
    "actual_event_or_state",
    "actual_report_of_opinion",
    "quoted_instruction",
    "hypothetical_or_conditional",
    "negated_actual_state",
}
_PROPOSED_SCOPES = (*sorted(_SOURCE_SCOPES), "unknown")


def compare_claim_answer(
    *,
    question: str,
    proposed_answer: str,
    source_answer: dict[str, Any],
    scope_review: dict[str, Any],
    delivered_units: Sequence[str],
    completion: Completion,
) -> dict[str, Any]:
    """Assess answer equivalence only after source responsiveness is established."""
    if not isinstance(question, str) or not question.strip().endswith("?"):
        raise ValueError("answer comparison needs the exact relation question")
    if not isinstance(proposed_answer, str) or not proposed_answer.strip():
        raise ValueError("answer comparison needs a proposed headline answer")
    if not delivered_units or any(
        not isinstance(unit, str) or not unit.strip() for unit in delivered_units
    ):
        raise ValueError("answer comparison needs delivered source speech")
    if (
        not isinstance(source_answer, dict)
        or source_answer.get("question") != question
        or source_answer.get("status") != "answered"
        or not isinstance(scope_review, dict)
        or scope_review.get("question") != question
        or scope_review.get("responsiveness") != "answers_question"
        or scope_review.get("answer_quote") != source_answer.get("answer_quote")
        or scope_review.get("citation") != source_answer.get("citation")
        or scope_review.get("scope") not in _SOURCE_SCOPES
        or scope_review.get("diagnostic_only") is not True
        or scope_review.get("production_approved") is not False
    ):
        raise ValueError("answer comparison needs a responsive, scoped source answer")
    citation = source_answer.get("citation")
    quote = source_answer.get("answer_quote")
    if not isinstance(citation, dict) or not isinstance(quote, str) or not quote:
        raise ValueError("answer comparison needs an exact cited source quote")
    first, last = citation.get("first_unit"), citation.get("last_unit")
    if (
        type(first) is not int
        or type(last) is not int
        or not 0 <= first <= last < len(delivered_units)
        or citation.get("text") != " ".join(delivered_units[first : last + 1])
        or quote not in citation["text"]
    ):
        raise ValueError("answer comparison citation differs from delivered speech")
    proposed = completion(
        "Independently re-read the complete source speech. First decide whether "
        "source_answer_quote actually answers the question, including its actor, role, "
        "time, referent and speech act. Words spoken as someone else's warning, rule or "
        "instruction do not establish that setting or event as an actual occurrence. "
        "Then classify the source event scope and compare proposed_answer with what "
        "the source actually establishes. Respect actor, role, quantity, polarity, "
        "attribution, time and actual-versus-conditional scope. The source quote is "
        "not automatically a responsive or true answer. A name merely mentioned "
        "elsewhere, opposite polarity, or a hypothetical amount presented as actual "
        "is different. Choose uncertain if the source leaves the relation unresolved. "
        "Do not invent a new source answer. Return output_schema JSON only.",
        {
            "question": question,
            "proposed_answer": proposed_answer,
            "source_answer_quote": quote,
            "cited_first_unit": first,
            "cited_last_unit": last,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {
            "responsiveness": {"type": "string", "enum": list(_RESPONSIVENESS)},
            "source_scope": {"type": "string", "enum": list(_PROPOSED_SCOPES)},
            "relation": {"type": "string", "enum": list(_RELATIONS)},
        },
        96,
    )
    if not isinstance(proposed, dict) or set(proposed) != {
        "responsiveness",
        "source_scope",
        "relation",
    }:
        raise ValueError("answer comparison has missing or unknown fields")
    if (
        proposed["responsiveness"] not in _RESPONSIVENESS
        or proposed["source_scope"] not in _PROPOSED_SCOPES
        or proposed["relation"] not in _RELATIONS
    ):
        raise ValueError("answer comparison returned an invalid relation or scope")
    independent_agreement = (
        proposed["responsiveness"] == "answers_question"
        and proposed["source_scope"] == scope_review["scope"]
    )
    return {
        "question": question,
        "proposed_answer": proposed_answer,
        "source_answer_quote": quote,
        "upstream_source_scope": scope_review["scope"],
        "independent_responsiveness": proposed["responsiveness"],
        "independent_source_scope": proposed["source_scope"],
        "independent_scope_agrees": independent_agreement,
        "raw_relation": proposed["relation"],
        "relation": proposed["relation"] if independent_agreement else "uncertain",
        "diagnostic_only": True,
        "production_approved": False,
    }
