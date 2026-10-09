"""Compare a headline answer with a cited source answer, after source-only review.

This is a diagnostic model judgment, not a qualified entailment decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from clipper.editorial_source_scope import NARRATIVE_ROLES, Completion

_RELATIONS = ("equivalent", "different", "uncertain")
_EVENT_LINKS = ("same_event", "different_event", "unresolved")
_SOURCE_SCOPES = {
    "actual_event_or_state",
    "actual_report_of_opinion",
    "quoted_instruction",
    "hypothetical_or_conditional",
    "negated_actual_state",
}


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
        or scope_review.get("narrative_role") not in NARRATIVE_ROLES
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
        "Compare proposed_answer with source_answer_quote as answers to the same "
        "question, using ALL source_units to resolve who was speaking, the story's "
        "narrative viewpoint, and the events being recounted. A podcast guest may "
        "recount a past firsthand experience; treat this as a recounted event, not "
        "a present live event and not as inherently hypothetical. Dialogue quoted "
        "INSIDE that story establishes that the dialogue was reported, but does not "
        "automatically establish that the dialogue's claims occurred. Conversely, "
        "a headline explicitly about what someone said may be supported by the quoted "
        "utterance itself. Track referents across natural speech, including pronouns, "
        "interruptions, scene changes, and time shifts. Do not infer new details merely "
        "because two statements occur in the same anecdote. Distinguish the roles "
        "of the headline claim and supporting source proposition: recounted event, "
        "present event/state, reported utterance, reported belief, conditional, "
        "general discussion, or unknown. Choose same_event only when the source "
        "supports the asserted relation for the SAME actors, action, occasion, scope "
        "and timeline; preserve actual-versus-conditional meaning, negation and "
        "attribution; different_event for incompatible references, unresolved for "
        "ambiguous linkage. Cite the narrowest contiguous original source-unit "
        "range establishing that event linkage; do not author source quotes. "
        "Relation equivalent requires the SAME meaning, not merely a similar topic. "
        "Return output_schema JSON only.",
        {
            "question": question,
            "proposed_answer": proposed_answer,
            "source_answer_quote": quote,
            "source_scope": scope_review["scope"],
            "blind_source_narrative_role": scope_review["narrative_role"],
            "cited_first_unit": first,
            "cited_last_unit": last,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {
            "relation": {"type": "string", "enum": list(_RELATIONS)},
            "claimed_narrative_role": {"type": "string", "enum": list(NARRATIVE_ROLES)},
            "source_narrative_role": {"type": "string", "enum": list(NARRATIVE_ROLES)},
            "event_link": {"type": "string", "enum": list(_EVENT_LINKS)},
            "event_first_unit": {"type": "integer", "enum": [-1, *range(len(delivered_units))]},
            "event_last_unit": {"type": "integer", "enum": [-1, *range(len(delivered_units))]},
        },
        160,
    )
    required = {
        "relation",
        "claimed_narrative_role",
        "source_narrative_role",
        "event_link",
        "event_first_unit",
        "event_last_unit",
    }
    if not isinstance(proposed, dict) or set(proposed) != required:
        raise ValueError("narrative comparison has missing or unknown fields")
    if (
        proposed["relation"] not in _RELATIONS
        or proposed["claimed_narrative_role"] not in NARRATIVE_ROLES
        or proposed["source_narrative_role"] not in NARRATIVE_ROLES
        or proposed["event_link"] not in _EVENT_LINKS
    ):
        raise ValueError("narrative comparison returned invalid relation or context labels")
    event_first, event_last = proposed["event_first_unit"], proposed["event_last_unit"]
    if (
        type(event_first) is not int
        or type(event_last) is not int
        or not (
            event_first == event_last == -1 or 0 <= event_first <= event_last < len(delivered_units)
        )
    ):
        raise ValueError("narrative comparison returned invalid source positions")
    if (proposed["event_link"] == "same_event") != (event_first >= 0):
        raise ValueError("narrative event linkage and source positions disagree")
    # An equivalent topic is not proof that it refers to the same event, or
    # that a quoted instruction, imagined outcome, or recollection is an
    # independently occurring event. Do not change a model's 'different' into
    # approval; ambiguous linkage can only reduce confidence.
    narrative_consistent = (
        proposed["claimed_narrative_role"] == proposed["source_narrative_role"]
        and proposed["claimed_narrative_role"] != "unknown"
        and proposed["source_narrative_role"] == scope_review["narrative_role"]
        and proposed["event_link"] == "same_event"
    )
    relation = (
        proposed["relation"]
        if proposed["relation"] != "equivalent" or narrative_consistent
        else "uncertain"
    )
    return {
        "question": question,
        "proposed_answer": proposed_answer,
        "source_answer_quote": quote,
        "source_scope": scope_review["scope"],
        "relation": relation,
        "narrative_evidence": {
            "claimed_role": proposed["claimed_narrative_role"],
            "source_role": proposed["source_narrative_role"],
            "blind_source_role": scope_review["narrative_role"],
            "event_link": proposed["event_link"],
            "event_source_span": {
                "first_unit": event_first,
                "last_unit": event_last,
                "text": " ".join(delivered_units[event_first : event_last + 1]),
            }
            if event_first >= 0
            else None,
            "relation_before_guard": proposed["relation"],
        },
        "diagnostic_only": True,
        "production_approved": False,
    }
