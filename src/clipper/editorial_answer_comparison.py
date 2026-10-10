"""Compare a headline answer with a cited source answer, after source-only review.

This is a diagnostic model judgment, not a qualified entailment decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from clipper.editorial_source_scope import NARRATIVE_ROLES, Completion

_RELATIONS = ("equivalent", "different", "uncertain")
_EVENT_LINKS = ("same_event", "different_event", "unresolved")
# A comparison is about what the candidate asserts. Source interpretation
# belongs to the independent source-only proposition and must not be predicted
# again using an incompatible source-scope vocabulary.
_CLAIM_TYPES = (
    "event_or_state",
    "reported_words",
    "reported_belief",
    "conditional_or_hypothetical",
    "unknown",
)


def _source_claim_type(scope_review: dict[str, Any]) -> str:
    proposition = scope_review.get("source_proposition")
    if isinstance(proposition, dict):
        if proposition.get("attribution") == "reported_speech":
            return "reported_words"
        if proposition.get("attribution") == "reported_belief":
            return "reported_belief"
        if proposition.get("modality") in {"conditional", "hypothetical"}:
            return "conditional_or_hypothetical"
        if proposition.get("attribution") in {
            "first_person_account",
            "third_person_account",
            "general_claim",
        }:
            return "event_or_state"
        return "unknown"
    # Legacy direct API inputs remain diagnostic, never source qualification.
    role = scope_review.get("narrative_role")
    return {
        "recounted_event": "event_or_state",
        "present_event_or_state": "event_or_state",
        "reported_utterance": "reported_words",
        "reported_belief": "reported_belief",
        "conditional_or_hypothetical": "conditional_or_hypothetical",
    }.get(role, "unknown")


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
    from clipper.editorial_propositions import validate_source_proposition

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
    if "source_proposition" in scope_review:
        validate_source_proposition(
            scope_review["source_proposition"],
            answer_quote=quote,
            citation=citation,
            delivered_units=delivered_units,
        )
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
        "because two statements occur in the same anecdote. Resolve fragments of "
        "speech that cross adjacent source-unit boundaries, and accept natural "
        "paraphrases which omit grammatical scaffolding without changing actors, "
        "roles, polarity, temporal meaning or event. Do not demand surface word "
        "identity when the same answer is clearly entailed. Distinguish the roles "
        "of the proposed claim only. source_proposition is a previously recorded "
        "source-only account with fixed attribution, modality and polarity: do "
        "not reinterpret it from the proposed headline. Claim type is either "
        "event_or_state, reported_words, reported_belief, "
        "conditional_or_hypothetical or unknown; these labels are distinct from "
        "any source-scope values. Choose same_event only when the source "
        "supports the asserted relation for the SAME actors, action, occasion, scope "
        "and timeline; preserve actual-versus-conditional meaning, negation and "
        "attribution. A different_event label means the source demonstrates that "
        "the ASSERTED relation or actor is incompatible, even if the statement "
        "belongs to the same scene. For different_event you MAY cite the original "
        "contradictory source units: negative evidence is still evidence. For "
        "same_event you MUST cite supporting units. For unresolved use -1/-1 "
        "because no event linkage was established. Cite the narrowest contiguous "
        "original source-unit range; do not author source quotes. "
        "Relation equivalent requires the SAME meaning, not merely a similar topic. "
        "Return output_schema JSON only.",
        {
            "question": question,
            "proposed_answer": proposed_answer,
            "source_answer_quote": quote,
            "source_scope": scope_review["scope"],
            "blind_source_narrative_role": scope_review["narrative_role"],
            "source_proposition": scope_review.get("source_proposition"),
            "cited_first_unit": first,
            "cited_last_unit": last,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {
            "relation": {"type": "string", "enum": list(_RELATIONS)},
            "claim_type": {"type": "string", "enum": list(_CLAIM_TYPES)},
            "event_link": {"type": "string", "enum": list(_EVENT_LINKS)},
            "event_first_unit": {"type": "integer", "enum": [-1, *range(len(delivered_units))]},
            "event_last_unit": {"type": "integer", "enum": [-1, *range(len(delivered_units))]},
        },
        160,
    )
    required = {
        "relation",
        "claim_type",
        "event_link",
        "event_first_unit",
        "event_last_unit",
    }
    if not isinstance(proposed, dict) or set(proposed) != required:
        raise ValueError("narrative comparison has missing or unknown fields")
    if (
        proposed["relation"] not in _RELATIONS
        or proposed["claim_type"] not in _CLAIM_TYPES
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
    # A model can cite the transcript to demonstrate a *contradiction*. The
    # earlier biconditional rejected all such negative evidence, obscuring three
    # correct 'different' judgments in real GPU qualification. Only unresolved
    # linkages must abstain from citations; positive matches must cite support.
    if (proposed["event_link"] == "same_event" and event_first < 0) or (
        proposed["event_link"] == "unresolved" and event_first >= 0
    ):
        raise ValueError("narrative event linkage and source positions disagree")
    # An equivalent topic is not proof that it refers to the same event, or
    # that a quoted instruction, imagined outcome, or recollection is an
    # independently occurring event. Do not change a model's 'different' into
    # approval; ambiguous linkage can only reduce confidence.
    source_type = _source_claim_type(scope_review)
    narrative_consistent = (
        proposed["claim_type"] == source_type
        and source_type != "unknown"
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
            "claimed_type": proposed["claim_type"],
            "source_type": source_type,
            "source_anchor_id": (scope_review.get("source_proposition") or {}).get("anchor_id"),
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
