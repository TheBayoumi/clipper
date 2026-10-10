"""Independent responsiveness/scope diagnostic for a source-only answer.

The classifier receives a question, delivered speech and an exact cited quote,
never a proposed headline answer. Its labels are unqualified model judgments;
they cannot authorize production or replace claim-level entailment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

Completion = Callable[[str, dict[str, Any], dict[str, Any], int], dict[str, Any]]
_RESPONSIVENESS = ("answers_question", "does_not_answer", "uncertain")
# Evidence meaning within a conversational narrative. This is source-only:
# no headline or proposed answer may influence which speech act was performed.
NARRATIVE_ROLES = (
    "recounted_event",
    "present_event_or_state",
    "reported_utterance",
    "reported_belief",
    "conditional_or_hypothetical",
    "general_discussion",
    "unknown",
)
_QUESTION_TARGETS = (
    "described_event_or_state",
    "reported_words_or_belief",
    "conditional_or_hypothetical",
    "unknown",
)
_SCOPES = (
    "actual_event_or_state",
    "actual_report_of_opinion",
    "quoted_instruction",
    "hypothetical_or_conditional",
    "negated_actual_state",
    "unknown",
)


def assess_source_answer_scope(
    *,
    question: str,
    source_answer: dict[str, Any],
    delivered_units: Sequence[str],
    completion: Completion,
) -> dict[str, Any]:
    """Re-read the cited answer in full context, independent of headline claims."""
    if not isinstance(question, str) or not question.strip().endswith("?"):
        raise ValueError("source-scope review needs a factual question")
    if not delivered_units or any(
        not isinstance(unit, str) or not unit.strip() for unit in delivered_units
    ):
        raise ValueError("source-scope review needs delivered speech")
    if not isinstance(source_answer, dict) or source_answer.get("question") != question:
        raise ValueError("source-scope review question does not match source answer")
    if source_answer.get("status") != "answered":
        return {
            "question": question,
            "responsiveness": "uncertain",
            "scope": "unknown",
            "narrative_role": "unknown",
            "reason": "source_answer_abstained",
            "diagnostic_only": True,
            "production_approved": False,
        }
    quote, citation = source_answer.get("answer_quote"), source_answer.get("citation")
    if not isinstance(quote, str) or not quote or not isinstance(citation, dict):
        raise ValueError("source-scope review needs an exact cited answer")
    first, last = citation.get("first_unit"), citation.get("last_unit")
    if (
        type(first) is not int
        or type(last) is not int
        or not 0 <= first <= last < len(delivered_units)
    ):
        raise ValueError("source-scope review citation positions are invalid")
    cited = " ".join(delivered_units[first : last + 1])
    if citation.get("text") != cited or quote not in cited:
        raise ValueError("source-scope review citation is not exact delivered speech")
    from clipper.editorial_propositions import extract_source_proposition

    source_proposition = extract_source_proposition(
        answer_quote=quote,
        citation=citation,
        delivered_units=delivered_units,
        completion=completion,
    )
    # Two independent questions: classify question intent WITHOUT source, then
    # decide its relation to the PRE-FROZEN source proposition. Neither can
    # change what the source quote is, nor insert an imagined event.
    target_response = completion(
        "Read the QUESTION ONLY, without transcript or answer. Choose "
        "described_event_or_state for what actually happened, where, when, "
        "who acted, whether something really occurred, or actual income; "
        "reported_words_or_belief for what a speaker said, claimed, reported, "
        "thought or believed; conditional_or_hypothetical for what could, "
        "might or would happen. Choose unknown if unclear. A question asking "
        "what a guest SAYS about income is about the guest's reported words; "
        "asking what the guest ACTUALLY earned is about the described state. "
        "Return output_schema JSON only.",
        {"question": question},
        {"question_target": {"type": "string", "enum": list(_QUESTION_TARGETS)}},
        80,
    )
    if (
        not isinstance(target_response, dict)
        or set(target_response) != {"question_target"}
        or target_response["question_target"] not in _QUESTION_TARGETS
    ):
        raise ValueError("question-only target classifier returned invalid fields")
    question_target = target_response["question_target"]

    # The final model judges only responsiveness. Scope and narrative role are
    # derived from the already recorded SOURCE-ONLY proposition, not re-guessed
    # in a question-conditioned model where the broadcast quote was mislabeled.
    judgement = completion(
        "Decide if the exact answer_quote and its source_proposition answer the "
        "question about the SAME actors, referents, event and timeframe, using "
        "the full source units for resolving pronouns. Negative answers ARE "
        "responsive: 'you don't make anything from Instagram' can answer what "
        "the speaker says views earn or whether the speaker receives income. "
        "Distinguish a quote about a speaker's own income from a universal claim. "
        "A quoted sentence like 'We're live on TV' inside an old story establishes "
        "only that those words were reported: it does NOT establish the event's "
        "actual broadcast setting. The quoted speech may answer 'what was said?' "
        "but not 'was the event broadcast?'. Conditional claims do not prove "
        "outcomes. Don't invent referents or unify separate stories; return "
        "uncertain when context is inadequate. Select a RESPONSIVENESS label only; "
        "source scope and narrative role are fixed and not for you to change. "
        "Return output_schema JSON only.",
        {
            "question": question,
            "question_target": question_target,
            "answer_quote": quote,
            "source_proposition": source_proposition,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {"responsiveness": {"type": "string", "enum": list(_RESPONSIVENESS)}},
        100,
    )
    if (
        not isinstance(judgement, dict)
        or set(judgement) != {"responsiveness"}
        or judgement["responsiveness"] not in _RESPONSIVENESS
    ):
        raise ValueError("source-question relation has invalid responsiveness")
    responsiveness = judgement["responsiveness"]
    attribution = source_proposition["attribution"]
    modality = source_proposition["modality"]
    polarity = source_proposition["polarity"]
    role = source_proposition["narrative_role"]

    if attribution == "reported_speech":
        scope = "quoted_instruction"
    elif attribution == "reported_belief":
        scope = "actual_report_of_opinion"
    elif modality in {"conditional", "hypothetical"}:
        scope = "hypothetical_or_conditional"
    elif polarity == "negated":
        scope = "negated_actual_state"
    elif attribution == "unknown" or role == "unknown":
        scope = "unknown"
    else:
        scope = "actual_event_or_state"

    correction = None
    if (
        attribution in {"reported_speech", "reported_belief"}
        and question_target == "described_event_or_state"
        and responsiveness == "answers_question"
    ):
        responsiveness = "does_not_answer"
        correction = "reported_words_do_not_directly_establish_event"
    elif question_target == "unknown" and responsiveness == "answers_question":
        responsiveness = "uncertain"
        correction = "question_intent_unresolved"
    elif scope == "unknown" and responsiveness == "answers_question":
        responsiveness = "uncertain"
        correction = "unresolved_source_proposition"

    return {
        "question": question,
        "answer_quote": quote,
        "citation": citation,
        "responsiveness": responsiveness,
        "scope": scope,
        "narrative_role": role,
        "question_target": question_target,
        "scope_correction": correction,
        "source_proposition": source_proposition,
        "diagnostic_only": True,
        "production_approved": False,
    }
