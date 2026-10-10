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
    proposed = completion(
        "Read the complete delivered source speech and the cited answer_quote. Judge "
        "whether that quote actually answers the question with the same actor, role, "
        "event, time and referent. A substring can be exact yet nonresponsive. "
        "Distinguish an actual setting from words quoted as a warning or instruction, "
        "and a real outcome from an if/would/could hypothetical. A speaker may actually "
        "report an opinion without the embedded opinion being an independently verified "
        "event. A guest can recount real experiences from years before the recording: "
        "classify these as recounted_event, not a present event and not hypothetical. "
        "Quoted dialogue inside an anecdote is a reported_utterance, even when it "
        "asserts that something occurred; it does not by itself validate that event. "
        "Track who speaks, who acts, what scene they describe, and whether the "
        "answer quote refers to that scene. Resolve meaning across contiguous "
        "speaker turns: a direct first-person report followed by an interlocutor's "
        "generic 'you' restatement may still describe the first speaker. A negative "
        "answer to a factual or yes/no question is still responsive; distinguish "
        "a negated actual state from an affirmative outcome or a hypothetical one. "
        "Classify narrative_role by the role the cited answer plays in the source, "
        "independent of any headline. "
        "If the referent or speech-act boundary cannot be resolved from these "
        "units, choose uncertain or unknown. Do not infer a headline or a proposed "
        "answer; neither is supplied. Return output_schema JSON only.",
        {
            "question": question,
            "answer_quote": quote,
            "cited_first_unit": first,
            "cited_last_unit": last,
            "source_units": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
        },
        {
            "responsiveness": {"type": "string", "enum": list(_RESPONSIVENESS)},
            "scope": {"type": "string", "enum": list(_SCOPES)},
            "narrative_role": {"type": "string", "enum": list(NARRATIVE_ROLES)},
        },
        80,
    )
    if not isinstance(proposed, dict) or set(proposed) != {
        "responsiveness",
        "scope",
        "narrative_role",
    }:
        raise ValueError("source-scope response has missing or unknown fields")
    if (
        proposed["responsiveness"] not in _RESPONSIVENESS
        or proposed["scope"] not in _SCOPES
        or proposed["narrative_role"] not in NARRATIVE_ROLES
    ):
        raise ValueError("source-scope response has invalid labels")
    # A quoted statement may correctly answer "what did X say?" but cannot,
    # merely by being quoted, establish the historical event or scene it claims.
    # Classify the question separately without the transcript or quoted answer: a
    # model must not justify a nonresponsive quote by treating the *question* as
    # if it had asked about speech. Use this second call only at the high-risk
    # speech-act boundary; all other scope assessments keep their original path.
    question_target = None
    correction = None
    if proposed["responsiveness"] == "answers_question" and proposed["narrative_role"] in {
        "reported_utterance",
        "reported_belief",
    }:
        classification = completion(
            "Interpret only the question, independently of any source transcript or "
            "proposed answer. Choose described_event_or_state when asking what really "
            "happened, the actual occasion/setting, or a person's actual action or "
            "state, even if the narrative is about the speaker's past. Choose "
            "reported_words_or_belief only when the question itself asks what "
            "someone said, claimed, was told, thought, or believed. Choose "
            "conditional_or_hypothetical when the question asks what might, could "
            "or would have happened. Do not assume quoted speech establishes an "
            "event. Return output_schema JSON only.",
            {"question": question},
            {"question_target": {"type": "string", "enum": list(_QUESTION_TARGETS)}},
            80,
        )
        if (
            not isinstance(classification, dict)
            or set(classification) != {"question_target"}
            or classification["question_target"] not in _QUESTION_TARGETS
        ):
            raise ValueError("question-only target classifier returned invalid fields")
        question_target = classification["question_target"]
        if question_target == "described_event_or_state":
            # This says only that the chosen quotation does not directly prove
            # the event being asked about. Source answer generation may recover
            # by finding *direct event evidence* in other original speech units.
            proposed = {**proposed, "responsiveness": "does_not_answer"}
            correction = "reported_words_do_not_directly_establish_event"
        elif question_target == "unknown":
            proposed = {**proposed, "responsiveness": "uncertain"}
            correction = "question_intent_unresolved"
    return {
        "question": question,
        "answer_quote": quote,
        "citation": citation,
        **proposed,
        "question_target": question_target,
        "scope_correction": correction,
        "diagnostic_only": True,
        "production_approved": False,
    }
