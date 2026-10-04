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
        "event. If the referent or speech-act boundary cannot be resolved from these "
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
        },
        80,
    )
    if not isinstance(proposed, dict) or set(proposed) != {"responsiveness", "scope"}:
        raise ValueError("source-scope response has missing or unknown fields")
    if proposed["responsiveness"] not in _RESPONSIVENESS or proposed["scope"] not in _SCOPES:
        raise ValueError("source-scope response has invalid labels")
    return {
        "question": question,
        "answer_quote": quote,
        "citation": citation,
        **proposed,
        "diagnostic_only": True,
        "production_approved": False,
    }
