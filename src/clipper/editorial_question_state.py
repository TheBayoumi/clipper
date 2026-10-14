"""Diagnostic, source-positioned answers to explicit dialogue questions.

Question marks in ASR are only candidate questions, and a cited answer is not
proof that the answer is semantically responsive. This module cannot approve a
cut; it exposes the discourse state that a later qualified gate must judge.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

Completion = Callable[[str, dict[str, Any], dict[str, Any], int], dict[str, Any]]
_STATUSES = {"answered", "unanswered", "uncertain"}


def _answer(
    completion: Completion,
    delivered_units: Sequence[str],
    excluded_after: Sequence[str],
    question_id: int,
    *,
    region: str,
) -> dict[str, Any]:
    if region == "delivered":
        candidate_ids = list(range(question_id + 1, len(delivered_units)))
        source_units = delivered_units
    elif region == "excluded":
        candidate_ids = list(range(len(excluded_after)))
        source_units = excluded_after
    else:
        raise ValueError("answer region must be delivered or excluded")
    response = completion(
        "Decide whether the candidate speech actually ANSWERS the exact question, "
        "including its requested actor, action, quantity, polarity and conditions. "
        "A related topic, partial premise, or acknowledgement is not an answer. "
        "Choose uncertain if reference or intent is ambiguous. Cite the shortest "
        "continuous answer unit range when answered; otherwise set both positions "
        "to -1. Return only output_schema JSON, without prose.",
        {
            "question": {"id": question_id, "text": delivered_units[question_id]},
            "delivered_context": [
                {"id": index, "text": unit} for index, unit in enumerate(delivered_units)
            ],
            "candidate_region": region,
            "candidate_units": [
                {"id": index, "text": source_units[index]} for index in candidate_ids
            ],
        },
        {
            "status": {"type": "string", "enum": sorted(_STATUSES)},
            "first_unit": {"type": "integer", "enum": [-1, *candidate_ids]},
            "last_unit": {"type": "integer", "enum": [-1, *candidate_ids]},
        },
        64,
    )
    if not isinstance(response, dict) or set(response) != {
        "status",
        "first_unit",
        "last_unit",
    }:
        raise ValueError("question answer has missing or extra fields")
    status, first, last = (
        response["status"],
        response["first_unit"],
        response["last_unit"],
    )
    if (
        not isinstance(status, str)
        or status not in _STATUSES
        or type(first) is not int
        or type(last) is not int
    ):
        raise ValueError("question answer has an invalid status or position")
    if status == "answered":
        if not candidate_ids or not candidate_ids[0] <= first <= last <= candidate_ids[-1]:
            raise ValueError("answered question needs a cited candidate source range")
        evidence: dict[str, Any] | None = {
            "first_unit": first,
            "last_unit": last,
            "text": " ".join(source_units[first : last + 1]),
        }
    else:
        if first != -1 or last != -1:
            raise ValueError("unanswered or uncertain question cannot cite an answer")
        evidence = None
    return {"status": status, "answer_source": evidence}


def audit_explicit_question_state(
    delivered_units: Sequence[str],
    excluded_after: Sequence[str],
    completion: Completion,
) -> dict[str, Any]:
    """Inventory every explicit question before asking where its answer occurs.

    Delivered-answer assessment is made without exposing excluded speech as a
    candidate. Only a delivered-unanswered question is then tested against
    excluded speech. This is a diagnostic signal, never cut approval.
    """
    if (
        not delivered_units
        or not excluded_after
        or any(not isinstance(unit, str) or not unit.strip() for unit in delivered_units)
        or any(not isinstance(unit, str) or not unit.strip() for unit in excluded_after)
    ):
        raise ValueError("question state needs delivered and excluded source speech")
    records: list[dict[str, Any]] = []
    for question_id, unit in enumerate(delivered_units):
        if "?" not in unit:
            continue
        delivered = _answer(
            completion, delivered_units, excluded_after, question_id, region="delivered"
        )
        excluded: dict[str, Any] | None = None
        if delivered["status"] == "unanswered":
            excluded = _answer(
                completion, delivered_units, excluded_after, question_id, region="excluded"
            )
        records.append(
            {
                "question_source": {"unit_id": question_id, "text": unit},
                "delivered": delivered,
                "excluded": excluded,
                "missing_answer_signal": (
                    delivered["status"] == "unanswered"
                    and excluded is not None
                    and excluded["status"] == "answered"
                ),
            }
        )
    return {
        "question_inventory": records,
        "missing_answer_question_ids": [
            record["question_source"]["unit_id"]
            for record in records
            if record["missing_answer_signal"]
        ],
        "inventory_scope": "explicit_transcript_question_marks_only",
        "diagnostic_only": True,
        "production_approved": False,
    }
