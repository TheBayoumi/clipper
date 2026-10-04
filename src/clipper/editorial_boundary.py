"""Source-position contracts for diagnosing unfinished conversational cuts.

An exact citation is not a semantic guarantee. This module only prevents a
model from asserting a missing question, clause or contrast without citing the
corresponding delivered obligation and excluded fulfillment.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

Completion = Callable[[str, dict[str, Any], dict[str, Any], int], dict[str, Any]]
_KINDS = {"none", "question", "clause", "contrast", "uncertain"}
_CONTRAST = re.compile(
    r"\b(?:but|however|instead|yet|although|though|not|never|no|nothing|zero)\b|n't\b",
    re.IGNORECASE,
)
_DANGLING = re.compile(
    r"\b(?:because|if|when|where|while|that|who|which|although|unless|and|but|so)\s*$"
    r"|\b(?:saying|telling|asking|told|said)\s+(?:him|her|them|me|us)\s*$",
    re.IGNORECASE,
)


def _range(units: Sequence[str], pointer: Any) -> dict[str, Any] | None:
    if not isinstance(pointer, dict) or set(pointer) != {"first_unit", "last_unit"}:
        raise ValueError("cut obligation needs an exact source-unit pointer")
    first, last = pointer["first_unit"], pointer["last_unit"]
    if type(first) is not int or type(last) is not int:
        raise ValueError("cut obligation positions must be integers")
    if first == last == -1:
        return None
    if not 0 <= first <= last < len(units):
        raise ValueError("cut obligation source positions are invalid")
    return {"first_unit": first, "last_unit": last, "text": " ".join(units[first : last + 1])}


def _dangling_clause(text: str) -> bool:
    return _DANGLING.search(text.rstrip(" .,!?:;\"'")) is not None


def validate_cut_obligation(
    delivered_units: Sequence[str],
    excluded_after: Sequence[str],
    final_unit_id: int,
    proposal: Any,
) -> dict[str, Any]:
    """Reject a missing-payoff explanation without the claimed source relation."""
    if (
        not delivered_units
        or not excluded_after
        or any(not isinstance(text, str) or not text.strip() for text in delivered_units)
        or any(not isinstance(text, str) or not text.strip() for text in excluded_after)
        or type(final_unit_id) is not int
        or not 0 <= final_unit_id < len(delivered_units)
    ):
        raise ValueError("cut obligation needs exact delivered and excluded speech")
    if not isinstance(proposal, dict) or set(proposal) != {
        "kind",
        "pending",
        "fulfillment",
        "reason",
    }:
        raise ValueError("cut obligation proposal has missing or extra fields")
    kind, reason = proposal["kind"], proposal["reason"]
    if (
        not isinstance(kind, str)
        or kind not in _KINDS
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError("cut obligation kind or reason is invalid")
    pending = _range(delivered_units, proposal["pending"])
    fulfillment = _range(excluded_after, proposal["fulfillment"])
    if kind in {"none", "uncertain"}:
        if pending is not None or fulfillment is not None:
            raise ValueError("non-assertive cut verdict cannot cite a missing obligation")
        if kind == "none" and _dangling_clause(delivered_units[final_unit_id]):
            raise ValueError("complete cut cannot end on an unfinished clause")
    elif pending is None or fulfillment is None:
        raise ValueError("missing payoff needs both delivered and excluded evidence")
    elif kind == "question":
        if "?" not in pending["text"]:
            raise ValueError("missing answer requires an explicit delivered question")
    elif pending["last_unit"] != final_unit_id:
        raise ValueError("missing clause or contrast must reach the final delivered point")
    elif kind == "clause" and not _dangling_clause(delivered_units[final_unit_id]):
        raise ValueError("unfinished clause needs a dangling delivered clause")
    elif kind == "contrast" and not _CONTRAST.search(fulfillment["text"]):
        raise ValueError("missing contrast needs contrastive excluded speech")
    return {
        "kind": kind,
        "pending_source": pending,
        "fulfillment_source": fulfillment,
        "reason": reason,
        "cut_complete": kind == "none",
        "diagnostic_only": True,
        "production_approved": False,
    }


def propose_cut_obligation(
    delivered_units: Sequence[str],
    excluded_after: Sequence[str],
    final_unit_id: int,
    completion: Completion,
) -> dict[str, Any]:
    """Ask for a concrete unfinished obligation, never a free-form veto."""
    if (
        not delivered_units
        or not excluded_after
        or any(not isinstance(text, str) or not text.strip() for text in delivered_units)
        or any(not isinstance(text, str) or not text.strip() for text in excluded_after)
        or type(final_unit_id) is not int
        or not 0 <= final_unit_id < len(delivered_units)
    ):
        raise ValueError("cut obligation needs exact delivered and excluded speech")
    if _dangling_clause(delivered_units[final_unit_id]):
        return {
            "kind": "clause",
            "pending_source": {
                "first_unit": final_unit_id,
                "last_unit": final_unit_id,
                "text": delivered_units[final_unit_id],
            },
            "fulfillment_source": None,
            "reason": "Delivered final clause is syntactically unfinished.",
            "cut_complete": False,
            "decision_origin": "python_syntax_v1",
            "diagnostic_only": True,
            "production_approved": False,
        }

    def pointer(count: int) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                key: {"type": "integer", "enum": [-1, *range(count)]}
                for key in ("first_unit", "last_unit")
            },
            "required": ["first_unit", "last_unit"],
            "additionalProperties": False,
        }

    proposal = completion(
        "Inspect the delivered clip and the excluded continuation. A missing payoff "
        "requires an explicit still-open delivered question or a contrast that excluded "
        "speech actually supplies. Python has already checked for a syntactically "
        "unfinished final clause. Topic similarity or optional elaboration is not a "
        "missing payoff. Choose none when there is no concrete open obligation; do not "
        "treat ordinary uncertainty as an invented obligation. For a missing obligation "
        "cite the delivered pending range and excluded fulfilling range. For none set "
        "both ranges to -1. Use source unit IDs, never rewritten text or "
        "an explanatory essay. Python will describe the cited decision. "
        "Return output_schema JSON.",
        {
            "delivered_units": [
                {"id": index, "text": text} for index, text in enumerate(delivered_units)
            ],
            "excluded_after": [
                {"id": index, "text": text} for index, text in enumerate(excluded_after)
            ],
            "final_substantive_unit_id": final_unit_id,
            "final_substantive_unit_text": delivered_units[final_unit_id],
        },
        {
            "kind": {"type": "string", "enum": ["none", "question", "contrast"]},
            "pending": pointer(len(delivered_units)),
            "fulfillment": pointer(len(excluded_after)),
        },
        80,
    )
    if (
        not isinstance(proposal, dict)
        or not isinstance(proposal.get("kind"), str)
        or set(proposal) != {"kind", "pending", "fulfillment"}
        or proposal["kind"]
        not in {
            "none",
            "question",
            "contrast",
        }
    ):
        raise ValueError("model proposed a cut kind outside the constrained contract")
    reviewed = validate_cut_obligation(
        delivered_units,
        excluded_after,
        final_unit_id,
        {
            **proposal,
            "reason": "Model selected positions; Python checked the cited cut contract.",
        },
    )
    return {**reviewed, "decision_origin": "model_positions_python_contract_v3"}
