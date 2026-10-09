"""Parser-owned headline obligations that a reviewer cannot silently omit.

These are syntactic obligations, not complete semantic claims. A parser warning
or uncovered content prevents even a diagnostic source-review manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_VERDICTS = {"supported", "unsupported", "uncertain"}


def build_syntax_obligations(inventory: dict[str, Any]) -> dict[str, Any]:
    """Enumerate every parsed predicate and role with immutable source-blind IDs."""
    if (
        not isinstance(inventory, dict)
        or not isinstance(inventory.get("headline"), str)
        or not inventory["headline"].strip()
        or not isinstance(inventory.get("frames"), list)
        or not inventory["frames"]
        or not isinstance(inventory.get("parse_warnings"), list)
        or not isinstance(inventory.get("uncovered_content_tokens"), list)
        or inventory.get("source_entailment_checked") is not False
        or inventory.get("production_approved") is not False
    ):
        raise ValueError("syntax obligations need an unapproved parser inventory")
    headline = inventory["headline"]
    obligations: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        if item["obligation_id"] in seen:
            raise ValueError("syntax inventory contains duplicate predicate or role IDs")
        seen.add(item["obligation_id"])
        obligations.append(item)

    for frame in inventory["frames"]:
        if not isinstance(frame, dict) or not isinstance(frame.get("roles"), list):
            raise ValueError("syntax inventory contains an invalid frame")
        token, first, last, predicate = (
            frame.get("predicate_token"),
            frame.get("predicate_first_char"),
            frame.get("predicate_last_char"),
            frame.get("predicate"),
        )
        if (
            type(token) is not int
            or token < 0
            or type(first) is not int
            or type(last) is not int
            or not 0 <= first < last <= len(headline)
            or not isinstance(predicate, str)
            or headline[first:last] != predicate
        ):
            raise ValueError("syntax inventory predicate differs from headline")
        add(
            {
                "obligation_id": f"predicate:{token}",
                "kind": "predicate",
                "predicate_token": token,
                "first_char": first,
                "last_char": last,
                "text": predicate,
            }
        )
        for role in frame["roles"]:
            if not isinstance(role, dict):
                raise ValueError("syntax inventory contains an invalid role")
            head, role_first, role_last, text = (
                role.get("head_token"),
                role.get("first_char"),
                role.get("last_char"),
                role.get("text"),
            )
            if (
                type(head) is not int
                or head < 0
                or type(role_first) is not int
                or type(role_last) is not int
                or not 0 <= role_first < role_last <= len(headline)
                or not isinstance(text, str)
                or headline[role_first:role_last] != text
                or not isinstance(role.get("role"), str)
                or not role["role"]
            ):
                raise ValueError("syntax inventory role differs from headline")
            add(
                {
                    "obligation_id": f"role:{token}:{head}",
                    "kind": role["role"],
                    "predicate_token": token,
                    "first_char": role_first,
                    "last_char": role_last,
                    "text": text,
                }
            )
    return {
        "headline": headline,
        "obligations": obligations,
        "parse_warnings": inventory["parse_warnings"],
        "uncovered_content_tokens": inventory["uncovered_content_tokens"],
        "ready_for_source_review": not inventory["parse_warnings"]
        and not inventory["uncovered_content_tokens"],
        "claim_inventory_semantically_qualified": False,
        "source_entailment_qualified": False,
        "diagnostic_only": True,
        "production_approved": False,
    }


def validate_obligation_reviews(
    manifest: dict[str, Any],
    reviews: list[dict[str, Any]],
    delivered_units: Sequence[str],
) -> dict[str, Any]:
    """Require one cited review per parser obligation; never certify its truth."""
    if (
        not isinstance(manifest, dict)
        or manifest.get("ready_for_source_review") is not True
        or manifest.get("production_approved") is not False
        or not isinstance(manifest.get("obligations"), list)
        or not manifest["obligations"]
    ):
        raise ValueError("obligation review needs a warning-free unapproved manifest")
    if not delivered_units or any(
        not isinstance(unit, str) or not unit.strip() for unit in delivered_units
    ):
        raise ValueError("obligation review needs delivered source speech")
    if not isinstance(reviews, list) or len(reviews) != len(manifest["obligations"]):
        raise ValueError("obligation review omitted or added an obligation")
    expected = {item["obligation_id"] for item in manifest["obligations"]}
    if len(expected) != len(manifest["obligations"]):
        raise ValueError("obligation manifest repeats an ID")
    seen: set[str] = set()
    validated = []
    for review in reviews:
        if not isinstance(review, dict) or set(review) != {
            "obligation_id",
            "verdict",
            "first_unit",
            "last_unit",
        }:
            raise ValueError("obligation review has missing or unknown fields")
        identifier, verdict = review["obligation_id"], review["verdict"]
        first, last = review["first_unit"], review["last_unit"]
        if (
            not isinstance(identifier, str)
            or identifier not in expected
            or identifier in seen
            or not isinstance(verdict, str)
            or verdict not in _VERDICTS
            or type(first) is not int
            or type(last) is not int
            or not (first == last == -1 or 0 <= first <= last < len(delivered_units))
            or (first >= 0 and last - first > 11)
            or (verdict == "supported" and first == -1)
        ):
            raise ValueError("obligation review has an invalid ID, verdict or citation")
        seen.add(identifier)
        validated.append(
            {
                **review,
                "source_text": " ".join(delivered_units[first : last + 1]) if first >= 0 else "",
            }
        )
    return {
        "headline": manifest["headline"],
        "reviews": validated,
        "all_recorded_obligations_labeled_supported": all(
            item["verdict"] == "supported" for item in validated
        ),
        "claim_inventory_semantically_qualified": False,
        "source_entailment_qualified": False,
        "diagnostic_only": True,
        "production_approved": False,
    }
