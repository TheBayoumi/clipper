"""Source-bound claim review records for contextual clip headlines.

This module validates provenance and coverage, not natural-language entailment.
An automated model's labels are proposals until independently qualified. Human
attestation is recorded separately from technical or publication approval.
"""

from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"\w+", re.UNICODE)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_KINDS = {
    "actor_action",
    "attribution",
    "event_modality",
    "condition",
    "relationship_role",
    "quantity_outcome",
    "setting_time",
}
_STATUSES = {"actual_event", "future_plan", "hypothetical", "quoted_instruction", "not_applicable"}
_REPORTING_STATUSES = {"actual_report", "future_report", "hypothetical_report", "no_report"}
_VERDICTS = {"supported", "unsupported", "uncertain"}


def create_claim_review_packet(
    *,
    headline: str,
    selected_units: list[str],
    excluded_before: list[str],
    excluded_after: list[str],
    reviewed_spans: dict[str, Any],
    source_video_id: str,
    source_sha256: str,
    transcript_sha256: str,
) -> dict[str, Any]:
    """Preserve exact delivered setup/payoff and issue an unapproved claim packet."""
    if (
        not isinstance(headline, str)
        or not headline.strip()
        or not isinstance(selected_units, list)
        or not selected_units
        or any(not isinstance(unit, str) or not unit.strip() for unit in selected_units)
        or not isinstance(excluded_before, list)
        or not isinstance(excluded_after, list)
        or not isinstance(source_video_id, str)
        or not _VIDEO_ID.fullmatch(source_video_id)
        or not isinstance(source_sha256, str)
        or not _SHA256.fullmatch(source_sha256)
        or not isinstance(transcript_sha256, str)
        or not _SHA256.fullmatch(transcript_sha256)
        or any(
            not isinstance(unit, str) or not unit.strip()
            for unit in (*excluded_before, *excluded_after)
        )
    ):
        raise ValueError("claim packet needs exact delivered speech and source identity")
    if not isinstance(reviewed_spans, dict) or set(reviewed_spans) != {
        "setup_quote",
        "resolution_quote",
    }:
        raise ValueError("claim packet needs reviewed setup and resolution")
    canonical: dict[str, dict[str, Any]] = {}
    for name, span in reviewed_spans.items():
        if not isinstance(span, dict) or set(span) != {"text", "first_unit", "last_unit"}:
            raise ValueError("claim packet has an invalid reviewed source span")
        first, last, source_text = _source_range(
            selected_units,
            {"first_unit": span["first_unit"], "last_unit": span["last_unit"]},
        )
        if not isinstance(span["text"], str) or not span["text"].strip():
            raise ValueError("claim packet has an empty reviewed source span")
        if span["text"] not in source_text:
            raise ValueError("claim packet reviewed quote is absent from cited source units")
        canonical[name] = {"text": span["text"], "first_unit": first, "last_unit": last}
    return {
        "schema": "clipper-headline-claim-packet-v1",
        "source_video_id": source_video_id,
        "source_sha256": source_sha256,
        "transcript_sha256": transcript_sha256,
        "headline": headline,
        "selected_source_units": [
            {"id": index, "text": unit} for index, unit in enumerate(selected_units)
        ],
        "excluded_before_context_only": excluded_before,
        "excluded_after_context_only": excluded_after,
        "reviewed_spans": canonical,
        "required_review": {
            "whole_headline": "unreviewed",
            "atomic_claims": "unreviewed",
            "attribution_and_modality": "unreviewed",
            "conditions_and_roles": "unreviewed",
        },
        "production_approved": False,
    }


def _source_range(units: list[str], value: Any) -> tuple[int, int, str]:
    if not isinstance(value, dict) or set(value) != {"first_unit", "last_unit"}:
        raise ValueError("claim review needs an exact source-unit range")
    first, last = value["first_unit"], value["last_unit"]
    if type(first) is not int or type(last) is not int or not 0 <= first <= last < len(units):
        raise ValueError("claim review source-unit range is invalid")
    return first, last, " ".join(units[first : last + 1])


def validate_claim_review(
    record: dict[str, Any],
    *,
    source_units: list[str],
    source_video_id: str,
    source_sha256: str,
    transcript_sha256: str,
    expected_headline: str,
) -> dict[str, Any]:
    """Validate an editorial attestation without inventing source support.

    Every headline word needs an atomic claim, and a separate whole-headline
    check guards relations omitted by decomposition. Context is expanded by
    Python around each citation so a model cannot hide adjacent qualifications.
    Even a structurally complete automated record never authorizes production.
    """
    if (
        not isinstance(source_units, list)
        or not source_units
        or any(not isinstance(unit, str) or not unit.strip() for unit in source_units)
        or not isinstance(source_video_id, str)
        or not _VIDEO_ID.fullmatch(source_video_id)
        or not isinstance(source_sha256, str)
        or not _SHA256.fullmatch(source_sha256)
        or not isinstance(transcript_sha256, str)
        or not _SHA256.fullmatch(transcript_sha256)
    ):
        raise ValueError("claim review needs exact nonempty source provenance")
    if not isinstance(record, dict) or set(record) != {
        "schema",
        "source_video_id",
        "source_sha256",
        "transcript_sha256",
        "headline",
        "reviewer",
        "whole_headline",
        "claims",
    }:
        raise ValueError("claim review has missing or unknown fields")
    if record["schema"] != "clipper-headline-claim-review-v1" or any(
        record[key] != expected
        for key, expected in (
            ("source_video_id", source_video_id),
            ("source_sha256", source_sha256),
            ("transcript_sha256", transcript_sha256),
        )
    ):
        raise ValueError("claim review source identity does not match")
    headline = record["headline"]
    if not isinstance(headline, str) or not headline.strip() or not _WORD.search(headline):
        raise ValueError("claim review needs a factual headline")
    if headline != expected_headline:
        raise ValueError("claim review headline differs from the draft being reviewed")
    reviewer = record["reviewer"]
    if (
        not isinstance(reviewer, dict)
        or set(reviewer) != {"kind", "identifier"}
        or not isinstance(reviewer["kind"], str)
        or reviewer["kind"] not in {"human", "automated"}
        or not isinstance(reviewer["identifier"], str)
        or not reviewer["identifier"].strip()
    ):
        raise ValueError("claim review needs an identified reviewer")

    def assess(value: Any) -> dict[str, Any]:
        required = {
            "verdict",
            "claimed_reporting_status",
            "source_reporting_status",
            "claimed_event_status",
            "source_event_status",
            "evidence",
            "reason",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("claim review has an invalid assertion contract")
        if (
            not isinstance(value["verdict"], str)
            or value["verdict"] not in _VERDICTS
            or not isinstance(value["claimed_reporting_status"], str)
            or value["claimed_reporting_status"] not in _REPORTING_STATUSES
            or not isinstance(value["source_reporting_status"], str)
            or value["source_reporting_status"] not in _REPORTING_STATUSES
            or not isinstance(value["claimed_event_status"], str)
            or value["claimed_event_status"] not in _STATUSES
            or not isinstance(value["source_event_status"], str)
            or value["source_event_status"] not in _STATUSES
            or not isinstance(value["reason"], str)
            or not value["reason"].strip()
        ):
            raise ValueError("claim review assertion labels are invalid")
        first, last, cited = _source_range(source_units, value["evidence"])
        if last - first > 11:
            raise ValueError("claim review citation is unfocused")
        if value["verdict"] == "supported" and (
            value["source_reporting_status"] != value["claimed_reporting_status"]
            or value["source_event_status"] != value["claimed_event_status"]
            or (
                # A reported state/opinion need not describe an event; an
                # unreported non-event cannot attest to a factual claim.
                value["source_event_status"] == "not_applicable"
                and value["source_reporting_status"] == "no_report"
            )
        ):
            raise ValueError("supported claim has mismatched reporting or event status")
        context_first, context_last = max(0, first - 2), min(len(source_units) - 1, last + 2)
        return {
            **value,
            "source_text": cited,
            "context": {
                "first_unit": context_first,
                "last_unit": context_last,
                "text": " ".join(source_units[context_first : context_last + 1]),
            },
        }

    whole = assess(record["whole_headline"])
    raw_claims = record["claims"]
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("claim review needs at least one atomic claim")
    coverage = [False] * len(headline)
    claims: list[dict[str, Any]] = []
    for raw in raw_claims:
        if not isinstance(raw, dict) or set(raw) != {
            "kind",
            "first_char",
            "last_char",
            "assessment",
        }:
            raise ValueError("claim review atomic claim has invalid fields")
        first, last = raw["first_char"], raw["last_char"]
        if (
            not isinstance(raw["kind"], str)
            or raw["kind"] not in _KINDS
            or type(first) is not int
            or type(last) is not int
            or not 0 <= first < last <= len(headline)
            or not _WORD.search(headline[first:last])
        ):
            raise ValueError("claim review atomic claim is not bound to headline words")
        for index in range(first, last):
            coverage[index] = True
        claims.append(
            {
                "kind": raw["kind"],
                "headline_text": headline[first:last],
                "first_char": first,
                "last_char": last,
                "assessment": assess(raw["assessment"]),
            }
        )
    uncovered = [
        match.group()
        for match in _WORD.finditer(headline)
        if not all(coverage[match.start() : match.end()])
    ]
    all_supported = (
        not uncovered
        and whole["verdict"] == "supported"
        and all(claim["assessment"]["verdict"] == "supported" for claim in claims)
    )
    return {
        "source_video_id": source_video_id,
        "source_sha256": source_sha256,
        "transcript_sha256": transcript_sha256,
        "headline": headline,
        "whole_headline": whole,
        "claims": claims,
        "uncovered_headline_words": uncovered,
        "all_claims_supported": all_supported,
        "review_status": (
            "human_attested_not_publication_approved"
            if all_supported and reviewer["kind"] == "human"
            else "needs_human_review"
        ),
        "production_approved": False,
    }
