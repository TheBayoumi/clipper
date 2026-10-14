"""Diagnostic claim-level reviewer using independently scoped source evidence.

The model proposes word spans and source positions; Python binds both to the
immutable headline and transcript, then validates the review contract. This is
an experiment, never an automatic production approval.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from clipper.editorial_review import validate_claim_review

_WORD = re.compile(r"\w+", re.UNICODE)
_KINDS = (
    "actor_action",
    "attribution",
    "event_modality",
    "condition",
    "relationship_role",
    "quantity_outcome",
    "setting_time",
)
_REPORTING = ("actual_report", "future_report", "hypothetical_report", "no_report")
_EVENT = ("actual_event", "future_plan", "hypothetical", "quoted_instruction", "not_applicable")
_VERDICT = ("supported", "unsupported", "uncertain")


class ClaimReviewer(Protocol):
    def _review_completion(
        self, prompt: str, payload: dict[str, Any], properties: dict[str, Any], tokens: int
    ) -> dict[str, Any]: ...


def audit_structured_claims(
    reviewer: ClaimReviewer,
    headline: str,
    source_units: list[str],
    *,
    source_video_id: str,
    source_sha256: str,
    transcript_sha256: str,
) -> dict[str, Any]:
    """Return an auditable, unapproved assessment of the exact proposed headline."""
    words = list(_WORD.finditer(headline))
    if not words or not source_units or any(not unit.strip() for unit in source_units):
        raise ValueError("structured claims require a headline and delivered source speech")
    word_index = [{"id": i, "text": match.group()} for i, match in enumerate(words)]
    source = [{"id": i, "text": unit} for i, unit in enumerate(source_units)]
    extraction = reviewer._review_completion(
        "Split the headline into its smallest checkable factual relations using the "
        "numbered word_index. Include actor/action, attribution, relationship, setting, "
        "quantity, negation, condition and temporal status when claimed. Overlapping "
        "claims are allowed, but every headline word must be covered. Do not inspect "
        "source speech. Return output_schema JSON.",
        {"headline": headline, "word_index": word_index},
        {
            "claims": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(_KINDS)},
                        "first_word": {"type": "integer", "minimum": 0},
                        "last_word": {"type": "integer", "minimum": 0},
                    },
                    "required": ["kind", "first_word", "last_word"],
                    "additionalProperties": False,
                },
            }
        },
        256,
    )
    claims = extraction.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 8:
        raise RuntimeError("structured review omitted atomic headline claims")
    covered = [False] * len(words)
    assertions: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"kind", "first_word", "last_word"}:
            raise RuntimeError("structured review returned an invalid word-span claim")
        first, last = claim["first_word"], claim["last_word"]
        if (
            claim["kind"] not in _KINDS
            or type(first) is not int
            or type(last) is not int
            or not 0 <= first <= last < len(words)
        ):
            raise RuntimeError("structured review claim is outside the exact headline")
        for index in range(first, last + 1):
            covered[index] = True
        assertions.append(
            {
                "kind": claim["kind"],
                "first_char": words[first].start(),
                "last_char": words[last].end(),
            }
        )
    if not all(covered):
        raise RuntimeError("structured review omitted headline words")

    def assess(claim_text: str) -> dict[str, Any]:
        proposed = reviewer._review_completion(
            "Judge this exact claim against the delivered source speech. Cite a continuous "
            "source unit range containing the actors and qualifying words needed for the "
            "claim; do not cite an isolated pronoun without its antecedent. Distinguish "
            "the act of reporting from the event or state being reported. Mark supported "
            "only if all relations, polarity and conditions are established. Do not infer "
            "an actual event from a hypothetical or quoted instruction. A headline saying "
            "'the speaker says X' claims an actual_report even if X is hypothetical. "
            "Return positions, "
            "not copied source text, in output_schema JSON.",
            {"claim": claim_text, "source_units": source},
            {
                "verdict": {"type": "string", "enum": list(_VERDICT)},
                "claimed_reporting_status": {"type": "string", "enum": list(_REPORTING)},
                "claimed_event_status": {"type": "string", "enum": list(_EVENT)},
                "first_unit": {"type": "integer", "minimum": 0},
                "last_unit": {"type": "integer", "minimum": 0},
                "reason": {"type": "string", "minLength": 1},
            },
            192,
        )
        required = {
            "verdict",
            "claimed_reporting_status",
            "claimed_event_status",
            "first_unit",
            "last_unit",
            "reason",
        }
        if not isinstance(proposed, dict) or set(proposed) != required:
            raise RuntimeError("structured review returned an invalid claim assessment")
        first, last = proposed["first_unit"], proposed["last_unit"]
        if (
            type(first) is not int
            or type(last) is not int
            or not 0 <= first <= last < len(source_units)
            or last - first > 11
        ):
            raise RuntimeError("structured review returned an invalid source citation")
        scoped = reviewer._review_completion(
            "Classify the original source speech at the cited positions independently "
            "of any proposed headline. A speaker can actually report a hypothetical "
            "event, future plan, quoted instruction, or non-event state/opinion. Keep "
            "the reporting act separate from the embedded event status. A speaker stating "
            "a proposition in this recording is an actual_report. Inspect the "
            "complete source context for antecedents and conditions, not only the cited "
            "fragment. Return output_schema JSON.",
            {
                "source_units": source,
                "cited_first_unit": first,
                "cited_last_unit": last,
            },
            {
                "source_reporting_status": {"type": "string", "enum": list(_REPORTING)},
                "source_event_status": {"type": "string", "enum": list(_EVENT)},
                "reason": {"type": "string", "minLength": 1},
            },
            128,
        )
        if not isinstance(scoped, dict) or set(scoped) != {
            "source_reporting_status",
            "source_event_status",
            "reason",
        }:
            raise RuntimeError("structured review returned an invalid source-scope assessment")
        return {
            "verdict": proposed["verdict"],
            "claimed_reporting_status": proposed["claimed_reporting_status"],
            "source_reporting_status": scoped["source_reporting_status"],
            "claimed_event_status": proposed["claimed_event_status"],
            "source_event_status": scoped["source_event_status"],
            "evidence": {"first_unit": first, "last_unit": last},
            "reason": proposed["reason"],
        }

    record: dict[str, Any] = {
        "schema": "clipper-headline-claim-review-v1",
        "source_video_id": source_video_id,
        "source_sha256": source_sha256,
        "transcript_sha256": transcript_sha256,
        "headline": headline,
        "reviewer": {"kind": "automated", "identifier": "structured-claim-diagnostic"},
        "whole_headline": assess(headline),
        "claims": [
            {
                **claim,
                "assessment": assess(headline[claim["first_char"] : claim["last_char"]]),
            }
            for claim in assertions
        ],
    }
    validated = validate_claim_review(
        record,
        source_units=source_units,
        source_video_id=source_video_id,
        source_sha256=source_sha256,
        transcript_sha256=transcript_sha256,
        expected_headline=headline,
    )
    return {"record": record, "validated": validated, "diagnostic_only": True}
