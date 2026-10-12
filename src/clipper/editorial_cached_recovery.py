"""Conservative cached-evidence recovery for private, unapproved clip drafts.

Never reclassify a rejected continuation or approve paraphrased claims. Derive
hooks only from exact delivered speech in previously completed source reviews.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .models import ClipCandidate, TranscriptSegment
from .tiktok import persistent_hook_layout

_WORD = re.compile(r"[A-Za-z0-9']+")
_DEPENDENT_OPENING = re.compile(
    r"^(?:so|that|this|he|she|they|it|then|like|and|well|yeah|uh|there|them|know|his|of)\b",
    re.IGNORECASE,
)


def _literal_payoff_title(text: str) -> str | None:
    """Use a standalone utterance, never paraphrase an uncertain model headline."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return None
    # A completed proposition before an explanation is still an exact source
    # quote; do not silently remove negation, conditionality or the subject.
    for marker in (" because ",):
        if marker in clean:
            clean = clean.split(marker, 1)[0].strip()
    if not (5 <= len(_WORD.findall(clean)) <= 12 and len(clean) <= 66):
        return None
    if _DEPENDENT_OPENING.match(clean):
        return None
    try:
        persistent_hook_layout(clean)
    except (OSError, ValueError):
        return None
    return clean


def recover_source_quote_drafts(
    assessments: list[dict[str, Any]],
    segments: Sequence[TranscriptSegment],
    *,
    video_id: str,
    min_duration: float,
    max_duration: float,
) -> tuple[list[ClipCandidate], dict[str, Any]]:
    """Salvage only explicit, already complete source speeches for private review.

    A missing contrast still requires a new explicit boundary judgment. Every
    rescued title is a verbatim source quote, never a failed LLM paraphrase.
    Only previously reviewed span flags can authorize this *draft* path; these
    drafts remain unverified factual claims, prohibited from publication.
    """
    result: list[ClipCandidate] = []
    seen_windows: set[tuple[float, float]] = set()
    diagnostics: list[dict[str, Any]] = []
    reasons = Counter[str]()
    for item in assessments:
        rejection = item.get("rejection")
        if rejection not in {"SPAN_REVIEW_REJECTED", "HEADLINE_REVIEW_REJECTED"}:
            continue
        review = item.get("exchange_review")
        if not isinstance(review, dict):
            reasons["missing_review"] += 1
            continue
        if rejection == "SPAN_REVIEW_REJECTED":
            relation = (review.get("continuation_review") or {}).get("relation")
            reasons[f"boundary_requires_independent_review:{relation}"] += 1
            diagnostics.append(
                {
                    "proposal_start": item.get("proposal_start"),
                    "proposal_end": item.get("proposal_end"),
                    "reason": "boundary_requires_independent_review",
                    "continuation_relation": relation,
                    "draft_approved": False,
                }
            )
            continue
        audits = review.get("headline_audits")
        if (
            not isinstance(audits, list)
            or not audits
            or any(
                not isinstance(audit, dict)
                or audit.get("verdict") == "unsupported"
                or any(
                    audit.get(key) == "unsupported"
                    for key in (
                        "actor_action",
                        "relationship_role",
                        "setting_time",
                        "quantities_outcomes",
                    )
                )
                for audit in audits
            )
            or any(
                review.get(key) is not True
                for key in ("opening_standalone", "payoff_complete", "ending_complete")
            )
            or review.get("contains_promotion_or_intro") is not False
            or not isinstance(review.get("delivered_units"), list)
            or not review["delivered_units"]
        ):
            reasons["failed_safety_or_audit"] += 1
            continue
        title = _literal_payoff_title(str(review["delivered_units"][-1]))
        if title is None:
            reasons["no_standalone_literal_payoff"] += 1
            continue
        if re.search(
            r"\b(?:touchdown|scored|score|won|defeated|gain)\b", title, re.IGNORECASE
        ) and re.search(
            r"\b(?:let's|imagine|suppose|pretend|what if|hypothetically)\b",
            " ".join(review["delivered_units"]),
            re.IGNORECASE,
        ):
            reasons["hypothetical_outcome_not_established"] += 1
            continue
        span = review.get("source_quote_spans")
        decision = item.get("decision")
        if (
            not isinstance(span, dict)
            or not isinstance(span.get("setup_quote"), dict)
            or not isinstance(span.get("resolution_quote"), dict)
            or not isinstance(decision, dict)
            or any(
                type(decision.get(dim)) is not int or not 3 <= decision[dim] <= 5
                for dim in ("opening", "story", "ending")
            )
        ):
            reasons["missing_canonical_evidence"] += 1
            continue
        source = " ".join(review["delivered_units"])
        if title.casefold() not in source.casefold():
            reasons["literal_quote_not_in_source"] += 1
            continue
        start, end = item.get("reviewed_start"), item.get("reviewed_end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            reasons["invalid_reviewed_window"] += 1
            continue
        if not min_duration <= end - start <= max_duration:
            reasons["invalid_reviewed_window"] += 1
            continue
        if (float(start), float(end)) in seen_windows:
            reasons["duplicate_reviewed_window"] += 1
            continue
        timed = " ".join(
            segment.text for segment in segments if segment.end > start and segment.start < end
        )
        if title.casefold() not in timed.casefold():
            reasons["not_in_timed_source"] += 1
            continue
        setup = span["setup_quote"]["text"]
        payoff = span["resolution_quote"]["text"]
        if (
            type(setup) is not str
            or type(payoff) is not str
            or not setup.strip()
            or not payoff.strip()
            or setup not in source
            or payoff not in source
        ):
            reasons["unmatched_setup_or_resolution"] += 1
            continue
        reason_fields = (
            f"semantic_hook={title.upper()}",
            "context_assessment=structured_local_instruct_model",
            *(
                f"context_{name}_margin={(decision[name] / 5 * 4 - 2):.6f}"
                for name in ("opening", "story", "ending")
            ),
            "headline_origin=verbatim_cached_source_quote_unapproved",
            "span_review=standalone_opening_delivered_payoff_complete_ending",
            f"setup_quote={setup}",
            f"payoff_quote={payoff}",
            "cached_recovery=verbatim_quote_private_review_only",
        )
        seen_windows.add((float(start), float(end)))
        result.append(
            ClipCandidate(
                video_id,
                float(start),
                float(end),
                source,
                sum(decision[key] for key in ("opening", "story", "ending")) * 100 / 15,
                reason_fields,
            )
        )
        diagnostics.append(
            {
                "proposal_start": item.get("proposal_start"),
                "proposal_end": item.get("proposal_end"),
                "reviewed_start": start,
                "reviewed_end": end,
                "exact_spoken_hook": title,
                "draft_approved": False,
                "reason": "source_quote_private_draft_not_factual_approval",
            }
        )
        reasons["source_quote_private_draft"] += 1
    return result, {
        "schema": "clipper-cached-recovery-v1",
        "selector_reassessment_count": 0,
        "boundary_reassessment_count": 0,
        "headline_model_reassessment_count": 0,
        "unapproved_draft_count": len(result),
        "breakdown": dict(reasons),
        "records": diagnostics,
        "automatic_publication_allowed": False,
    }
