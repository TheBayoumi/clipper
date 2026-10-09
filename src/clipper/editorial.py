"""Evidence-labeled editorial screening for source-grounded review drafts.

The five weighted criteria use transcript *proxies*, never pretend to measure
genuine emotion, semantic payoff, or whether important action survives a crop.
Visual suitability and hook integrity require explicit review. Draft selection
is permitted while publication approval remains false.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from clipper.models import ClipCandidate, TranscriptSegment
from clipper.tiktok import (
    creative_hook_from_text,
    distinct_hook_from_text,
    source_headline_candidates,
)

RUBRIC_VERSION = "podcast-structured-v1-review-drafts"
WEIGHTS = {"opening": 25, "story": 25, "emotion": 10, "visuals": 15, "retention": 25}
_WORDS = re.compile(r"[A-Za-z0-9$']+")
_NUMBERS = re.compile(r"(?<!\w)\$?\d[\d,.]*(?:%|k|m)?\b", re.IGNORECASE)
_STOP = frozenset(
    {
        "i",
        "you",
        "he",
        "she",
        "we",
        "the",
        "a",
        "an",
        "and",
        "for",
        "is",
        "it",
        "bro",
        "like",
        "that",
        "this",
        "to",
        "in",
        "on",
        "of",
        "my",
        "your",
        "are",
        "was",
        "with",
        "have",
        "be",
        "do",
        "not",
        "just",
        "so",
    }
)
CriterionBasis = Literal[
    "transcript_proxy", "semantic_model", "machine_measured", "manual_verified", "manual_required"
]
IntegrityStatus = Literal["unverified", "pass", "fail"]


@dataclass(frozen=True, slots=True)
class CriterionRating:
    score: float | None
    basis: CriterionBasis
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.score is not None and not 0 <= self.score <= 5:
            raise ValueError("criterion ratings must be on a 0-5 scale")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EditorialReview:
    """Explicit reviewer evidence; never inferred from ASR or a filename."""

    visual_score: float | None = None
    visual_notes: str = ""
    visual_basis: Literal["machine_measured", "manual_verified"] = "manual_verified"
    integrity_passed: bool | None = None
    integrity_notes: str = ""

    def __post_init__(self) -> None:
        if self.visual_score is not None and (
            not 0 <= self.visual_score <= 5 or not self.visual_notes.strip()
        ):
            raise ValueError("visual rating needs a 0-5 score and reviewer evidence")
        if self.integrity_passed is not None and not self.integrity_notes.strip():
            raise ValueError("integrity decision requires reviewer evidence")


@dataclass(frozen=True, slots=True)
class EditorialPick:
    clip: ClipCandidate
    hook: str
    hook_score: int
    editorial_score: float
    weighted_points: float
    score_coverage: int
    criteria: dict[str, CriterionRating]
    integrity_status: IntegrityStatus
    integrity_evidence: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self.clip),
            "rubric_version": RUBRIC_VERSION,
            "weights": WEIGHTS.copy(),
            "criterion_scores": {key: value.to_dict() for key, value in self.criteria.items()},
            "hook_candidate": self.hook,
            "hook_score": self.hook_score,
            "editorial_score": self.editorial_score,
            "weighted_points": self.weighted_points,
            "score_coverage": self.score_coverage,
            "score_basis": (
                "structured_local_instruct_model_ratings_requires_source_review"
                if "context_assessment=structured_local_instruct_model" in self.clip.reasons
                else "provisional_context_similarity_ranking_not_calibrated_quality"
            ),
            "integrity_gate": {
                "status": self.integrity_status,
                "evidence": list(self.integrity_evidence),
            },
            "editorial_reasons": list(self.reasons),
            "human_review_required": (
                self.criteria["visuals"].score is None
                or self.criteria["emotion"].score is None
                or self.integrity_status != "pass"
            ),
            "publish_approved": False,
            "needs_checks": [
                "Required subjects visible; important action survives the portrait crop",
                "No embedded or newly added logos; captions match exact speech",
                "First two seconds contain a real spoken or visual hook",
                "Setup, tension and payoff survive the final cut",
                "Creative hook makes no unsupported claims; inspect source context",
                "Check campaign eligibility and source rights before publication",
            ],
        }


def _clamp(value: float) -> float:
    return round(max(0.0, min(5.0, value)), 2)


def _reason_float(candidate: ClipCandidate, prefix: str) -> float | None:
    for reason in candidate.reasons:
        if reason.startswith(prefix):
            try:
                value = float(reason.removeprefix(prefix))
                return value if math.isfinite(value) else None
            except ValueError:
                return None
    return None


def _semantic_hook_override(candidate: ClipCandidate) -> str | None:
    for reason in candidate.reasons:
        if reason.startswith("semantic_hook="):
            value = reason.removeprefix("semantic_hook=").strip()
            return value or None
    return None


def _context_rating(candidate: ClipCandidate, dimension: str) -> CriterionRating:
    margin = _reason_float(candidate, f"context_{dimension}_margin=")
    if margin is None or not -2 <= margin <= 2:
        return CriterionRating(
            None, "manual_required", ("context assessment is missing or invalid",)
        )
    structured = "context_assessment=structured_local_instruct_model" in candidate.reasons
    return CriterionRating(
        _clamp((margin + 2) / 4 * 5),
        "semantic_model",
        (
            f"context_{dimension}_contrast={margin:.6f}",
            (
                "structured local instruct-model assessment; source review still required"
                if structured
                else "embedding contrast is ranking evidence, not calibrated editorial quality"
            ),
            *(reason for reason in candidate.reasons if reason.startswith("context_evidence=")),
        ),
    )


def _grounded_numbers(hook: str, source: str) -> bool:
    def normalize(text: str) -> set[str]:
        return {re.sub(r"[,$]", "", match).lower() for match in _NUMBERS.findall(text)}

    return normalize(hook).issubset(normalize(source))


def _reviewed_exchange_headline(candidate: ClipCandidate, hook: str) -> bool:
    """Accept a paraphrase only with retained exact-span review evidence."""
    required = {
        "headline_origin=reviewed_full_exchange_summary",
        "span_review=standalone_opening_delivered_payoff_complete_ending",
        "context_assessment=structured_local_instruct_model",
    }
    if not required.issubset(candidate.reasons) or hook != _semantic_hook_override(candidate):
        return False
    if not 4 <= len(_WORDS.findall(hook)) <= 14:
        return False
    for prefix in ("setup_quote=", "payoff_quote="):
        quotes = [
            reason.removeprefix(prefix) for reason in candidate.reasons if reason.startswith(prefix)
        ]
        if (
            len(quotes) != 1
            or not quotes[0].strip()
            or quotes[0].casefold() not in candidate.text.casefold()
        ):
            return False
    return True


def candidate_gate_failures(
    candidate: ClipCandidate,
    *,
    segments: Sequence[TranscriptSegment] | None = None,
    hook_override: str | None = None,
    allow_review_only_opening: bool = False,
) -> list[str]:
    """Explain each failed gate instead of hiding distinct causes in one label."""
    text = candidate.text.strip()
    words = _WORDS.findall(text)
    failed: list[str] = []
    # Missing contextual evidence fails closed; lexical cues never substitute.
    story_margin = _reason_float(candidate, "context_story_margin=")
    ending_margin = _reason_float(candidate, "context_ending_margin=")
    opening_margin = _reason_float(candidate, "context_opening_margin=")
    if any(
        value is None or not -2 <= value <= 2
        for value in (story_margin, ending_margin, opening_margin)
    ):
        failed.append("CONTEXTUAL_ASSESSMENT_REQUIRED")
    elif (
        story_margin is not None
        and ending_margin is not None
        and (story_margin <= 0 or ending_margin <= 0)
    ):
        failed.append("UNRESOLVED_CONTEXTUAL_MOMENT")
    if not words:
        failed.append("EMPTY_TRANSCRIPT")
    hook = creative_hook_from_text(text) if hook_override is None else hook_override.strip()
    if not hook:
        failed.append("NO_GROUNDED_HOOK")
    elif not _grounded_numbers(hook, text):
        failed.append("UNSUPPORTED_NUMERICAL_HOOK")
    elif hook.upper() not in source_headline_candidates(text) and not _reviewed_exchange_headline(
        candidate, hook
    ):
        failed.append("UNSUPPORTED_SOURCE_HOOK")
    if opening_margin is not None and opening_margin <= 0 and not allow_review_only_opening:
        failed.append("CONTEXT_OPENING_NEEDS_REVIEW")
    return failed


def evaluate_candidate(
    candidate: ClipCandidate,
    *,
    segments: Sequence[TranscriptSegment] | None = None,
    review: EditorialReview | None = None,
    hook_override: str | None = None,
    allow_review_only_opening: bool = False,
) -> EditorialPick | None:
    failed = candidate_gate_failures(
        candidate,
        segments=segments,
        hook_override=hook_override,
        allow_review_only_opening=allow_review_only_opening,
    )
    if failed:
        return None
    text = candidate.text.strip()
    hook = creative_hook_from_text(text) if hook_override is None else hook_override.strip()
    opening = _context_rating(candidate, "opening")
    ratings = {
        "opening": opening,
        "story": _context_rating(candidate, "story"),
        "emotion": CriterionRating(
            None, "manual_required", ("vocal delivery and emotion require audio/video evidence",)
        ),
        "visuals": CriterionRating(
            None,
            "manual_required",
            (
                "inspect final portrait crop and sampled source frames",
                "confirm required subjects and the important action remain visible",
            ),
        ),
        "retention": _context_rating(candidate, "ending"),
    }
    if review is not None and review.visual_score is not None:
        ratings["visuals"] = CriterionRating(
            review.visual_score, review.visual_basis, (review.visual_notes.strip(),)
        )

    integrity: IntegrityStatus = "unverified"
    evidence: tuple[str, ...] = (
        (
            "headline summarizes the selected exchange; "
            "retained setup/payoff quotes match its source text"
            if _reviewed_exchange_headline(candidate, hook)
            else (
                "headline preserves an intact source sentence "
                "including its subject, negation and amounts"
            )
        ),
        "source context, implied outcome and visual claims still require human verification",
    )
    if review is not None and review.integrity_passed is not None:
        integrity = "pass" if review.integrity_passed else "fail"
        evidence = (review.integrity_notes.strip(),)
    if integrity == "fail":
        return None

    coverage = sum(WEIGHTS[name] for name, rating in ratings.items() if rating.score is not None)
    points = sum(
        WEIGHTS[name] * rating.score / 5
        for name, rating in ratings.items()
        if rating.score is not None
    )
    score = round(points * 100 / coverage, 2)
    return EditorialPick(
        clip=candidate,
        hook=hook,
        hook_score=round(((_reason_float(candidate, "context_hook_margin=") or 0) + 2) * 1.5),
        editorial_score=score,
        weighted_points=round(points, 2),
        score_coverage=coverage,
        criteria=ratings,
        integrity_status=integrity,
        integrity_evidence=evidence,
        reasons=(
            f"opening={opening.score}/5 ({opening.basis}; see assessment evidence)",
            f"story={ratings['story'].score}/5 ({ratings['story'].basis})",
            f"ending={ratings['retention'].score}/5 ({ratings['retention'].basis})",
            "portrait visuals and editorial integrity require source review",
            *(
                ("review_only_weak_opening_needs_manual_first_two_seconds_approval",)
                if (_reason_float(candidate, "context_opening_margin=") or 0) <= 0
                else ()
            ),
        ),
    )


def _similar(left: EditorialPick, right: EditorialPick) -> bool:
    lw = set(_WORDS.findall(left.clip.text.lower())) - _STOP
    rw = set(_WORDS.findall(right.clip.text.lower())) - _STOP
    return bool(lw and rw) and len(lw & rw) / len(lw | rw) > 0.70


MAX_RENDERABLE_CLIPS = 20


def select_editorial_moments(
    candidates: list[ClipCandidate],
    *,
    render_safety_limit: int = MAX_RENDERABLE_CLIPS,
    segments: Sequence[TranscriptSegment] | None = None,
    allow_review_only_opening: bool = False,
    review_provider: Callable[[ClipCandidate], EditorialReview | None] | None = None,
) -> tuple[list[EditorialPick], list[dict[str, object]]]:
    """Rank contextual review drafts; never invent an editorial-quality cutoff."""
    if not 1 <= render_safety_limit <= MAX_RENDERABLE_CLIPS:
        raise ValueError(f"render safety limit must be 1-{MAX_RENDERABLE_CLIPS} for one runner job")
    qualified: list[EditorialPick] = []
    rejected: list[dict[str, object]] = []
    for candidate in candidates:
        hook_override = _semantic_hook_override(candidate)
        review = review_provider(candidate) if review_provider is not None else None
        pick = evaluate_candidate(
            candidate,
            segments=segments,
            review=review,
            hook_override=hook_override,
            allow_review_only_opening=allow_review_only_opening,
        )
        if pick is None:
            failed = candidate_gate_failures(
                candidate,
                segments=segments,
                hook_override=hook_override,
                allow_review_only_opening=allow_review_only_opening,
            )
            rejected.append(
                {
                    "start": candidate.start,
                    "end": candidate.end,
                    "reason": failed[0] if failed else "MANUAL_INTEGRITY_GATE",
                    "failed_gates": failed,
                    "opening_score": _context_rating(candidate, "opening").score,
                }
            )
        else:
            qualified.append(pick)
    ordered = sorted(
        qualified, key=lambda pick: (-pick.editorial_score, -pick.clip.score, pick.clip.start)
    )
    chosen: list[EditorialPick] = []
    used_hooks: set[str] = set()
    for pick in ordered:
        overlap = any(
            pick.clip.start < other.clip.end + 1.5 and other.clip.start < pick.clip.end + 1.5
            for other in chosen
        )
        if overlap or any(_similar(pick, other) for other in chosen):
            rejected.append(
                {
                    "start": pick.clip.start,
                    "end": pick.clip.end,
                    "reason": "DUPLICATE_OR_OVERLAP",
                    "editorial_score": pick.editorial_score,
                }
            )
            continue
        semantic_headline = _semantic_hook_override(pick.clip)
        headline = (
            semantic_headline
            if semantic_headline and semantic_headline.casefold() not in used_hooks
            else ("" if semantic_headline else distinct_hook_from_text(pick.clip.text, used_hooks))
        )
        if (
            not headline
            or headline.startswith('THE MOMENT: "')
            or not _grounded_numbers(headline, pick.clip.text)
        ):
            rejected.append(
                {
                    "start": pick.clip.start,
                    "end": pick.clip.end,
                    "reason": "NO_CREATOR_GRADE_GROUNDED_HOOK",
                    "editorial_score": pick.editorial_score,
                }
            )
            continue
        if len(chosen) >= render_safety_limit:
            rejected.append(
                {
                    "start": pick.clip.start,
                    "end": pick.clip.end,
                    "reason": "RENDER_SAFETY_LIMIT",
                    "editorial_score": pick.editorial_score,
                }
            )
            continue
        chosen.append(replace(pick, hook=headline))
        used_hooks.add(headline.casefold())
    return chosen, rejected
