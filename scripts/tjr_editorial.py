"""Evidence-labeled editorial screening for TJR review drafts.

The five weighted criteria use transcript *proxies*, never pretend to measure
genuine emotion, semantic payoff, or whether important action survives a crop.
Visual suitability and hook integrity require explicit review. Draft selection
is permitted while publication approval remains false.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from clipper.models import ClipCandidate, TranscriptSegment
from clipper.tiktok import creative_hook_from_text, distinct_hook_from_text

RUBRIC_VERSION = "tjr-editorial-v3-creator-quality"
WEIGHTS = {"opening": 25, "story": 25, "emotion": 10, "visuals": 15, "retention": 25}
MIN_CREATOR_EDITORIAL_SCORE = 74.0
MAX_SCORE_DROP_FROM_BEST = 8.0
_WORDS = re.compile(r"[A-Za-z0-9$']+")
_NUMBERS = re.compile(r"(?<!\w)\$?\d[\d,.]*(?:%|k|m)?\b", re.IGNORECASE)
_REACTION = re.compile(
    r"\b(damn|wow|no way|what the hell|insane|unbelievable|wait|"
    r"look at that|are you serious|holy shit|no shot|what the fuck)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"^(why|how|what|who|when|where|did|does|can|should)\b", re.I)
_ACTION = re.compile(
    r"\b(hit|broke|break|stopped|entered|reversed|crashed|spiked|"
    r"lost|made|jumped|risk|mistake)\b",
    re.IGNORECASE,
)
_TOPIC = re.compile(
    r"\b(trad(?:e|ing|ers?)|market|price|risk|coin|futures|nasdaq|"
    r"order blocks?|bullish|bearish|position|equilibrium|chart|candle|"
    r"stop loss|profit|loss|retrace|liquidity|pullback|breakout|"
    r"daily bias|nasdaq|\bNQ\b|\bES\b|vwap|scalp|entry|exits?)\b",
    re.IGNORECASE,
)
_OFF_TOPIC = re.compile(r"\b(cocaine|ketamine|weed|stoner|snort|smoking)\b", re.I)
_SETUP = re.compile(r"\b(why|how|what|if|when|because|plan|setup|trade)\b", re.I)
_TENSION = re.compile(
    r"\b(but|however|instead|risk|lost|loss|wrong|mistake|against|"
    r"problem|stopped out|unexpected|reverse|reversal|chase)\b",
    re.IGNORECASE,
)
_PAYOFF = re.compile(
    r"\b(so|because|therefore|that's why|avoid|learn|manage|"
    r"decide|exit|fix|result|instead|plan|before)\b",
    re.IGNORECASE,
)
_WEAK_OPENINGS = (
    "okay so",
    "and then",
    "all right so",
    "uh ",
    "um ",
    "so basically",
    "let me just",
    "we should be",
    "so we have",
    "let's see",
)
_UNFINISHED = re.compile(r"\b(and then|but|because|and|when|which|that|i'm)\W*$", re.I)
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
CriterionBasis = Literal["transcript_proxy", "manual_verified", "manual_required"]
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
            "score_basis": "normalized_observed_criteria_not_predicted_virality",
            "integrity_gate": {
                "status": self.integrity_status,
                "evidence": list(self.integrity_evidence),
            },
            "editorial_reasons": list(self.reasons),
            "human_review_required": (
                self.criteria["visuals"].score is None or self.integrity_status != "pass"
            ),
            "publish_approved": False,
            "needs_checks": [
                "TJR actually visible; important action survives the portrait crop",
                "No embedded or newly added logos; captions match exact speech",
                "First two seconds contain a real spoken or visual hook",
                "Setup, tension and payoff survive the final cut",
                "Creative hook makes no unsupported claims; inspect source context",
                "Check campaign eligibility and source rights before publication",
            ],
        }


def _clamp(value: float) -> float:
    return round(max(0.0, min(5.0, value)), 2)


def _opening_text(
    candidate: ClipCandidate, segments: Sequence[TranscriptSegment] | None
) -> tuple[str, str]:
    """Use actual first-2s word timestamps when available, never later speech."""
    if segments:
        timed = sorted(
            (
                word
                for segment in segments
                if segment.end > candidate.start and segment.start < candidate.end
                for word in segment.words
                if word.end > candidate.start and word.start < candidate.end
            ),
            key=lambda word: (word.start, word.end),
        )
        if timed:
            immediate = [
                word.text
                for word in timed
                if word.start < candidate.start + 2.0 and word.end > candidate.start
            ]
            return " ".join(immediate), "first_two_seconds_word_aligned"
    words = _WORDS.findall(candidate.text)
    estimate = min(12, max(3, round(len(words) * 2 / max(candidate.duration, 1))))
    return " ".join(words[:estimate]), "estimated_opening_no_word_timestamps"


def _opening_rating(
    candidate: ClipCandidate, segments: Sequence[TranscriptSegment] | None
) -> CriterionRating:
    opening, basis = _opening_text(candidate, segments)
    lowered = opening.lower().strip()
    if not lowered:
        return CriterionRating(0, "transcript_proxy", (basis, "no aligned opening speech"))
    reaction = bool(_REACTION.search(opening))
    question = bool(_QUESTION.search(opening)) or "?" in opening
    specific = bool(_NUMBERS.search(opening))
    action = bool(_ACTION.search(opening))
    weak = any(lowered.startswith(prefix) for prefix in _WEAK_OPENINGS)
    score = 1.0 + 2.0 * (reaction or question) + float(specific) + float(action) - 2.0 * weak
    return CriterionRating(
        _clamp(score),
        "transcript_proxy",
        (
            basis,
            f"opening_excerpt={opening[:130]}",
            f"reaction_cue={reaction}; question_cue={question}; specific_stake={specific}",
            f"action_cue={action}; weak_opening={weak}",
            "spoken text alone cannot verify an audible or visual hook",
        ),
    )


def _story_rating(text: str) -> CriterionRating:
    words = _WORDS.findall(text)
    third = max(1, len(words) // 3)
    beginning = " ".join(words[:third])
    middle = " ".join(words[third : 2 * third])
    end = " ".join(words[2 * third :])
    setup = bool(_SETUP.search(beginning)) or "?" in beginning
    tension = bool(_TENSION.search(middle)) or bool(_TENSION.search(text))
    payoff = bool(_PAYOFF.search(end))
    unfinished = bool(_UNFINISHED.search(text.rstrip()))
    score = 0.5 + float(setup) + 1.5 * tension + 2.0 * payoff - 2.0 * unfinished
    return CriterionRating(
        _clamp(score),
        "transcript_proxy",
        (
            f"setup_cue={setup}; tension_cue={tension}; payoff_cue={payoff}",
            f"unfinished_boundary={unfinished}",
            "lexical cues are not proof of an intelligible setup or payoff",
        ),
    )


def _emotion_rating(text: str) -> CriterionRating:
    words = _WORDS.findall(text)
    early = " ".join(words[: max(1, len(words) // 2)])
    reaction = bool(_REACTION.search(early))
    emphasis = bool(
        re.search(r"\b(never|really|i can't|can't believe|surprised|excited)\b", early, re.I)
    )
    # A measured absence of reaction words is NOT evidence of a boring delivery.
    return CriterionRating(
        4.0 if reaction else 2.5 if emphasis else 2.0,
        "transcript_proxy",
        (
            f"reaction_word_cue={reaction}; emphasis_word_cue={emphasis}",
            "genuine tone, humor and conviction require audio/video review",
        ),
    )


def _retention_rating(text: str) -> CriterionRating:
    tokens = [word.lower() for word in _WORDS.findall(text)]
    third = max(1, len(tokens) // 3)
    chunks = [tokens[:third], tokens[third : 2 * third], tokens[2 * third :]]
    content = [
        [token for token in chunk if token not in _STOP and len(token) > 2] for chunk in chunks
    ]
    earlier = set(content[0])
    novelty: list[float] = []
    for chunk in content[1:]:
        novel = sum(word not in earlier for word in chunk)
        novelty.append(novel / len(chunk) if chunk else 0.0)
        earlier.update(chunk)
    unique_triples = {tuple(tokens[index : index + 3]) for index in range(max(0, len(tokens) - 2))}
    total_triples = max(0, len(tokens) - 2)
    repetition = (1.0 - len(unique_triples) / total_triples) if total_triples else 0.0
    score = 1.0 + 4.0 * sum(novelty) / 2.0 - 2.0 * repetition
    return CriterionRating(
        _clamp(score),
        "transcript_proxy",
        (
            f"middle_new_token_ratio={novelty[0]:.2f}",
            f"ending_new_token_ratio={novelty[1]:.2f}",
            f"repeated_trigram_fraction={repetition:.2f}",
            "new vocabulary is only a proxy for meaningful new information or action",
        ),
    )


def _grounded_numbers(hook: str, source: str) -> bool:
    def normalize(text: str) -> set[str]:
        return {re.sub(r"[,$]", "", match).lower() for match in _NUMBERS.findall(text)}

    return normalize(hook).issubset(normalize(source))


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
    if _OFF_TOPIC.search(text):
        failed.append("POLICY_SENSITIVE_OFF_TOPIC")
    story = _story_rating(text)
    authentic_reaction = _REACTION.search(text) and (story.score or 0) >= 4
    if not _TOPIC.search(text) and not authentic_reaction:
        failed.append("NO_TRADING_CONTEXT_OR_COMPLETE_REACTION")
    if not 28 <= len(words) <= 155:
        failed.append("WORD_COUNT_OUT_OF_RANGE")
    if not 0.9 <= len(words) / max(candidate.duration, 1.0) <= 5.0:
        failed.append("SPEECH_DENSITY_OUT_OF_RANGE")
    hook = creative_hook_from_text(text) if hook_override is None else hook_override.strip()
    if not hook:
        failed.append("NO_GROUNDED_HOOK")
    elif not _grounded_numbers(hook, text):
        failed.append("UNSUPPORTED_NUMERICAL_HOOK")
    opening = _opening_rating(candidate, segments)
    if opening.score is None or opening.score < 2:
        if not allow_review_only_opening:
            failed.append("WEAK_FIRST_TWO_SECONDS")
        elif (
            opening.score is None
            or not _TOPIC.search(text)
            or (story.score or 0) < 3
            or not text.rstrip().endswith((".", "!", "?"))
        ):
            # A persistent on-screen hook can be tried only on a complete,
            # substantive trading story. It cannot repair a cut-off ending.
            failed.append("WEAK_OPENING_WITHOUT_COMPLETE_TRADING_STORY")
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
    opening = _opening_rating(candidate, segments)
    ratings = {
        "opening": opening,
        "story": _story_rating(text),
        "emotion": _emotion_rating(text),
        "visuals": CriterionRating(
            None,
            "manual_required",
            (
                "inspect final portrait crop and sampled source frames",
                "confirm TJR and important action stay visible throughout",
            ),
        ),
        "retention": _retention_rating(text),
    }
    if review is not None and review.visual_score is not None:
        ratings["visuals"] = CriterionRating(
            review.visual_score, "manual_verified", (review.visual_notes.strip(),)
        )

    integrity: IntegrityStatus = "unverified"
    evidence = (
        "generated hook contains no numerical claim absent from the source transcript",
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
        hook_score=round(opening.score * 1.2),
        editorial_score=score,
        weighted_points=round(points, 2),
        score_coverage=coverage,
        criteria=ratings,
        integrity_status=integrity,
        integrity_evidence=evidence,
        reasons=(
            f"first_two_seconds={opening.score:.1f}/5 (transcript proxy)",
            f"story={ratings['story'].score:.1f}/5 (lexical proxy)",
            f"retention={ratings['retention'].score:.1f}/5 (lexical proxy)",
            "portrait visuals and editorial integrity require source review",
            *(
                ("review_only_weak_opening_needs_manual_first_two_seconds_approval",)
                if opening.score is not None and opening.score < 2
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
) -> tuple[list[EditorialPick], list[dict[str, object]]]:
    """Qualify every creator-grade moment; the limit is infrastructure safety only."""
    if not 1 <= render_safety_limit <= MAX_RENDERABLE_CLIPS:
        raise ValueError(
            f"render safety limit must be 1-{MAX_RENDERABLE_CLIPS} for one runner job"
        )
    qualified: list[EditorialPick] = []
    rejected: list[dict[str, object]] = []
    for candidate in candidates:
        pick = evaluate_candidate(
            candidate,
            segments=segments,
            allow_review_only_opening=allow_review_only_opening,
        )
        if pick is None:
            failed = candidate_gate_failures(
                candidate,
                segments=segments,
                allow_review_only_opening=allow_review_only_opening,
            )
            rejected.append(
                {
                    "start": candidate.start,
                    "end": candidate.end,
                    "reason": failed[0] if failed else "MANUAL_INTEGRITY_GATE",
                    "failed_gates": failed,
                    "opening_score": _opening_rating(candidate, segments).score,
                }
            )
        else:
            qualified.append(pick)
    ordered = sorted(
        qualified, key=lambda pick: (-pick.editorial_score, -pick.clip.score, pick.clip.start)
    )
    best_score = ordered[0].editorial_score if ordered else 0.0
    dynamic_floor = max(MIN_CREATOR_EDITORIAL_SCORE, best_score - MAX_SCORE_DROP_FROM_BEST)
    chosen: list[EditorialPick] = []
    used_hooks: set[str] = set()
    for pick in ordered:
        if pick.editorial_score < dynamic_floor:
            rejected.append(
                {
                    "start": pick.clip.start,
                    "end": pick.clip.end,
                    "reason": "BELOW_CREATOR_QUALITY_FLOOR",
                    "editorial_score": pick.editorial_score,
                    "dynamic_floor": round(dynamic_floor, 2),
                }
            )
            continue
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
        headline = distinct_hook_from_text(pick.clip.text, used_hooks)
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
        chosen.append(replace(pick, hook=headline))
        used_hooks.add(headline.casefold())
        # This is not an editorial target or quota. It only bounds pathological
        # render fan-out after every accepted moment has independently passed quality.
        if len(chosen) >= render_safety_limit:
            break
    return chosen, rejected
