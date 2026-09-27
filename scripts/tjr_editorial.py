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

RUBRIC_VERSION = "tjr-editorial-v1"
WEIGHTS = {"opening": 25, "story": 25, "emotion": 10, "visuals": 15, "retention": 25}
_WORDS = re.compile(r"[A-Za-z0-9$']+")
_NUMBERS = re.compile(r"(?<!\w)\$?\d[\d,.]*(?:%|k|m)?\b", re.IGNORECASE)
_REACTION = re.compile(
    r"\b(damn|wow|no way|what the hell|insane|unbelievable|wait|"
    r"look at that|are you serious)\b",
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
    r"stop loss|profit|loss)\b",
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


def evaluate_candidate(
    candidate: ClipCandidate,
    *,
    segments: Sequence[TranscriptSegment] | None = None,
    review: EditorialReview | None = None,
    hook_override: str | None = None,
) -> EditorialPick | None:
    text = candidate.text.strip()
    words = _WORDS.findall(text)
    if _OFF_TOPIC.search(text) or not _TOPIC.search(text):
        return None
    if not 28 <= len(words) <= 155:
        return None
    density = len(words) / max(candidate.duration, 1.0)
    if not 0.9 <= density <= 5.0:
        return None

    hook = creative_hook_from_text(text) if hook_override is None else hook_override.strip()
    if not hook or not _grounded_numbers(hook, text):
        return None

    opening = _opening_rating(candidate, segments)
    if opening.score is None or opening.score < 2:
        return None
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
        ),
    )


def _similar(left: EditorialPick, right: EditorialPick) -> bool:
    lw = set(_WORDS.findall(left.clip.text.lower())) - _STOP
    rw = set(_WORDS.findall(right.clip.text.lower())) - _STOP
    return bool(lw and rw) and len(lw & rw) / len(lw | rw) > 0.70


def select_editorial_moments(
    candidates: list[ClipCandidate],
    *,
    batch_limit: int = 12,
    segments: Sequence[TranscriptSegment] | None = None,
) -> tuple[list[EditorialPick], list[dict[str, object]]]:
    """Rank provisional draft candidates; never imply visual or integrity approval."""
    if not 1 <= batch_limit <= 20:
        raise ValueError("editorial batch limit must be 1-20 for one runner job")
    qualified: list[EditorialPick] = []
    rejected: list[dict[str, object]] = []
    for candidate in candidates:
        pick = evaluate_candidate(candidate, segments=segments)
        if pick is None:
            rejected.append(
                {
                    "start": candidate.start,
                    "end": candidate.end,
                    "reason": "TOPIC_LENGTH_DENSITY_OPENING_OR_INTEGRITY_GATE",
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
        headline = distinct_hook_from_text(pick.clip.text, used_hooks)
        if not headline or not _grounded_numbers(headline, pick.clip.text):
            rejected.append(
                {
                    "start": pick.clip.start,
                    "end": pick.clip.end,
                    "reason": "NO_UNIQUE_GROUNDED_HOOK",
                }
            )
            continue
        chosen.append(replace(pick, hook=headline))
        used_hooks.add(headline.casefold())
        if len(chosen) >= batch_limit:
            break
    return chosen, rejected
