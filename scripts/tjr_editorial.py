"""Quality-first TJR candidate selection. AI-assisted screening, not publication approval.

Select every sufficiently strong, non-overlapping original quote/reaction within
the configured render-time budget; do not impose an arbitrary two-clip limit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from clipper.models import ClipCandidate

_WORDS = re.compile(r"[a-zA-Z0-9$']+")
_REACTION = re.compile(
    r"\b(damn|wait|what the hell|no way|crazy|actually|insane|lost|stopped out|"
    r"made|million|thousand|profit|risk|mistake|wrong|truth|never|secret|"
    r"why|how|what|who|when)\b|[$][0-9]|[0-9]+\s*(?:k|percent|%)",
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
_UNFINISHED_ENDINGS = ("and then", "but", "because", "and", "when", "which", "that", "i'm")
_STOP = frozenset({"i", "you", "the", "a", "and", "for", "is", "it", "bro", "like", "that"})


@dataclass(frozen=True)
class EditorialPick:
    clip: ClipCandidate
    hook: str
    hook_score: int
    editorial_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self.clip),
            "hook_candidate": self.hook,
            "hook_score": self.hook_score,
            "editorial_score": self.editorial_score,
            "editorial_reasons": list(self.reasons),
            "publish_approved": False,
            "needs_checks": [
                "TJR clearly visible and positively contextualized",
                "No embedded or newly added logos anywhere",
                "Exact speech, captions, and cut boundary match source audio",
                "First two seconds contain a real spoken/visual hook",
                "No duplicate moment and no artificial or misleading return claim",
            ],
        }


def evaluate_candidate(candidate: ClipCandidate) -> EditorialPick | None:
    text = candidate.text.strip()
    words = _WORDS.findall(text)
    if not 28 <= len(words) <= 155:
        return None
    first = " ".join(words[:12])
    beginning = " ".join(words[:7]).lower()
    reaction = bool(_REACTION.search(first))
    opening_question = "?" in text[:90] or beginning.startswith(("why ", "how ", "what ", "who "))
    specific = bool(re.search(r"[$][0-9]|\b[0-9]+(?:[.,][0-9]+)?(?:k|%)?\b", first, re.I))
    weak = any(beginning.startswith(term) for term in _WEAK_OPENINGS)
    hook_score = int(reaction) * 2 + int(opening_question) * 2 + int(specific) * 2
    if weak:
        hook_score -= 2
    # Avoid videos that don't quickly establish a question, reaction or stakes.
    if hook_score <= 0:
        return None
    density = len(words) / max(candidate.duration, 1.0)
    # Genuine reaction hooks can be delivered deliberately; do not reject
    # authentic moments solely for moderately paced speech.
    if density < 0.90 or density > 5.0:
        return None
    unfinished = any(text.lower().rstrip(" .!?").endswith(x) for x in _UNFINISHED_ENDINGS)
    editorial_score = candidate.score + hook_score * 4.0 - int(unfinished) * 3.0
    reasons = (
        "spoken hook near beginning",
        "specific stakes" if specific else "contextual reaction",
        "cut boundary requires editorial check" if unfinished else "independent moment",
    )
    return EditorialPick(candidate, first, hook_score, round(editorial_score, 2), reasons)


def _similar(left: EditorialPick, right: EditorialPick) -> bool:
    lw = set(_WORDS.findall(left.clip.text.lower())) - _STOP
    rw = set(_WORDS.findall(right.clip.text.lower())) - _STOP
    if not lw or not rw:
        return False
    return len(lw & rw) / len(lw | rw) > 0.70


def select_editorial_moments(
    candidates: list[ClipCandidate], *, batch_limit: int = 12
) -> tuple[list[EditorialPick], list[dict[str, object]]]:
    """Batch limit controls one run's cost, not lifetime clip inventory."""
    if batch_limit < 1 or batch_limit > 20:
        raise ValueError("editorial batch limit must be 1-20 for one runner job")
    qualified: list[EditorialPick] = []
    rejected: list[dict[str, object]] = []
    for item in candidates:
        evaluated = evaluate_candidate(item)
        if evaluated is None:
            rejected.append(
                {"start": item.start, "end": item.end, "reason": "WEAK_HOOK_OR_LOW_SPEECH_DENSITY"}
            )
        else:
            qualified.append(evaluated)
    ordered = sorted(qualified, key=lambda p: (-p.editorial_score, p.clip.start))
    chosen: list[EditorialPick] = []
    for pick in ordered:
        overlap = any(
            pick.clip.start < other.clip.end + 1.5 and other.clip.start < pick.clip.end + 1.5
            for other in chosen
        )
        if overlap or any(_similar(pick, other) for other in chosen):
            rejected.append(
                {"start": pick.clip.start, "end": pick.clip.end, "reason": "DUPLICATE_OR_OVERLAP"}
            )
            continue
        chosen.append(pick)
        if len(chosen) >= batch_limit:
            break
    return chosen, rejected
