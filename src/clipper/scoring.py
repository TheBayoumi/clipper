from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .models import CampaignBrief, ClipCandidate, TranscriptSegment

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_HOOK_WORDS = {
    "secret",
    "mistake",
    "why",
    "how",
    "never",
    "best",
    "worst",
    "truth",
    "imagine",
    "here's",
    "stop",
    "warning",
    "problem",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _phrase_present(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _window_score(
    brief: CampaignBrief,
    text: str,
    duration: float,
) -> tuple[float, tuple[str, ...]]:
    tokens = _tokens(text)
    counts = Counter(tokens)
    reasons: list[str] = []
    keyword_hits = sum(min(counts[word.lower()], 2) for word in brief.keywords)
    negative_hits = sum(counts[word.lower()] for word in brief.negative_keywords)
    required_hits = sum(_phrase_present(text, phrase) for phrase in brief.required_phrases)
    hook_hits = sum(counts[word] for word in _HOOK_WORDS)

    target = (brief.min_clip_seconds + brief.max_clip_seconds) / 2
    duration_quality = max(0.0, 1.0 - abs(duration - target) / max(target, 1.0))
    density = len(set(tokens)) / max(len(tokens), 1)

    score = keyword_hits * 3.0 + required_hits * 5.0 + hook_hits * 1.25
    score += duration_quality * 2.0 + density
    score -= negative_hits * 5.0
    score -= max(0, len(tokens) - 190) * 0.02

    if keyword_hits:
        reasons.append(f"keyword_hits={keyword_hits}")
    if required_hits:
        reasons.append(f"required_phrase_hits={required_hits}")
    if hook_hits:
        reasons.append(f"hook_hits={hook_hits}")
    if negative_hits:
        reasons.append(f"negative_hits={negative_hits}")
    reasons.append(f"duration_quality={duration_quality:.2f}")
    return round(score, 4), tuple(reasons)


def _overlap_ratio(left: ClipCandidate, right: ClipCandidate) -> float:
    if left.video_id != right.video_id:
        return 0.0
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0


def score_transcript(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
    *,
    limit: int = 20,
) -> list[ClipCandidate]:
    if not segments:
        return []
    candidates: list[ClipCandidate] = []
    for start_index, first in enumerate(segments):
        text_parts: list[str] = []
        for segment in segments[start_index:]:
            duration = segment.end - first.start
            if duration > brief.max_clip_seconds:
                break
            text_parts.append(segment.text)
            if duration < brief.min_clip_seconds:
                continue
            text = " ".join(text_parts).strip()
            score, reasons = _window_score(brief, text, duration)
            candidates.append(
                ClipCandidate(
                    video_id=video_id,
                    start=max(0.0, math.floor(first.start * 10) / 10),
                    end=math.ceil(segment.end * 10) / 10,
                    text=text,
                    score=score,
                    reasons=reasons,
                )
            )
            if text.endswith((".", "!", "?")):
                break

    selected: list[ClipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.start)):
        if any(_overlap_ratio(candidate, existing) >= 0.55 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def select_diverse_clips(
    candidates: Sequence[ClipCandidate],
    *,
    clip_count: int,
    max_per_source: int,
) -> list[ClipCandidate]:
    selected: list[ClipCandidate] = []
    counts: Counter[str] = Counter()
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.video_id, item.start)):
        if counts[candidate.video_id] >= max_per_source:
            continue
        selected.append(candidate)
        counts[candidate.video_id] += 1
        if len(selected) >= clip_count:
            break
    return selected
