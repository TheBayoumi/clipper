from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace

from .models import (
    CampaignBrief,
    ClipConcept,
    EditorialBeat,
    EditorialScores,
    EditorialScoreWeights,
    EditPlan,
    HookMode,
    HookVariant,
    SourceSpan,
    StoryMoment,
    TranscriptSegment,
)
from .wordstream import (
    find_phrase_anchor,
    first_complete_word,
    flatten_source_words,
    normalized_tokens,
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(
    r"(?:\$\s*)?\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b",
    re.I,
)
_HOOK_NUMBER_RE = re.compile(
    r"(?:\$\s*)?\b\d[\d,.]*\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)"
    r"\s+(?:years?|hours?|people|players?|dollars?|bucks?|grand|times?|pounds?|cars?)\b",
    re.I,
)
_HOUSEKEEPING_PHRASES = (
    "what's up guys",
    "welcome back",
    "like and subscribe",
    "subscribe to",
    "this is the podcast",
    "and this is the",
)
_INCOMPLETE_ENDINGS = {
    "and",
    "but",
    "because",
    "so",
    "to",
    "the",
    "a",
    "an",
    "of",
    "for",
    "with",
    "or",
    "you're",
    "we're",
    "they're",
    "i'm",
    "it's",
    "i",
    "you",
    "we",
    "they",
}
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "like",
    "just",
    "really",
    "kind",
    "thing",
    "things",
}
_FILLER_START = (
    "so ",
    "um ",
    "uh ",
    "like ",
    "you know ",
    "i mean ",
    "and ",
    "but ",
    "yeah ",
    "okay ",
    "right ",
    "though ",
    "where ",
    "year ",
    "month ",
    "before we head out",
)
_HOOK_WORDS = {
    "secret",
    "mistake",
    "never",
    "best",
    "worst",
    "truth",
    "problem",
    "crazy",
    "insane",
    "million",
    "money",
    "broke",
    "failed",
    "failure",
    "won",
    "winning",
    "risk",
    "quit",
    "dropped",
    "first",
    "only",
    "exactly",
    "changed",
    "biggest",
    "hardest",
    "easy",
    "impossible",
}
_TENSION_WORDS = {
    "but",
    "however",
    "never",
    "against",
    "versus",
    "wrong",
    "hate",
    "lost",
    "lose",
    "failure",
    "failed",
    "problem",
    "risk",
    "scared",
    "pressure",
    "fight",
    "hard",
    "difficult",
    "crazy",
}
_EMOTION_WORDS = {
    "love",
    "hate",
    "scared",
    "afraid",
    "happy",
    "sad",
    "excited",
    "crazy",
    "insane",
    "proud",
    "confident",
    "confidence",
    "stress",
    "pressure",
    "dream",
    "regret",
    "hurt",
    "angry",
}
_RESOLUTION_WORDS = {
    "because",
    "therefore",
    "that's why",
    "so now",
    "ended up",
    "realized",
    "learned",
    "changed",
    "result",
    "finally",
    "eventually",
    "now",
    "today",
}
_PAYOFF_SIGNALS = {
    "paid",
    "won",
    "earned",
    "made",
    "got",
    "learned",
    "realized",
    "changed",
    "fixed",
    "saved",
    "worked",
    "finally",
}
_QUESTION_OPENERS = (
    "why ",
    "how ",
    "what ",
    "when ",
    "where ",
    "who ",
    "did ",
    "do ",
    "can ",
    "would ",
)

_CONTEXT_CONTINUATION_OPENERS = (
    "as ",
    "and ",
    "but ",
    "yeah ",
    "right ",
    "like ",
)
_CONTEXT_META_PHRASES = (
    "what are you trying to say",
    "what do you mean",
    "you see what i'm saying",
    "you know what i mean",
    "what are we talking about",
)
_MONEY_CLAIM_WORDS = {
    "dollar",
    "dollars",
    "buck",
    "bucks",
    "million",
    "thousand",
    "paid",
    "made",
    "earned",
    "won",
    "bought",
    "purchase",
    "cost",
    "money",
}


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in _WORD_RE.findall(text)]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in _STOP and len(token) > 2]


def transcript_fingerprint(text: str) -> str:
    normalized = " ".join(_tokens(text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _record_rejection(
    rejections: list[dict[str, object]] | None,
    *,
    video_id: str,
    stage: str,
    reason: str,
    start: float | None = None,
    end: float | None = None,
    text: str = "",
    scores: EditorialScores | None = None,
) -> None:
    if rejections is None:
        return
    rejections.append(
        {
            "concept_id": (
                f"{video_id}-{transcript_fingerprint(text)[:10]}" if text.strip() else None
            ),
            "video_id": video_id,
            "stage": stage,
            "decision": "REJECT",
            "reasons": [reason],
            "source_start": start,
            "source_end": end,
            "text": text[:500],
            "scores": scores.to_dict() if scores is not None else {},
        }
    )


def semantic_similarity(left: str, right: str) -> float:
    left_counts = Counter(_content_tokens(left))
    right_counts = Counter(_content_tokens(right))
    if not left_counts or not right_counts:
        return 0.0
    shared = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    cosine = dot / max(left_norm * right_norm, 1e-9)
    union = set(left_counts) | set(right_counts)
    jaccard = len(shared) / max(len(union), 1)
    return round(0.75 * cosine + 0.25 * jaccard, 4)


def _clamp_score(value: float) -> float:
    return round(min(10.0, max(0.0, value)), 2)


def _sentences(text: str) -> list[str]:
    items = [item.strip() for item in _SENTENCE_RE.split(text.strip()) if item.strip()]
    return items or ([text.strip()] if text.strip() else [])


def _starts_weak(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith(_FILLER_START) or lowered.startswith(
        ("it ", "this ", "that ", "they ")
    )


def _context_dependence_penalty(text: str) -> float:
    lowered = text.strip().lower()
    first_sentence = _sentences(lowered)[0] if lowered else ""
    penalty = 0.0
    if first_sentence.startswith(_CONTEXT_CONTINUATION_OPENERS):
        penalty += 1.4
    if any(phrase in first_sentence for phrase in _CONTEXT_META_PHRASES):
        penalty += 2.4
    first_tokens = _tokens(first_sentence)[:10]
    dependent = sum(
        token in {"he", "she", "they", "it", "this", "that", "him", "her"} for token in first_tokens
    )
    if dependent >= 3:
        penalty += min(1.6, dependent * 0.35)
    return penalty


def start_boundary_score(text: str) -> float:
    stripped = text.strip()
    lowered = stripped.lower()
    tokens = _tokens(stripped[:180])
    score = 4.5
    if not stripped:
        return 0.0
    if _starts_weak(stripped):
        score -= 3.2
    score -= _context_dependence_penalty(stripped)
    first_tokens = tokens[:4]
    if len(first_tokens) >= 3 and len(set(first_tokens)) <= 2:
        score -= 1.2
    if stripped.endswith("?") or lowered.startswith(_QUESTION_OPENERS):
        score += 2.0
    if _NUMBER_RE.search(stripped[:140]):
        score += 1.5
    score += min(2.0, sum(token in _HOOK_WORDS for token in tokens) * 0.7)
    if len(tokens) >= 5:
        score += 0.5
    return _clamp_score(score)


def end_boundary_score(text: str, *, next_gap: float = 0.0) -> float:
    stripped = text.rstrip()
    if not stripped:
        return 0.0
    score = 3.0
    punctuated = stripped.endswith((".", "!", "?"))
    paused = next_gap >= 0.35
    terminal_evidence = punctuated or paused
    if punctuated:
        score += 3.0
    if paused:
        score += min(1.5, next_gap)
    if not terminal_evidence:
        score -= 1.8
    tokens = _tokens(stripped)
    last_token = tokens[-1] if tokens else ""
    if last_token in _INCOMPLETE_ENDINGS:
        score -= 3.2
    if stripped.endswith((",", ";", ":")):
        score -= 2.5
    last = _sentences(stripped)[-1] if stripped else ""
    if terminal_evidence and any(word in last.lower() for word in _RESOLUTION_WORDS):
        score += 1.5
    if len(_tokens(last)) < 3:
        score -= 1.0
    return _clamp_score(score)


def _campaign_relevance(brief: CampaignBrief, text: str) -> float:
    tokens = Counter(_tokens(text))
    hits = sum(min(tokens[key.lower()], 2) for key in brief.keywords)
    phrase_hits = sum(phrase.lower() in text.lower() for phrase in brief.required_phrases)
    return _clamp_score(2.0 + hits * 1.1 + phrase_hits * 2.0)


def score_editorial_text(
    brief: CampaignBrief,
    text: str,
    *,
    start_text: str | None = None,
    end_text: str | None = None,
    next_gap: float = 0.0,
) -> EditorialScores:
    tokens = _tokens(text)
    content = _content_tokens(text)
    counts = Counter(tokens)
    sentence_count = max(1, len(_sentences(text)))
    hook = start_boundary_score(start_text or (_sentences(text)[0] if text else text))
    curiosity = 2.0
    curiosity += 2.0 if "?" in text else 0.0
    curiosity += min(
        3.0, sum(counts[word] for word in {"why", "how", "secret", "never", "only"}) * 0.7
    )
    curiosity += 1.2 if _NUMBER_RE.search(text) else 0.0
    payoff = end_boundary_score(end_text or text, next_gap=next_gap)
    standalone = 7.2 - (2.0 if _starts_weak(text) else 0.0) - _context_dependence_penalty(text)
    if any(phrase in text.lower() for phrase in _HOUSEKEEPING_PHRASES):
        standalone -= 3.0
    standalone -= min(
        2.0,
        sum(counts[word] for word in {"he", "she", "they", "it", "that"})
        / max(len(tokens), 1)
        * 20,
    )
    emotional = 2.0 + min(6.0, sum(counts[word] for word in _EMOTION_WORDS) * 1.2)
    tension = 1.5 + min(7.0, sum(counts[word] for word in _TENSION_WORDS) * 0.9)
    information = 3.0 + min(3.0, len(set(content)) / max(len(content), 1) * 4.0)
    information += min(3.0, len(_NUMBER_RE.findall(text)) * 0.8)
    quoteability = 3.5 + (1.5 if sentence_count <= 7 else 0.0)
    quoteability += min(3.0, sum(counts[word] for word in _HOOK_WORDS) * 0.6)
    specificity = 2.5 + min(4.0, len(_NUMBER_RE.findall(text)) * 1.0)
    specificity += min(
        2.5, sum(1 for token in content if len(token) >= 7) / max(len(content), 1) * 7
    )
    relevance = _campaign_relevance(brief, text)
    completeness = (hook * 0.35) + (payoff * 0.55) + (1.0 if sentence_count >= 2 else 0.0)
    retention = (
        hook * 0.32 + curiosity * 0.22 + payoff * 0.24 + emotional * 0.08 + information * 0.14
    )
    return EditorialScores(
        hook_strength=_clamp_score(hook),
        curiosity=_clamp_score(curiosity),
        payoff_strength=_clamp_score(payoff),
        standalone_clarity=_clamp_score(standalone),
        emotional_energy=_clamp_score(emotional),
        information_value=_clamp_score(information),
        controversy_or_tension=_clamp_score(tension),
        quoteability=_clamp_score(quoteability),
        specificity=_clamp_score(specificity),
        campaign_relevance=_clamp_score(relevance),
        story_completeness=_clamp_score(completeness),
        retention_potential=_clamp_score(retention),
    )


def aggregate_editorial_score(
    scores: EditorialScores, weights: EditorialScoreWeights | None = None
) -> float:
    active = weights or EditorialScoreWeights()
    weight_map = active.to_dict() if hasattr(active, "to_dict") else asdict(active)
    payload = scores.to_dict()
    total_weight = sum(weight_map.values())
    return round(
        sum(payload[key] * weight for key, weight in weight_map.items()) / total_weight,
        4,
    )


def _moment_type(text: str) -> str:
    lowered = text.lower()
    if "?" in text:
        return "question_answer"
    if _NUMBER_RE.search(text) and any(
        word in lowered for word in ("money", "$", "million", "thousand", "paid", "bought")
    ):
        return "money_story"
    if any(word in lowered for word in ("failed", "failure", "mistake", "lost", "regret")):
        return "failure_lesson"
    if any(word in lowered for word in ("best", "worst", "never", "truth", "should")):
        return "strong_opinion"
    if any(word in lowered for word in ("because", "then", "when i", "eventually", "ended up")):
        return "story"
    if _NUMBER_RE.search(text):
        return "specific_fact"
    return "insight"


def _topic(text: str, campaign_keywords: Sequence[str]) -> str:
    lowered = text.lower()
    token_counts = Counter(_tokens(text))
    categories: dict[str, set[str]] = {
        "giveaway": {
            "giveaway",
            "charity",
            "philanthropy",
            "donate",
            "donation",
            "computers",
            "pcs",
        },
        "family": {"dad", "father", "mom", "mother", "parents", "brother", "family"},
        "skins": {"skin", "skins", "female", "male", "icon", "character", "hitbox"},
        "training": {
            "training",
            "coach",
            "reaction",
            "practice",
            "exercise",
            "film",
            "aim",
            "mental",
            "breathing",
        },
        "career": {"career", "school", "degree", "job", "college", "homeschool"},
        "competition": {
            "tournament",
            "tournaments",
            "qualify",
            "qualified",
            "rank",
            "ranked",
            "competitive",
            "winnings",
            "win",
            "won",
            "prize",
        },
        "money": {
            "money",
            "million",
            "thousand",
            "dollars",
            "bucks",
            "paid",
            "earned",
            "bought",
            "purchase",
            "cost",
            "invested",
            "investing",
            "salary",
        },
        "business": {"business", "company", "product", "brand", "sold", "sales", "customer"},
        "streaming": {"stream", "streaming", "twitch", "youtube", "viewers", "creator", "content"},
    }
    scores = {
        topic: sum(token_counts[token] for token in signals)
        for topic, signals in categories.items()
    }
    if "give away" in lowered:
        scores["giveaway"] += 3
    if "dropped out" in lowered:
        scores["career"] += 3
    if "world cup" in lowered:
        scores["competition"] += 3
    if "$" in text:
        scores["money"] += 2
    best_topic, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 1:
        return best_topic
    for keyword in campaign_keywords:
        normalized = keyword.lower().strip()
        if normalized in lowered and normalized not in {"fortnite", "gaming", "gamer"}:
            return normalized.replace(" ", "-")
    campaign_tokens = {keyword.lower() for keyword in campaign_keywords}
    counts = Counter(token for token in _content_tokens(text) if token not in campaign_tokens)
    words = [word for word, _ in counts.most_common(3)]
    if words:
        return "-".join(words)
    if "fortnite" in lowered or "gaming" in lowered or "gamer" in lowered:
        return "gaming"
    return "general"


def _setup_payoff(text: str) -> tuple[str, str]:
    sentences = _sentences(text)
    if not sentences:
        return "", ""
    return sentences[0][:220], sentences[-1][:220]


def discover_story_moments(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
    *,
    min_seconds: float = 22.0,
    max_seconds: float = 90.0,
) -> list[StoryMoment]:
    if not segments:
        return []
    moments: list[StoryMoment] = []
    start_index = 0
    index = 0
    while index < len(segments):
        start = segments[start_index].start
        current = segments[index]
        duration = current.end - start
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        next_gap = max(0.0, next_segment.start - current.end) if next_segment else 1.0
        recent_start = max(start_index, index - 3)
        recent_text = " ".join(item.text for item in segments[recent_start : index + 1])
        future_text = (
            " ".join(item.text for item in segments[index + 1 : min(len(segments), index + 5)])
            if next_segment
            else ""
        )
        topic_shift = bool(future_text) and semantic_similarity(recent_text, future_text) < 0.08
        natural_boundary = current.text.rstrip().endswith((".", "!", "?")) and (
            next_gap >= 0.45 or current.text.rstrip().endswith("?") or topic_shift
        )
        should_split = duration >= max_seconds or (duration >= min_seconds and natural_boundary)
        at_end = index + 1 == len(segments)
        if should_split or at_end:
            block = segments[start_index : index + 1]
            text = " ".join(item.text.strip() for item in block).strip()
            if text and current.end - block[0].start >= min(8.0, min_seconds):
                scores = score_editorial_text(
                    brief,
                    text,
                    start_text=block[0].text,
                    end_text=block[-1].text,
                    next_gap=next_gap,
                )
                setup, payoff = _setup_payoff(text)
                fingerprint = transcript_fingerprint(text)
                moments.append(
                    StoryMoment(
                        moment_id=f"{video_id}-moment-{len(moments) + 1:03d}",
                        video_id=video_id,
                        start=block[0].start,
                        end=block[-1].end,
                        text=text,
                        moment_type=_moment_type(text),
                        topic=_topic(text, brief.keywords),
                        setup=setup,
                        payoff=payoff,
                        scores=scores,
                        score=aggregate_editorial_score(scores, brief.editorial.score_weights),
                        transcript_fingerprint=fingerprint,
                    )
                )
            start_index = index + 1
        index += 1
    return moments


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union else 0.0


def _candidate_segments(
    moment: StoryMoment,
    segments: Sequence[TranscriptSegment],
    *,
    max_clip_seconds: float,
) -> list[TranscriptSegment]:
    # StoryMoment is a semantic discovery seed, not a hard clip boundary. A question/setup
    # near the end of a moment often resolves in the next moment, so permit a bounded
    # forward search while keeping starts anchored to the original moment.
    start_context = min(3.0, max_clip_seconds * 0.08)
    end_context = min(18.0, max_clip_seconds * 0.45)
    return [
        segment
        for segment in segments
        if segment.end > moment.start - start_context and segment.start < moment.end + end_context
    ]


def mine_clip_concepts(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
    moments: Sequence[StoryMoment],
    *,
    rejections: list[dict[str, object]] | None = None,
    stats: dict[str, int] | None = None,
) -> list[ClipConcept]:
    candidates: list[ClipConcept] = []
    candidate_starts = 0
    eligible_endpoints = 0
    concepts_after_quality = 0
    per_moment_limit = 4
    per_start_endpoint_limit = 2
    for moment in moments:
        local = _candidate_segments(moment, segments, max_clip_seconds=brief.max_clip_seconds)
        moment_candidates: list[ClipConcept] = []
        for start_index, first in enumerate(local):
            # Forward context exists only to complete the original moment; never seed a new
            # concept from the next semantic block. A tiny pre-roll allowance helps recover
            # questions split a cue or two before the moment boundary.
            if first.start < moment.start - 2.0 or first.start >= moment.end:
                continue
            candidate_starts += 1
            if start_boundary_score(first.text) < 1.8 and start_index not in {0, 1}:
                _record_rejection(
                    rejections,
                    video_id=video_id,
                    stage="editorial_quality",
                    reason="weak_start_boundary",
                    start=first.start,
                    end=first.end,
                    text=first.text,
                )
                continue

            text_parts: list[str] = []
            endpoint_options: list[tuple[float, float, ClipConcept]] = []
            saw_min_duration = False
            saw_housekeeping = False
            best_end_score = 0.0
            best_rejected_text = ""
            best_rejected_end = first.end
            for end_index in range(start_index, len(local)):
                ending = local[end_index]
                duration = ending.end - first.start
                if duration > brief.max_clip_seconds:
                    break
                text_parts.append(ending.text.strip())
                if duration < brief.min_clip_seconds:
                    continue
                saw_min_duration = True
                eligible_endpoints += 1
                next_segment = local[end_index + 1] if end_index + 1 < len(local) else None
                next_gap = max(0.0, next_segment.start - ending.end) if next_segment else 0.5
                text_value = " ".join(text_parts).strip()
                if any(phrase in text_value.lower() for phrase in _HOUSEKEEPING_PHRASES):
                    saw_housekeeping = True
                    continue

                end_score = end_boundary_score(ending.text, next_gap=next_gap)
                best_end_score = max(best_end_score, end_score)
                if end_score >= best_end_score:
                    best_rejected_text = text_value
                    best_rejected_end = ending.end
                scores = score_editorial_text(
                    brief,
                    text_value,
                    start_text=first.text,
                    end_text=ending.text,
                    next_gap=next_gap,
                )
                # Discovery is intentionally high-recall. Final EditPlan endpoint selection
                # still enforces semantic closure. Here, a weak endpoint reduces score instead
                # of deleting the whole story before later context can resolve it.
                if scores.hook_strength < 2.4:
                    continue
                if scores.story_completeness < 3.2:
                    continue
                score = aggregate_editorial_score(scores, brief.editorial.score_weights)
                closure_bonus = min(1.0, max(0.0, end_score - 3.0) * 0.18)
                ranked_score = score + closure_bonus
                topic = _topic(text_value, brief.keywords)
                setup, payoff = _setup_payoff(text_value)
                fingerprint = transcript_fingerprint(text_value)
                concepts_after_quality += 1
                endpoint_options.append(
                    (
                        end_score,
                        ranked_score,
                        ClipConcept(
                            concept_id=f"{video_id}-{fingerprint[:10]}",
                            video_id=video_id,
                            source_start=math.floor(first.start * 100) / 100,
                            source_end=math.ceil(ending.end * 100) / 100,
                            text=text_value,
                            topic=topic,
                            setup=setup,
                            payoff=payoff,
                            moment_type=_moment_type(text_value),
                            recommended_duration=round(duration, 3),
                            scores=scores,
                            score=score,
                            semantic_cluster="unassigned",
                            transcript_fingerprint=fingerprint,
                        ),
                    )
                )

            if endpoint_options:
                # Favor semantic closure first, then editorial score. Keep a second materially
                # different endpoint so downstream ranking can trade brevity against payoff.
                endpoint_options.sort(key=lambda item: (-item[0], -item[1], item[2].source_end))
                chosen: list[ClipConcept] = []
                for _end_score, _ranked, candidate in endpoint_options:
                    if any(
                        abs(candidate.source_end - prior.source_end) < 1.5
                        or semantic_similarity(candidate.text, prior.text) >= 0.92
                        for prior in chosen
                    ):
                        continue
                    chosen.append(candidate)
                    if len(chosen) >= per_start_endpoint_limit:
                        break
                moment_candidates.extend(chosen)
            elif saw_min_duration:
                reason = "podcast_housekeeping" if saw_housekeeping else "no_semantic_closure"
                _record_rejection(
                    rejections,
                    video_id=video_id,
                    stage="editorial_quality",
                    reason=reason,
                    start=first.start,
                    end=best_rejected_end,
                    text=best_rejected_text or first.text,
                )

        moment_candidates.sort(key=lambda item: (-item.score, item.source_start))
        kept: list[ClipConcept] = []
        for candidate in moment_candidates:
            if any(
                _overlap(
                    candidate.source_start, candidate.source_end, item.source_start, item.source_end
                )
                >= 0.68
                and semantic_similarity(candidate.text, item.text) >= 0.55
                for item in kept
            ):
                _record_rejection(
                    rejections,
                    video_id=video_id,
                    stage="semantic_dedup",
                    reason="near_duplicate_within_story_moment",
                    start=candidate.source_start,
                    end=candidate.source_end,
                    text=candidate.text,
                    scores=candidate.scores,
                )
                continue
            kept.append(candidate)
            if len(kept) >= per_moment_limit:
                break
        candidates.extend(kept)

    # Preserve episode-level recall while keeping the configured pool bounded. Candidate
    # variants are first collapsed to semantic representatives, then temporal floors prevent
    # a long episode's highest-scoring section from monopolizing the discovery budget.
    concepts_after_moment_dedup = len(candidates)
    if not candidates:
        if stats is not None:
            stats.update(
                {
                    "candidate_starts": candidate_starts,
                    "eligible_endpoints": eligible_endpoints,
                    "concepts_after_quality": concepts_after_quality,
                    "concepts_after_moment_dedup": 0,
                    "semantic_representatives": 0,
                    "raw_pool": 0,
                }
            )
        return []
    clustered = cluster_concepts(
        candidates, similarity_threshold=brief.diversity.semantic_similarity_threshold
    )
    cluster_groups: dict[str, list[ClipConcept]] = defaultdict(list)
    for candidate in clustered:
        cluster_groups[candidate.semantic_cluster].append(candidate)

    def opportunity_priority(item: ClipConcept) -> float:
        scores = item.scores
        return (
            item.score
            + scores.specificity * 0.10
            + scores.information_value * 0.08
            + scores.quoteability * 0.08
            + scores.curiosity * 0.06
            + scores.payoff_strength * 0.05
        )

    representatives = [
        max(items, key=lambda item: (opportunity_priority(item), -item.source_start))
        for items in cluster_groups.values()
    ]
    representatives.sort(key=lambda item: (-opportunity_priority(item), item.source_start))
    pool_size = brief.production.candidate_pool_size
    episode_end = max((segment.end for segment in segments), default=0.0)
    bucket_width = episode_end / 5 if episode_end else 0.0
    selected: list[ClipConcept] = []
    selected_ids: set[str] = set()
    if bucket_width:
        temporal_floor = max(1, min(5, pool_size // 7))
        for bucket in range(5):
            bucket_items = [
                item
                for item in representatives
                if min(4, int(item.source_start / bucket_width)) == bucket
            ]
            for item in bucket_items[:temporal_floor]:
                if len(selected) >= pool_size:
                    break
                selected.append(item)
                selected_ids.add(item.concept_id)
    for candidate in representatives:
        if len(selected) >= pool_size:
            break
        if candidate.concept_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.concept_id)
    selected.sort(key=lambda item: (-item.score, item.source_start))
    final_pool = selected[:pool_size]
    if stats is not None:
        stats.update(
            {
                "candidate_starts": candidate_starts,
                "eligible_endpoints": eligible_endpoints,
                "concepts_after_quality": concepts_after_quality,
                "concepts_after_moment_dedup": concepts_after_moment_dedup,
                "semantic_representatives": len(representatives),
                "raw_pool": len(final_pool),
            }
        )
    return final_pool


def cluster_concepts(
    concepts: Sequence[ClipConcept], *, similarity_threshold: float
) -> list[ClipConcept]:
    representatives: list[ClipConcept] = []
    clustered: list[ClipConcept] = []
    for concept in sorted(concepts, key=lambda item: (-item.score, item.source_start)):
        cluster_index: int | None = None
        for index, representative in enumerate(representatives):
            if semantic_similarity(concept.text, representative.text) >= similarity_threshold:
                cluster_index = index
                break
        if cluster_index is None:
            representatives.append(concept)
            cluster_index = len(representatives) - 1
        clustered.append(replace(concept, semantic_cluster=f"cluster-{cluster_index + 1:02d}"))
    return clustered


def select_distinct_concepts(
    brief: CampaignBrief,
    concepts: Sequence[ClipConcept],
    *,
    rejections: list[dict[str, object]] | None = None,
) -> list[ClipConcept]:
    clustered = cluster_concepts(
        concepts, similarity_threshold=brief.diversity.semantic_similarity_threshold
    )
    selected: list[ClipConcept] = []
    topic_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    def priority(item: ClipConcept) -> tuple[float, float]:
        scores = item.scores
        tokens = set(_tokens(item.text))
        concrete_money_claim = bool(_NUMBER_RE.search(item.text)) and bool(
            tokens & _MONEY_CLAIM_WORDS
        )
        age_or_year_claim = bool(_NUMBER_RE.search(item.text)) and bool(
            tokens & {"age", "aged", "year", "years", "old"}
        )
        campaign_bonus = max(0.0, scores.campaign_relevance - 2.0) * 0.14
        opportunity_bonus = (
            scores.specificity * 0.05
            + scores.information_value * 0.04
            + scores.quoteability * 0.04
            + scores.curiosity * 0.04
            + scores.payoff_strength * 0.06
            + (0.8 if concrete_money_claim else 0.0)
            + (0.35 if age_or_year_claim else 0.0)
        )
        return item.score + campaign_bonus + opportunity_bonus, -item.source_start

    for concept in sorted(clustered, key=priority, reverse=True):
        if topic_counts[concept.topic] >= brief.diversity.max_concepts_per_topic:
            _record_rejection(
                rejections,
                video_id=concept.video_id,
                stage="topic_diversity",
                reason="topic_quota",
                start=concept.source_start,
                end=concept.source_end,
                text=concept.text,
                scores=concept.scores,
            )
            continue
        if cluster_counts[concept.semantic_cluster] >= 1:
            _record_rejection(
                rejections,
                video_id=concept.video_id,
                stage="semantic_dedup",
                reason="semantic_cluster_duplicate",
                start=concept.source_start,
                end=concept.source_end,
                text=concept.text,
                scores=concept.scores,
            )
            continue
        selected.append(concept)
        topic_counts[concept.topic] += 1
        cluster_counts[concept.semantic_cluster] += 1
        if len(selected) >= brief.production.concept_count:
            break
    return selected


def _source_excerpt(text: str, *, max_words: int = 8) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words).upper()
    return (" ".join(words[:max_words]) + " …").upper()


def _trim_question_preamble(sentence: str) -> str:
    """Return the first real interrogative phrase without host/filler preamble."""
    match = re.search(
        r"\b(?:why|how|what(?:'s)?|when|where|who|did|do|can|would)\b",
        sentence,
        flags=re.I,
    )
    if match is None or match.start() == 0:
        return sentence.strip()
    trimmed = sentence[match.start() :].strip()
    return trimmed if len(_tokens(trimmed)) >= 4 else sentence.strip()


def _find_hook_sentence(text: str, mode: HookMode) -> str | None:
    candidates: list[tuple[float, str]] = []
    for sentence in _sentences(text):
        lowered = sentence.lower().strip()
        candidate = _trim_question_preamble(sentence) if mode == "question" else sentence.strip()
        tokens = _tokens(candidate)
        base = start_boundary_score(candidate)
        if mode == "question" and candidate.endswith("?"):
            quality = base + (1.0 if len(tokens) >= 5 else 0.0)
        elif mode == "number" and _HOOK_NUMBER_RE.search(sentence):
            number_count = len(_HOOK_NUMBER_RE.findall(sentence))
            purchase_bonus = (
                1.0
                if any(
                    word in tokens for word in {"bought", "made", "won", "paid", "earned", "cost"}
                )
                else 0.0
            )
            quality = base + min(2.0, number_count * 0.6) + purchase_bonus
        elif mode == "conflict" and any(word in tokens for word in _TENSION_WORDS):
            tension_count = sum(word in _TENSION_WORDS for word in tokens)
            quality = base + min(2.0, tension_count * 0.45)
        elif mode == "strong_opinion" and (
            "i think" in lowered
            or "i believe" in lowered
            or any(word in tokens for word in {"best", "worst", "never", "should", "truth"})
        ):
            quality = base + 1.0
        else:
            continue
        candidates.append((quality, candidate))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], len(item[1])))[1]


def _segment_for_text(
    segments: Sequence[TranscriptSegment], concept: ClipConcept, sentence: str
) -> TranscriptSegment | None:
    needle = " ".join(_tokens(sentence)[:6])
    for segment in segments:
        if segment.end <= concept.source_start or segment.start >= concept.source_end:
            continue
        haystack = " ".join(_tokens(segment.text))
        if needle and needle in haystack:
            return segment
    return None


def _hook_start_anchor(
    segments: Sequence[TranscriptSegment], concept: ClipConcept, sentence: str
) -> tuple[float, str] | None:
    anchor = find_phrase_anchor(segments, concept.source_start, concept.source_end, sentence)
    if anchor is not None:
        return anchor.source_start, anchor.text
    segment = _segment_for_text(segments, concept, sentence)
    if segment is None:
        return None
    fallback = first_complete_word(segments, segment.start, min(segment.end, concept.source_end))
    return (
        (fallback.source_start, fallback.text)
        if fallback is not None
        else (segment.start, sentence.split()[0])
    )


def _hook_start_time(
    segments: Sequence[TranscriptSegment], concept: ClipConcept, sentence: str
) -> float | None:
    anchor = _hook_start_anchor(segments, concept, sentence)
    return anchor[0] if anchor is not None else None


def generate_hook_variants(
    brief: CampaignBrief,
    concept: ClipConcept,
    segments: Sequence[TranscriptSegment],
) -> list[HookVariant]:
    variants: list[HookVariant] = []
    enabled = brief.hooks.enabled
    campaign_priority = max(0.0, concept.scores.campaign_relevance - 2.0) * 0.28
    for mode in enabled:
        start = concept.source_start
        end = concept.source_end
        overlay: str | None = None
        rationale = "strongest natural source boundary"
        score_delta = campaign_priority
        caption_anchor = first_complete_word(segments, start, end)
        if mode == "direct":
            score_delta += concept.scores.hook_strength * 0.05
        elif mode == "curiosity_text":
            overlay = _source_excerpt(concept.payoff or concept.setup)
            rationale = "truthful source-derived curiosity overlay"
            score_delta += concept.scores.curiosity * 0.06
        elif mode in {"question", "number", "conflict", "strong_opinion"}:
            sentence = _find_hook_sentence(concept.text, mode)
            if sentence is None:
                continue
            hook_anchor = _hook_start_anchor(segments, concept, sentence)
            hook_start = hook_anchor[0] if hook_anchor is not None else None
            if hook_start is not None and hook_start > concept.source_start:
                candidate_duration = end - hook_start
                if candidate_duration >= brief.min_clip_seconds:
                    start = hook_start
                    caption_anchor = (
                        first_complete_word(segments, start, end)
                        if hook_anchor is None
                        else find_phrase_anchor(segments, start, end, sentence)
                    )
            overlay = _source_excerpt(sentence)
            rationale = f"source-derived {mode.replace('_', ' ')} hook"
            source_quality = start_boundary_score(sentence)
            if mode == "question":
                score_delta += source_quality * 0.06 + concept.scores.curiosity * 0.035
            elif mode == "number":
                score_delta += source_quality * 0.055 + concept.scores.specificity * 0.04
            elif mode == "conflict":
                score_delta += source_quality * 0.05 + concept.scores.controversy_or_tension * 0.04
            else:
                score_delta += source_quality * 0.05 + concept.scores.quoteability * 0.04
        elif mode == "payoff_first":
            continue
        span = SourceSpan(round(start, 3), round(end, 3))
        fingerprint_seed = f"{concept.concept_id}|{span.start:.3f}|{span.end:.3f}|{overlay or ''}"
        fingerprint = hashlib.sha256(fingerprint_seed.encode("utf-8")).hexdigest()[:20]
        variants.append(
            HookVariant(
                variant_id=f"{concept.concept_id}-{mode}",
                concept_id=concept.concept_id,
                mode=mode,
                source_spans=(span,),
                overlay_text=overlay,
                score=round(concept.score + score_delta, 4),
                rationale=rationale,
                fingerprint=fingerprint,
                caption_start_source_time=(
                    round(caption_anchor.source_start, 3) if caption_anchor is not None else None
                ),
                caption_start_word=(caption_anchor.text if caption_anchor is not None else None),
            )
        )
    unique: dict[str, HookVariant] = {}
    for variant in sorted(variants, key=lambda item: (-item.score, item.variant_id)):
        unique.setdefault(variant.fingerprint, variant)
    return list(unique.values())[: brief.production.variants_per_concept]


def _signal_times(
    concept: ClipConcept, segments: Sequence[TranscriptSegment]
) -> Iterable[tuple[float, float]]:
    for segment in segments:
        if segment.end <= concept.source_start or segment.start >= concept.source_end:
            continue
        relative = segment.start - concept.source_start
        if relative < 3.5 or relative > concept.duration - 2.5:
            continue
        text = segment.text
        if _NUMBER_RE.search(text) or any(
            token in _HOOK_WORDS | _TENSION_WORDS for token in _tokens(text)
        ):
            yield relative, min(concept.duration, relative + min(1.25, segment.duration))


def _next_speech_start(segments: Sequence[TranscriptSegment], after: float) -> float | None:
    starts = [segment.start for segment in segments if segment.start >= after - 1e-6]
    return min(starts) if starts else None


def _semantic_end(
    concept: ClipConcept,
    span: SourceSpan,
    segments: Sequence[TranscriptSegment],
    *,
    max_end: float,
) -> float:
    end = min(span.end, max_end)
    unanswered_setup = concept.setup.rstrip().endswith(
        "?"
    ) and not concept.payoff.rstrip().endswith("?")
    if not unanswered_setup or concept.scores.payoff_strength >= 7.0:
        return end
    future = [
        segment
        for segment in segments
        if segment.start >= concept.source_end - 1e-6 and segment.end <= max_end + 1e-6
    ]
    for index, segment in enumerate(future):
        next_start = future[index + 1].start if index + 1 < len(future) else segment.end
        next_gap = max(0.0, next_start - segment.end)
        tokens = set(_tokens(segment.text))
        payoff_signal = bool(tokens & _PAYOFF_SIGNALS) or bool(_NUMBER_RE.search(segment.text))
        if payoff_signal and end_boundary_score(segment.text, next_gap=next_gap) >= 6.0:
            return segment.end
    return end


def _speech_aware_tail(
    end: float,
    segments: Sequence[TranscriptSegment],
    *,
    tail: float,
    max_end: float,
) -> float:
    desired = min(max_end, end + tail)
    next_start = _next_speech_start(segments, end + 1e-4)
    if next_start is not None and next_start < desired:
        return max(end, next_start)
    return desired


def _overlay_duplicates_opening(
    overlay: str | None, span: SourceSpan, segments: Sequence[TranscriptSegment]
) -> bool:
    if not overlay:
        return False
    words = [
        word.text
        for word in flatten_source_words(segments)
        if word.source_start >= span.start - 1e-6 and word.source_end <= span.end + 1e-6
    ][:8]
    opening = " ".join(words)
    left, right = normalized_tokens(overlay), normalized_tokens(opening)
    if not left or not right:
        return False
    shared = len(set(left) & set(right)) / max(1, min(len(set(left)), len(set(right))))
    return shared >= 0.8 or left[: min(5, len(left))] == right[: min(5, len(right))]


def build_edit_plan(
    brief: CampaignBrief,
    concept: ClipConcept,
    variant: HookVariant,
    segments: Sequence[TranscriptSegment],
) -> EditPlan:
    if len(variant.source_spans) != 1:
        raise ValueError("multi-span hook variants require the reorder renderer")
    span = variant.source_spans[0]
    tail = brief.editorial.post_speech_tail_seconds
    max_end = span.start + brief.max_clip_seconds
    semantic_end = _semantic_end(concept, span, segments, max_end=max_end)
    final_end = _speech_aware_tail(semantic_end, segments, tail=tail, max_end=max_end)
    span = SourceSpan(span.start, round(final_end, 3))
    beats: list[EditorialBeat] = []
    if brief.editorial.punch_ins_enabled:
        for start, end in _signal_times(concept, segments):
            adjusted_start = max(0.0, start - (span.start - concept.source_start))
            adjusted_end = min(span.duration, end - (span.start - concept.source_start))
            if adjusted_end - adjusted_start < 0.25:
                continue
            beats.append(EditorialBeat(adjusted_start, adjusted_end, "punch_in", 0.07))
            if (
                len([beat for beat in beats if beat.beat_type == "punch_in"])
                >= brief.editorial.max_punch_ins_per_clip
            ):
                break
    if span.duration >= 1.4:
        beats.append(EditorialBeat(span.duration - 1.25, span.duration, "payoff_hold", 0.0))
    return EditPlan(
        plan_id=f"plan-{variant.variant_id}",
        video_id=concept.video_id,
        concept_id=concept.concept_id,
        variant_id=variant.variant_id,
        hook_mode=variant.mode,
        source_spans=(span,),
        hook_text=(
            None
            if _overlay_duplicates_opening(variant.overlay_text, span, segments)
            else variant.overlay_text
        ),
        beats=tuple(beats),
        caption_platform=brief.editorial.platform,
        score=variant.score,
        transcript_fingerprint=concept.transcript_fingerprint,
        caption_start_source_time=variant.caption_start_source_time,
        caption_start_word=variant.caption_start_word,
    )


def select_render_plans(plans: Sequence[EditPlan], *, budget: int) -> list[EditPlan]:
    ranked = sorted(plans, key=lambda item: (-item.score, item.concept_id, item.variant_id))
    unique: list[EditPlan] = []
    used_concepts: set[str] = set()
    for plan in ranked:
        if plan.concept_id in used_concepts:
            continue
        unique.append(plan)
        used_concepts.add(plan.concept_id)

    selected: list[EditPlan] = []
    selected_ids: set[str] = set()
    video_ids = {plan.video_id for plan in unique}
    if budget >= 3 and len(video_ids) == 1 and unique:
        episode_end = max(span.end for plan in unique for span in plan.source_spans)
        third = episode_end / 3 if episode_end else 0.0

        def bucket(plan: EditPlan) -> int:
            if not third:
                return 0
            return min(2, int(plan.source_spans[0].start / third))

        # Seed one quality leader from every represented third. This prevents a long source's
        # strongest local section from hiding the rest of the episode without forcing even spacing.
        for period in range(3):
            candidates = [plan for plan in unique if bucket(plan) == period]
            if candidates and len(selected) < budget:
                best = candidates[0]
                selected.append(best)
                selected_ids.add(best.plan_id)
        max_per_period = max(1, math.ceil(budget * 0.5))
        period_counts = Counter(bucket(plan) for plan in selected)
        for plan in unique:
            if len(selected) >= budget:
                break
            if plan.plan_id in selected_ids:
                continue
            period = bucket(plan)
            if period_counts[period] >= max_per_period:
                continue
            selected.append(plan)
            selected_ids.add(plan.plan_id)
            period_counts[period] += 1

    for plan in unique:
        if len(selected) >= budget:
            return selected
        if plan.plan_id in selected_ids:
            continue
        selected.append(plan)
        selected_ids.add(plan.plan_id)
    for plan in ranked:
        if len(selected) >= budget:
            break
        if plan.plan_id in selected_ids:
            continue
        selected.append(plan)
        selected_ids.add(plan.plan_id)
    return selected


def select_render_plan_queue(
    plans: Sequence[EditPlan], *, budget: int
) -> tuple[list[EditPlan], list[EditPlan]]:
    primary = select_render_plans(plans, budget=budget)
    primary_ids = {plan.plan_id for plan in primary}
    ranked = sorted(plans, key=lambda item: (-item.score, item.concept_id, item.variant_id))
    reserves = [plan for plan in ranked if plan.plan_id not in primary_ids]
    return primary, reserves


def select_submission_shortlist(
    plans: Sequence[EditPlan], *, clip_count: int, max_per_source: int
) -> list[EditPlan]:
    selected: list[EditPlan] = []
    source_counts: Counter[str] = Counter()
    concepts: set[str] = set()
    for plan in sorted(plans, key=lambda item: (-item.score, item.concept_id)):
        if plan.concept_id in concepts or source_counts[plan.video_id] >= max_per_source:
            continue
        selected.append(plan)
        concepts.add(plan.concept_id)
        source_counts[plan.video_id] += 1
        if len(selected) >= clip_count:
            break
    return selected
