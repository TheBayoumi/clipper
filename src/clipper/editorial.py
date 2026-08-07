from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
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


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in _WORD_RE.findall(text)]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in _STOP and len(token) > 2]


def transcript_fingerprint(text: str) -> str:
    normalized = " ".join(_tokens(text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


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


def start_boundary_score(text: str) -> float:
    stripped = text.strip()
    lowered = stripped.lower()
    tokens = _tokens(stripped[:180])
    score = 4.5
    if not stripped:
        return 0.0
    if _starts_weak(stripped):
        score -= 3.2
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
    standalone = 7.2 - (2.0 if _starts_weak(text) else 0.0)
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
    for keyword in campaign_keywords:
        if keyword.lower() in lowered:
            return keyword.lower().replace(" ", "-")
    counts = Counter(_content_tokens(text))
    words = [word for word, _ in counts.most_common(3)]
    return "-".join(words) if words else "general"


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
    moment: StoryMoment, segments: Sequence[TranscriptSegment]
) -> list[TranscriptSegment]:
    return [
        segment for segment in segments if segment.end > moment.start and segment.start < moment.end
    ]


def mine_clip_concepts(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
    moments: Sequence[StoryMoment],
) -> list[ClipConcept]:
    candidates: list[ClipConcept] = []
    per_moment_limit = 4
    for moment in moments:
        local = _candidate_segments(moment, segments)
        moment_candidates: list[ClipConcept] = []
        for start_index, first in enumerate(local):
            if start_boundary_score(first.text) < 3.1 and start_index not in {0, 1}:
                continue
            text_parts: list[str] = []
            for end_index in range(start_index, len(local)):
                ending = local[end_index]
                duration = ending.end - first.start
                if duration > brief.max_clip_seconds:
                    break
                text_parts.append(ending.text.strip())
                if duration < brief.min_clip_seconds:
                    continue
                next_segment = local[end_index + 1] if end_index + 1 < len(local) else None
                next_gap = max(0.0, next_segment.start - ending.end) if next_segment else 0.5
                if (
                    brief.editorial.semantic_endings
                    and end_boundary_score(ending.text, next_gap=next_gap) < 4.8
                ):
                    continue
                text = " ".join(text_parts).strip()
                if any(phrase in text.lower() for phrase in _HOUSEKEEPING_PHRASES):
                    continue
                scores = score_editorial_text(
                    brief,
                    text,
                    start_text=first.text,
                    end_text=ending.text,
                    next_gap=next_gap,
                )
                score = aggregate_editorial_score(scores, brief.editorial.score_weights)
                if scores.hook_strength < 2.4 or scores.story_completeness < 4.0:
                    continue
                topic = _topic(text, brief.keywords)
                setup, payoff = _setup_payoff(text)
                fingerprint = transcript_fingerprint(text)
                moment_candidates.append(
                    ClipConcept(
                        concept_id=f"{video_id}-{fingerprint[:10]}",
                        video_id=video_id,
                        source_start=math.floor(first.start * 100) / 100,
                        source_end=math.ceil(ending.end * 100) / 100,
                        text=text,
                        topic=topic,
                        setup=setup,
                        payoff=payoff,
                        moment_type=_moment_type(text),
                        recommended_duration=round(duration, 3),
                        scores=scores,
                        score=score,
                        semantic_cluster="unassigned",
                        transcript_fingerprint=fingerprint,
                    )
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
                continue
            kept.append(candidate)
            if len(kept) >= per_moment_limit:
                break
        candidates.extend(kept)
    candidates.sort(key=lambda item: (-item.score, item.source_start))
    return candidates[: brief.production.candidate_pool_size]


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
    brief: CampaignBrief, concepts: Sequence[ClipConcept]
) -> list[ClipConcept]:
    clustered = cluster_concepts(
        concepts, similarity_threshold=brief.diversity.semantic_similarity_threshold
    )
    selected: list[ClipConcept] = []
    topic_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    def priority(item: ClipConcept) -> tuple[float, float]:
        campaign_bonus = max(0.0, item.scores.campaign_relevance - 2.0) * 0.28
        return item.score + campaign_bonus, -item.source_start

    for concept in sorted(clustered, key=priority, reverse=True):
        if topic_counts[concept.topic] >= brief.diversity.max_concepts_per_topic:
            continue
        if cluster_counts[concept.semantic_cluster] >= 1:
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


def _find_hook_sentence(text: str, mode: HookMode) -> str | None:
    candidates: list[tuple[float, str]] = []
    for sentence in _sentences(text):
        lowered = sentence.lower().strip()
        tokens = _tokens(sentence)
        base = start_boundary_score(sentence)
        if mode == "question" and (sentence.endswith("?") or lowered.startswith(_QUESTION_OPENERS)):
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
        candidates.append((quality, sentence))
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


def _hook_start_time(
    segments: Sequence[TranscriptSegment], concept: ClipConcept, sentence: str
) -> float | None:
    needle = _tokens(sentence)[:5]
    if not needle:
        return None
    timed_words = [
        word
        for segment in segments
        if segment.end > concept.source_start and segment.start < concept.source_end
        for word in segment.words
        if word.end > concept.source_start and word.start < concept.source_end
    ]
    flattened = [(_tokens(word.text)[0] if _tokens(word.text) else "") for word in timed_words]
    for index in range(0, max(0, len(flattened) - len(needle) + 1)):
        if flattened[index : index + len(needle)] == needle:
            return timed_words[index].start
    segment = _segment_for_text(segments, concept, sentence)
    return segment.start if segment is not None else None


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
            hook_start = _hook_start_time(segments, concept, sentence)
            if hook_start is not None and hook_start > concept.source_start:
                candidate_duration = end - hook_start
                if candidate_duration >= brief.min_clip_seconds:
                    start = hook_start
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
    span = SourceSpan(span.start, round(min(max_end, span.end + tail), 3))
    beats: list[EditorialBeat] = []
    for start, end in _signal_times(concept, segments):
        adjusted_start = max(0.0, start - (span.start - concept.source_start))
        adjusted_end = min(span.duration, end - (span.start - concept.source_start))
        if adjusted_end - adjusted_start < 0.25:
            continue
        beats.append(EditorialBeat(adjusted_start, adjusted_end, "punch_in", 0.07))
        if len(beats) >= brief.editorial.max_punch_ins_per_clip:
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
        hook_text=variant.overlay_text,
        beats=tuple(beats),
        caption_platform=brief.editorial.platform,
        score=variant.score,
        transcript_fingerprint=concept.transcript_fingerprint,
    )


def select_render_plans(plans: Sequence[EditPlan], *, budget: int) -> list[EditPlan]:
    ranked = sorted(plans, key=lambda item: (-item.score, item.concept_id, item.variant_id))
    selected: list[EditPlan] = []
    used_concepts: set[str] = set()
    for plan in ranked:
        if plan.concept_id in used_concepts:
            continue
        selected.append(plan)
        used_concepts.add(plan.concept_id)
        if len(selected) >= budget:
            return selected
    for plan in ranked:
        if plan in selected:
            continue
        selected.append(plan)
        if len(selected) >= budget:
            break
    return selected


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
