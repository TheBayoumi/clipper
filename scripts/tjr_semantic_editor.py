"""Source-level semantic editorial discovery for any configured campaign.

The semantic model discovers topic/event structure across the whole transcript.
Regex rules remain downstream safety checks; they are not the primary editor.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from clipper.models import CampaignBrief, ClipCandidate, TranscriptSegment
from clipper.tiktok import creative_hook_from_text, source_headline_candidates

SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"
CREATOR_MOMENT_DESCRIPTIONS = (
    "a self-contained funny or surprising podcast reaction, memorable quote, disagreement, "
    "challenge, reveal, flex, comparison, confession, story, or payoff",
    "a specific guest or host story with a clear setup and memorable consequence, lesson, "
    "punchline, surprising detail, or emotionally strong reaction",
)
EVENT_DESCRIPTIONS = {
    "reaction_surprise": (
        "a strong authentic reaction, surprise, disbelief, excitement, laughter, embarrassment, "
        "or unexpected moment"
    ),
    "memorable_quote_opinion": (
        "a memorable quote, bold opinion, provocative statement, unusual belief, or concise "
        "take that makes sense outside the full episode"
    ),
    "story_confession": (
        "a personal story, confession, behind-the-scenes anecdote, embarrassing moment, "
        "career story, or unusual experience"
    ),
    "conflict_disagreement": (
        "a disagreement, challenge, confrontation, controversial exchange, pushback, or "
        "strong difference of opinion"
    ),
    "comparison_flex": (
        "a comparison, challenge, flex, one-upmanship, ranking, money detail, status claim, "
        "or claim that one thing is better, rarer, harder, or worth more"
    ),
    "reveal_payoff": (
        "a reveal, discovery, shocking detail, unexpected answer, consequence, resolution, "
        "or satisfying payoff"
    ),
    "lesson_explanation": (
        "a specific explanation, lesson, insight, reasoning chain, advice, or why something matters"
    ),
}
OFF_TOPIC_DESCRIPTIONS = (
    "empty filler, housekeeping, greetings, repeated chatter, or transcript noise with no "
    "distinct reaction, claim, reveal, lesson, result, or memorable quote",
    "sponsor boilerplate, generic calls to action, navigation chatter, or technical noise "
    "without a creator moment",
)
MIN_EVENT_SIMILARITY = 0.34
# Structural rubric descriptions are embedded, never matched as speech keywords.
CONTEXT_RUBRIC = {
    "story": (
        "A self-contained podcast exchange establishes a specific situation or claim, "
        "develops it, and delivers a meaningful answer, consequence, insight or punchline.",
        "An isolated fragment, routine announcement or disconnected chatter has no "
        "standalone situation and meaningful development or answer.",
    ),
    "opening": (
        "The opening introduces an intelligible specific situation, claim or question "
        "that gives a new viewer a reason to hear the rest of the exchange.",
        "The opening is an acknowledgement or continuation whose subject and stakes "
        "cannot be understood without the preceding conversation.",
    ),
    "ending": (
        "The ending answers or resolves the preceding exchange with a consequence, "
        "insight, complete opinion or punchline that can stand on its own.",
        "The ending starts a new question or trails off before the answer, consequence "
        "or point of the preceding exchange is delivered.",
    ),
}
_WORD = re.compile(r"[A-Za-z0-9$%'.-]+")


@dataclass(frozen=True, slots=True)
class SemanticUnit:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


EmbeddingFn = Callable[[Sequence[str]], list[list[float]]]


class FastEmbedder:
    """Lazy ONNX embedding backend; no hosted API and no PyTorch dependency."""

    def __init__(self, model_name: str = SEMANTIC_MODEL) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "semantic editorial pass requires fastembed; install the production extras"
            ) from exc
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, threads=2)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [
            [float(value) for value in vector]
            for vector in self._model.embed(list(texts), batch_size=32)
        ]


def _embedding_batch(backend: EmbeddingFn, texts: Sequence[str], stage: str) -> list[list[float]]:
    unique = list(dict.fromkeys(texts))
    values = backend(unique) if unique else []
    if len(values) != len(unique):
        raise RuntimeError(f"{stage} backend returned an incomplete batch")
    width = len(values[0]) if values else 0
    if any(
        len(value) != width
        or not value
        or any(not math.isfinite(number) for number in value)
        or _norm(value) == 0
        for value in values
    ):
        raise RuntimeError(f"{stage} backend returned invalid embedding vectors")
    lookup = dict(zip(unique, values, strict=True))
    return [lookup[text] for text in texts]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= 0:
        return 0.0
    return max(
        -1.0,
        min(
            1.0,
            sum(float(a) * float(b) for a, b in zip(left, right, strict=False)) / denominator,
        ),
    )


def _average(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("semantic embedding dimensions are inconsistent")
    return [
        sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(width)
    ]


def _thought_units(segments: Sequence[TranscriptSegment]) -> list[SemanticUnit]:
    """Build natural thought units before any candidate window exists."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    units: list[SemanticUnit] = []
    group: list[TranscriptSegment] = []
    for index, segment in enumerate(ordered):
        if group and segment.start - group[-1].end >= 1.8:
            units.append(
                SemanticUnit(
                    group[0].start,
                    group[-1].end,
                    " ".join(item.text.strip() for item in group).strip(),
                )
            )
            group = []
        group.append(segment)
        next_segment = ordered[index + 1] if index + 1 < len(ordered) else None
        next_pause = next_segment.start - segment.end if next_segment is not None else 99.0
        duration = group[-1].end - group[0].start
        sentence_end = segment.text.rstrip().endswith((".", "!", "?"))
        if sentence_end or next_pause >= 0.65 or duration >= 9.0 or next_segment is None:
            text = " ".join(item.text.strip() for item in group).strip()
            if text:
                units.append(SemanticUnit(group[0].start, group[-1].end, text))
            group = []
    return units


def _extractive_hook(text: str, *, max_words: int = 16) -> str:
    """Keep complete source meaning; never clip a sentence at a word quota."""
    headline = creative_hook_from_text(text)
    return headline if len(_WORD.findall(headline)) <= max_words else ""


def _event_labels(
    unit_vectors: Sequence[Sequence[float]],
    event_vectors: Sequence[Sequence[float]],
) -> tuple[list[str], list[float]]:
    names = list(EVENT_DESCRIPTIONS)
    labels: list[str] = []
    strengths: list[float] = []
    for vector in unit_vectors:
        scores = [_cosine(vector, event_vector) for event_vector in event_vectors]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        labels.append(names[best_index])
        strengths.append(scores[best_index])
    return labels, strengths


def _region_ids(units: Sequence[SemanticUnit], vectors: Sequence[Sequence[float]]) -> list[int]:
    if not units:
        return []
    region = 0
    result = [region]
    for index in range(1, len(units)):
        pause = units[index].start - units[index - 1].end
        similarity = _cosine(vectors[index - 1], vectors[index])
        if pause >= 1.8 or similarity < 0.30:
            region += 1
        result.append(region)
    return result


def _anchor_window(
    index: int,
    units: Sequence[SemanticUnit],
    vectors: Sequence[Sequence[float]],
    regions: Sequence[int],
    *,
    min_seconds: float,
    max_seconds: float,
) -> tuple[int, int] | None:
    left = right = index
    region = regions[index]
    anchor_vector = vectors[index]
    # Preserve an actual spoken setup rather than greedily adding earlier filler.
    # Question words describe structure; the supplied brief controls topic relevance.
    question_start: int | None = None
    for preceding in range(index, -1, -1):
        if regions[preceding] != region:
            break
        if units[index].end - units[preceding].start > max_seconds:
            break
        text = units[preceding].text.strip()
        if "?" in text and _cosine(anchor_vector, vectors[preceding]) >= 0.48:
            question_start = preceding
            left = preceding
            break
    while units[right].end - units[left].start < min_seconds:
        options: list[tuple[float, int, int]] = []
        if question_start is None and left > 0 and regions[left - 1] == region:
            options.append((_cosine(anchor_vector, vectors[left - 1]), left - 1, right))
        if (
            right + 1 < len(units)
            and regions[right + 1] == region
            and not (question_start is not None and "?" in units[right + 1].text)
        ):
            options.append((_cosine(anchor_vector, vectors[right + 1]), left, right + 1))
        if not options:
            return None
        _, new_left, new_right = max(options, key=lambda item: item[0])
        if units[new_right].end - units[new_left].start > max_seconds:
            return None
        left, right = new_left, new_right

    if question_start is not None:
        # Stop after a complete answer; do not drag the next question into the clip.
        while (
            not units[right].text.rstrip().endswith((".", "!"))
            and right + 1 < len(units)
            and regions[right + 1] == region
            and units[right + 1].end - units[left].start <= max_seconds
            and "?" not in units[right + 1].text
        ):
            right += 1
        return left, right

    while True:
        options = []
        if question_start is None and left > 0 and regions[left - 1] == region:
            duration = units[right].end - units[left - 1].start
            similarity = _cosine(anchor_vector, vectors[left - 1])
            if duration <= max_seconds and similarity >= 0.48:
                options.append((similarity, left - 1, right))
        if right + 1 < len(units) and regions[right + 1] == region:
            duration = units[right + 1].end - units[left].start
            similarity = _cosine(anchor_vector, vectors[right + 1])
            if duration <= max_seconds and similarity >= 0.48:
                options.append((similarity, left, right + 1))
        if not options:
            break
        _, left, right = max(options, key=lambda item: item[0])
    return left, right


def _overlap(left: ClipCandidate, right: ClipCandidate) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


def build_semantic_editorial_candidates(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
    *,
    embedder: EmbeddingFn | None = None,
) -> tuple[list[ClipCandidate], dict[str, Any]]:
    """Rank topic-independent contextual review drafts, not publication verdicts."""
    units = _thought_units(segments)
    audit: dict[str, Any] = {
        "architecture": "podcast_contextual_editor_v4",
        "semantic_model": SEMANTIC_MODEL,
        "language_scope": "English transcripts; topic independent",
        "semantic_unit_count": len(units),
        "candidate_count": 0,
        "fixed_candidate_or_output_quota": False,
        "campaign_keyword_gate": False,
        "score_cutoff": None,
        "assessment_basis": "contrastive_embedding_proxy_requires_editorial_review",
    }
    if not units:
        return [], audit
    backend = embedder or FastEmbedder()
    texts = [unit.text for unit in units]
    event_texts = list(EVENT_DESCRIPTIONS.values())
    references = [*event_texts, *OFF_TOPIC_DESCRIPTIONS]
    embedded = _embedding_batch(backend, [*texts, *references], "semantic embedding")
    if len(embedded) != len(texts) + len(references):
        raise RuntimeError("semantic embedding backend returned an incomplete batch")
    vectors = embedded[: len(texts)]
    event_vectors = embedded[len(texts) : len(texts) + len(event_texts)]
    noise_vectors = embedded[len(texts) + len(event_texts) :]
    labels, strengths = _event_labels(vectors, event_vectors)
    regions = _region_ids(units, vectors)
    anchors = [
        index
        for index, vector in enumerate(vectors)
        if strengths[index] >= MIN_EVENT_SIMILARITY
        and strengths[index] > max(_cosine(vector, other) for other in noise_vectors)
    ]
    # For each anchor, retain the original proposal plus shorter whole-thought
    # alternatives. Context assessment chooses a repair before rejecting a span.
    windows: dict[tuple[int, int], int] = {}
    original_bounds: dict[int, tuple[int, int]] = {}
    for anchor in anchors:
        bounds = _anchor_window(
            anchor,
            units,
            vectors,
            regions,
            min_seconds=brief.min_clip_seconds,
            max_seconds=brief.max_clip_seconds,
        )
        if bounds is None:
            continue
        left, right = bounds
        original_bounds[anchor] = bounds
        for new_left in range(left, min(anchor, left + 2) + 1):
            for new_right in range(max(anchor, right - 2), right + 1):
                duration = units[new_right].end - units[new_left].start
                if brief.min_clip_seconds <= duration <= brief.max_clip_seconds:
                    key = (new_left, new_right)
                    previous = windows.get(key)
                    if previous is None or strengths[anchor] > strengths[previous]:
                        windows[key] = anchor
    spans = list(windows)
    rubric_texts = [text for pair in CONTEXT_RUBRIC.values() for text in pair]
    assessment_texts: list[str] = []
    for left, right in spans:
        assessment_texts.extend(
            [
                " ".join(unit.text for unit in units[left : right + 1]),
                units[left].text,
                " ".join(unit.text for unit in units[max(left, right - 1) : right + 1]),
            ]
        )
    context_vectors = _embedding_batch(
        backend, [*assessment_texts, *rubric_texts], "context assessment"
    )
    if len(context_vectors) != len(assessment_texts) + len(rubric_texts):
        raise RuntimeError("context assessment backend returned an incomplete batch")
    if any(len(vector) != len(vectors[0]) for vector in context_vectors):
        raise RuntimeError("context assessment embedding dimensions changed")
    rubric_vectors = context_vectors[len(assessment_texts) :]
    headline_choices = {
        bounds: source_headline_candidates(
            " ".join(unit.text for unit in units[bounds[0] : bounds[1] + 1])
        )
        for bounds in spans
    }
    unique_headlines = list(
        dict.fromkeys(hook for choices in headline_choices.values() for hook in choices)
    )
    headline_vectors = _embedding_batch(backend, unique_headlines, "headline assessment")
    if len(headline_vectors) != len(unique_headlines):
        raise RuntimeError("headline assessment backend returned an incomplete batch")
    headline_embeddings = dict(zip(unique_headlines, headline_vectors, strict=True))
    proposals: list[tuple[ClipCandidate, list[float]]] = []
    rejected_context = 0
    repaired_count = 0
    for index, (left, right) in enumerate(spans):
        anchor = windows[(left, right)]
        margins = {
            name: _cosine(context_vectors[index * 3 + dim], rubric_vectors[dim * 2])
            - _cosine(context_vectors[index * 3 + dim], rubric_vectors[dim * 2 + 1])
            for dim, name in enumerate(CONTEXT_RUBRIC)
        }
        # No absolute quality-score floor: positive contrast establishes a draft
        # worth reviewing, never proof of payoff, emotion or publication quality.
        if margins["story"] <= 0 or margins["ending"] <= 0:
            rejected_context += 1
            continue
        text = " ".join(unit.text for unit in units[left : right + 1])
        choices = headline_choices[(left, right)]
        if not choices:
            continue
        # Choose source-grounded headlines by relation to the full exchange;
        # punctuation and reaction tokens confer no editorial-score bonus.
        whole = context_vectors[index * 3]
        hook = max(choices, key=lambda choice: _cosine(whole, headline_embeddings[choice]))
        coherence_pairs = [_cosine(vectors[i - 1], vectors[i]) for i in range(left + 1, right + 1)]
        coherence = sum(coherence_pairs) / len(coherence_pairs) if coherence_pairs else 1.0
        score = 100 * (margins["story"] + margins["ending"] + margins["opening"]) + 20 * coherence
        reasons = (
            "candidate_origin=podcast_contextual_editor",
            f"semantic_model={SEMANTIC_MODEL}",
            f"semantic_event={labels[anchor]}",
            f"event_similarity={strengths[anchor]:.4f}",
            f"semantic_coherence={coherence:.4f}",
            *(f"context_{name}_margin={value:.12g}" for name, value in margins.items()),
            f"semantic_hook={hook}",
            f"boundary_repaired={str((left, right) != original_bounds[anchor]).lower()}",
            "start_boundary=whole_source_thought",
            "end_boundary=context_assessed_whole_source_thought",
        )
        proposals.append(
            (
                ClipCandidate(
                    video_id,
                    math.floor(units[left].start * 10) / 10,
                    math.ceil(units[right].end * 10) / 10,
                    text,
                    round(score, 4),
                    reasons,
                ),
                whole,
            )
        )
    selected: list[tuple[ClipCandidate, list[float]]] = []
    for candidate, vector in sorted(proposals, key=lambda item: (-item[0].score, item[0].start)):
        if any(
            _overlap(candidate, other) >= 0.55 or _cosine(vector, other_vector) >= 0.90
            for other, other_vector in selected
        ):
            continue
        selected.append((candidate, vector))
        repaired_count += int("boundary_repaired=true" in candidate.reasons)
    candidates = [candidate for candidate, _ in selected]
    audit.update(
        {
            "semantic_region_count": len(set(regions)),
            "event_anchor_count": len(anchors),
            "boundary_variants_assessed": len(spans),
            "context_rejected_variants": rejected_context,
            "boundary_repaired_candidate_count": repaired_count,
            "candidate_count": len(candidates),
            "event_distribution": dict(Counter(labels[index] for index in anchors)),
            "context_rubric": CONTEXT_RUBRIC,
            "campaign_relevance_policy": "source_eligibility_separate_from_editorial_topic",
        }
    )
    return candidates, audit
