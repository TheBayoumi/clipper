"""Source-level semantic editorial discovery for TJR.

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

SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"
CREATOR_MOMENT_DESCRIPTIONS = (
    "a self-contained funny or surprising creator reaction, memorable quote, disagreement, "
    "challenge, reveal, flex, comparison, or payoff",
    "a concrete trading psychology, risk-management, market decision, financial result, "
    "trading lesson, or money-related creator moment",
)
EVENT_DESCRIPTIONS = {
    "reaction_surprise": (
        "a strong authentic reaction, surprise, disbelief, excitement, laughter, or "
        "unexpected moment"
    ),
    "comparison_challenge": (
        "a comparison, challenge, flex, one-upmanship, ranking, or claim that one thing "
        "is better, rarer, or worth more"
    ),
    "reveal_showcase": (
        "a reveal, discovery, showcase, collection item, unusual detail, or visual payoff"
    ),
    "story_payoff": "a clear setup, escalation, consequence, resolution, or what happened next",
    "explanation_argument": (
        "a specific explanation, disagreement, opinion, reasoning chain, or why something matters"
    ),
    "trading_decision": (
        "a concrete financial-market trade, setup, entry, exit, thesis, or decision"
    ),
    "risk_management": (
        "risk management, stop loss, position sizing, discipline, protecting capital, "
        "or a trading lesson"
    ),
}
OFF_TOPIC_DESCRIPTIONS = (
    "empty filler, housekeeping, greetings, repeated chatter, or transcript noise with no "
    "distinct reaction, claim, reveal, lesson, result, or memorable quote",
    "sponsor boilerplate, generic calls to action, navigation chatter, or technical noise "
    "without a creator moment",
)
MIN_EVENT_SIMILARITY = 0.34
MIN_CAMPAIGN_SIMILARITY = 0.30
MIN_RELEVANCE_MARGIN = 0.02
_WORD = re.compile(r"[A-Za-z0-9$%'.-]+")
_LEADING_FILLER = re.compile(
    r"^(?:(?:okay|ok|so|well|basically|right|alright|all right|you know|i mean)\b[ ,.-]*)+",
    re.IGNORECASE,
)


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
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(value) for value in vector] for vector in self._model.embed(list(texts))]


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


def _extractive_hook(text: str, *, max_words: int = 12) -> str:
    """Use a source-extractive, semantically salient phrase instead of a fixed template."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = _LEADING_FILLER.sub("", cleaned).strip(" ,.-")
    words = _WORD.findall(cleaned)
    if len(words) < 3:
        return ""
    return " ".join(words[:max_words]).upper()


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
    while units[right].end - units[left].start < min_seconds:
        options: list[tuple[float, int, int]] = []
        if left > 0 and regions[left - 1] == region:
            options.append((_cosine(anchor_vector, vectors[left - 1]), left - 1, right))
        if right + 1 < len(units) and regions[right + 1] == region:
            options.append((_cosine(anchor_vector, vectors[right + 1]), left, right + 1))
        if not options:
            return None
        _, new_left, new_right = max(options, key=lambda item: item[0])
        if units[new_right].end - units[new_left].start > max_seconds:
            return None
        left, right = new_left, new_right

    while True:
        options = []
        if left > 0 and regions[left - 1] == region:
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
    """Discover complete 0..N candidate stories from full-source semantic structure."""
    units = _thought_units(segments)
    if not units:
        return [], {
            "architecture": "source_level_semantic_campaign_event_segmentation_v3",
            "semantic_model": SEMANTIC_MODEL,
            "semantic_unit_count": 0,
            "campaign_relevant_unit_count": 0,
            "out_of_domain_unit_count": 0,
            "event_anchor_count": 0,
            "candidate_count": 0,
            "event_distribution": {},
            "fixed_candidate_or_output_quota": False,
        }
    backend = embedder or FastEmbedder()
    texts = [unit.text for unit in units]
    event_texts = list(EVENT_DESCRIPTIONS.values())
    campaign_texts = [
        f"{brief.title}. {brief.objective}.",
        *[f"{brief.title} creator moment about {keyword}" for keyword in brief.keywords],
        *CREATOR_MOMENT_DESCRIPTIONS,
    ]
    reference_texts = [*event_texts, *campaign_texts, *OFF_TOPIC_DESCRIPTIONS]
    embedded = backend([*texts, *reference_texts])
    expected = len(texts) + len(reference_texts)
    if len(embedded) != expected:
        raise RuntimeError("semantic embedding backend returned an incomplete batch")
    unit_vectors = embedded[: len(texts)]
    references = embedded[len(texts) :]
    event_count = len(event_texts)
    campaign_count = len(campaign_texts)
    event_vectors = references[:event_count]
    campaign_vectors = references[event_count : event_count + campaign_count]
    off_topic_vectors = references[event_count + campaign_count :]
    labels, strengths = _event_labels(unit_vectors, event_vectors)
    regions = _region_ids(units, unit_vectors)
    campaign_scores = [
        max((_cosine(vector, prototype) for prototype in campaign_vectors), default=0.0)
        for vector in unit_vectors
    ]
    off_topic_scores = [
        max((_cosine(vector, other) for other in off_topic_vectors), default=0.0)
        for vector in unit_vectors
    ]
    relevance_margins = [
        campaign - off_topic
        for campaign, off_topic in zip(campaign_scores, off_topic_scores, strict=True)
    ]

    # First decide whether a unit belongs to this campaign. Only then may it be
    # classified into an editorial event. This prevents a closed event taxonomy
    # from forcing unrelated fashion/lifestyle speech into a trading label.
    relevant_units = [
        index
        for index in range(len(units))
        if campaign_scores[index] >= MIN_CAMPAIGN_SIMILARITY
        and relevance_margins[index] >= MIN_RELEVANCE_MARGIN
    ]
    anchors = [
        index
        for index in relevant_units
        if strengths[index] >= MIN_EVENT_SIMILARITY and len(_WORD.findall(units[index].text)) >= 3
    ]
    provisional: list[tuple[ClipCandidate, list[float]]] = []
    seen_windows: set[tuple[int, int]] = set()
    for anchor in anchors:
        bounds = _anchor_window(
            anchor,
            units,
            unit_vectors,
            regions,
            min_seconds=brief.min_clip_seconds,
            max_seconds=brief.max_clip_seconds,
        )
        if bounds is None or bounds in seen_windows:
            continue
        seen_windows.add(bounds)
        left, right = bounds
        window_units = units[left : right + 1]
        text = " ".join(unit.text for unit in window_units).strip()
        if not text:
            continue
        start = max(0.0, math.floor(window_units[0].start * 10) / 10)
        end = math.ceil(window_units[-1].end * 10) / 10
        if not brief.min_clip_seconds <= end - start <= brief.max_clip_seconds:
            continue
        local_vectors = unit_vectors[left : right + 1]
        coherence_pairs = [
            _cosine(local_vectors[index - 1], local_vectors[index])
            for index in range(1, len(local_vectors))
        ]
        coherence = sum(coherence_pairs) / len(coherence_pairs) if coherence_pairs else 1.0
        event_strength = strengths[anchor]
        hook = _extractive_hook(units[anchor].text)
        if not hook:
            continue
        score = round(
            70 * event_strength + 20 * max(0.0, coherence) + 20 * max(0.0, campaign_scores[anchor]),
            4,
        )
        reasons = (
            "candidate_origin=source_level_semantic_event",
            f"semantic_model={SEMANTIC_MODEL}",
            f"semantic_event={labels[anchor]}",
            f"event_similarity={event_strength:.4f}",
            f"campaign_relevance={campaign_scores[anchor]:.4f}",
            f"off_topic_similarity={off_topic_scores[anchor]:.4f}",
            f"relevance_margin={relevance_margins[anchor]:.4f}",
            f"semantic_coherence={coherence:.4f}",
            f"semantic_region={regions[anchor]}",
            f"semantic_hook={hook}",
            "start_boundary=semantic_thought_boundary",
            "end_boundary=semantic_thought_boundary",
        )
        provisional.append(
            (ClipCandidate(video_id, start, end, text, score, reasons), _average(local_vectors))
        )

    ordered = sorted(provisional, key=lambda item: (-item[0].score, item[0].start))
    selected: list[tuple[ClipCandidate, list[float]]] = []
    for candidate, vector in ordered:
        duplicate = False
        for existing, existing_vector in selected:
            if _overlap(candidate, existing) >= 0.55 or _cosine(vector, existing_vector) >= 0.90:
                duplicate = True
                break
        if not duplicate:
            selected.append((candidate, vector))

    event_distribution = Counter(labels[index] for index in anchors)
    candidates = [candidate for candidate, _ in selected]
    audit = {
        "architecture": "source_level_semantic_campaign_event_segmentation_v3",
        "semantic_model": SEMANTIC_MODEL,
        "semantic_unit_count": len(units),
        "semantic_region_count": len(set(regions)),
        "campaign_relevant_unit_count": len(relevant_units),
        "out_of_domain_unit_count": len(units) - len(relevant_units),
        "event_anchor_count": len(anchors),
        "candidate_count": len(candidates),
        "event_distribution": dict(event_distribution),
        "campaign_domain_gate": {
            "minimum_similarity": MIN_CAMPAIGN_SIMILARITY,
            "minimum_margin_over_non_campaign": MIN_RELEVANCE_MARGIN,
            "positive_prototype_count": len(campaign_texts),
            "negative_prototype_count": len(OFF_TOPIC_DESCRIPTIONS),
            "positive_policy": "campaign_objective_keywords_plus_creator_moment_prototypes",
        },
        "campaign_relevance_policy": "embedding_contrast_against_generic_non_moments",
        "candidate_policy": "campaign_relevance_then_event_then_safety_and_visual_qualification",
        "fixed_candidate_or_output_quota": False,
    }
    return candidates, audit
