from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import speech

DEFAULT_SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_EVENT_DESCRIPTIONS = {
    "reaction_surprise": (
        "a strong authentic reaction, surprise, disbelief, excitement, embarrassment, "
        "or unexpected moment"
    ),
    "memorable_quote_opinion": (
        "a memorable quote, bold opinion, provocative statement, unusual belief, or "
        "concise take that makes sense outside the full source"
    ),
    "story_experience": (
        "a specific story, behind-the-scenes detail, unusual experience, or concrete "
        "example with a meaningful consequence"
    ),
    "conflict_stakes": (
        "a disagreement, threat, risk, confrontation, difficult choice, or explanation "
        "of meaningful stakes"
    ),
    "reveal_payoff": (
        "a reveal, discovery, surprising detail, unexpected answer, consequence, "
        "resolution, or satisfying payoff"
    ),
    "lesson_explanation": (
        "a specific explanation, insight, reasoning chain, mechanic, strategy, or why "
        "something matters"
    ),
}

DEFAULT_OFF_TOPIC_DESCRIPTIONS = (
    "empty filler, housekeeping, greetings, repeated chatter, or transcript noise "
    "without a distinct claim, reveal, explanation, result, or memorable moment",
    "generic promotional boilerplate, navigation chatter, or technical noise without "
    "a self-contained idea or payoff",
)

DEFAULT_CONTEXT_RUBRIC = {
    "story": (
        "A self-contained spoken exchange establishes a specific situation or claim, "
        "develops it, and delivers a meaningful answer, consequence, insight, or payoff.",
        "An isolated fragment or disconnected chatter has no standalone situation and "
        "meaningful development or answer.",
    ),
    "opening": (
        "The opening introduces an intelligible specific situation, claim, or question "
        "that a new viewer can understand without earlier context.",
        "The opening is an acknowledgement or continuation whose subject cannot be "
        "understood without preceding speech.",
    ),
    "ending": (
        "The ending resolves the selected exchange with an answer, consequence, insight, "
        "complete opinion, or payoff.",
        "The ending starts a new unanswered question or trails off before the selected "
        "idea is resolved.",
    ),
}


@dataclass(frozen=True)
class SemanticUnit:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SemanticWindow:
    start: float
    end: float
    text: str
    anchor_time: float
    event_label: str
    event_similarity: float
    opening_quality: float
    story_quality: float
    ending_quality: float
    coherence: float
    score: float
    vector: tuple[float, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


EmbeddingFn = Callable[[Sequence[str]], list[list[float]]]


class FastEmbedder:
    """Lazy local semantic embedding backend owned by Clipper."""

    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "semantic spoken-content discovery requires fastembed; "
                "install the campaign analysis runtime"
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


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
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


def _embedding_batch(backend: EmbeddingFn, texts: Sequence[str], stage: str) -> list[list[float]]:
    unique = list(dict.fromkeys(texts))
    values = backend(unique) if unique else []
    if len(values) != len(unique):
        raise RuntimeError(f"{stage} backend returned an incomplete batch")
    width = len(values[0]) if values else 0
    if any(
        not value
        or len(value) != width
        or any(not math.isfinite(number) for number in value)
        or _norm(value) == 0
        for value in values
    ):
        raise RuntimeError(f"{stage} backend returned invalid embedding vectors")
    lookup = dict(zip(unique, values, strict=True))
    return [lookup[text] for text in texts]


def _closed_ending(text: str) -> bool:
    ending = text.rstrip().rstrip(chr(34) + chr(39) + chr(0x201D) + chr(0x2019))
    return ending.endswith((".", "!")) and not ending.endswith("...")


def thought_units(transcript: speech.Transcript) -> list[SemanticUnit]:
    statements = list(speech.statement_segments(transcript, maximum_pause_seconds=1.8))
    if not statements:
        statements = list(transcript.segments)
    ordered = sorted(statements, key=lambda item: (item.start, item.end))
    units: list[SemanticUnit] = []
    group: list[speech.TranscriptSegment] = []
    for index, segment in enumerate(ordered):
        if group and segment.start - group[-1].end >= 1.8:
            text = " ".join(item.text.strip() for item in group).strip()
            if text:
                units.append(SemanticUnit(group[0].start, group[-1].end, text))
            group = []
        group.append(segment)
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        next_pause = following.start - segment.end if following is not None else 99.0
        duration = group[-1].end - group[0].start
        sentence_end = segment.text.rstrip().endswith((".", "!", "?"))
        if sentence_end or next_pause >= 0.65 or duration >= 9.0 or following is None:
            text = " ".join(item.text.strip() for item in group).strip()
            if text:
                units.append(SemanticUnit(group[0].start, group[-1].end, text))
            group = []
    return units


def _region_ids(units: Sequence[SemanticUnit], vectors: Sequence[Sequence[float]]) -> list[int]:
    if not units:
        return []
    region = 0
    result = [region]
    for index in range(1, len(units)):
        pause = units[index].start - units[index - 1].end
        similarity = cosine(vectors[index - 1], vectors[index])
        if pause >= 1.8 or similarity < 0.30:
            region += 1
        result.append(region)
    return result


def _window_bounds(
    anchor: int,
    units: Sequence[SemanticUnit],
    regions: Sequence[int],
    minimum: float,
    maximum: float,
) -> list[tuple[int, int]]:
    region = regions[anchor]
    left_floor = anchor
    while left_floor > 0 and regions[left_floor - 1] == region:
        if units[anchor].end - units[left_floor - 1].start > maximum:
            break
        left_floor -= 1
    right_ceiling = anchor
    while right_ceiling + 1 < len(units) and regions[right_ceiling + 1] == region:
        if units[right_ceiling + 1].end - units[anchor].start > maximum:
            break
        right_ceiling += 1

    result: list[tuple[int, int]] = []
    for left in range(left_floor, anchor + 1):
        for right in range(anchor, right_ceiling + 1):
            duration = units[right].end - units[left].start
            if duration < minimum - 1e-3 or duration > maximum + 1e-3:
                continue
            if not _closed_ending(units[right].text):
                continue
            question_indexes = [
                index for index in range(left, right + 1) if "?" in units[index].text
            ]
            if question_indexes and question_indexes[-1] == right:
                continue
            result.append((left, right))
    return result


def _quality(margin: float) -> float:
    return max(0.0, min(1.0, 0.5 + 1.5 * margin))


def discover_windows(
    transcript: speech.Transcript,
    config: dict[str, Any],
    backend: EmbeddingFn,
) -> tuple[list[SemanticWindow], dict[str, Any]]:
    semantic_cfg = dict(config.get("spoken_content", {}).get("semantic") or {})
    editor = dict(config.get("semantic_editor") or {})
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 12.0))
    event_threshold = float(semantic_cfg.get("minimum_event_similarity", 0.34))
    minimum_context_margin = float(semantic_cfg.get("minimum_context_margin", 0.0))

    units = thought_units(transcript)
    diagnostics: dict[str, Any] = {
        "semantic_model": str(semantic_cfg.get("model") or DEFAULT_SEMANTIC_MODEL),
        "semantic_unit_count": len(units),
        "event_anchor_count": 0,
        "boundary_variants_assessed": 0,
        "context_rejected_variants": 0,
        "candidate_count": 0,
        "campaign_keyword_gate": False,
        "fixed_candidate_quota": False,
    }
    if not units:
        return [], diagnostics

    event_descriptions = dict(DEFAULT_EVENT_DESCRIPTIONS)
    configured_events = semantic_cfg.get("event_descriptions")
    if isinstance(configured_events, dict) and configured_events:
        event_descriptions = {
            str(key): str(value) for key, value in configured_events.items() if str(value).strip()
        }
    off_topic = [
        str(item)
        for item in semantic_cfg.get("off_topic_descriptions", DEFAULT_OFF_TOPIC_DESCRIPTIONS)
        if str(item).strip()
    ]
    texts = [unit.text for unit in units]
    event_names = list(event_descriptions)
    references = [*event_descriptions.values(), *off_topic]
    embedded = _embedding_batch(backend, [*texts, *references], "semantic discovery")
    vectors = embedded[: len(texts)]
    event_vectors = embedded[len(texts) : len(texts) + len(event_names)]
    noise_vectors = embedded[len(texts) + len(event_names) :]

    labels: list[str] = []
    strengths: list[float] = []
    for vector in vectors:
        scores = [cosine(vector, event_vector) for event_vector in event_vectors]
        best = max(range(len(scores)), key=scores.__getitem__)
        labels.append(event_names[best])
        strengths.append(scores[best])

    regions = _region_ids(units, vectors)
    anchors = [
        index
        for index, vector in enumerate(vectors)
        if strengths[index] >= event_threshold
        and (
            not noise_vectors
            or strengths[index] > max(cosine(vector, noise) for noise in noise_vectors)
        )
    ]
    diagnostics["event_anchor_count"] = len(anchors)
    diagnostics["semantic_region_count"] = len(set(regions))
    diagnostics["event_distribution"] = {
        name: sum(1 for index in anchors if labels[index] == name) for name in event_names
    }

    bounds_to_anchor: dict[tuple[int, int], int] = {}
    for anchor in anchors:
        for bounds in _window_bounds(anchor, units, regions, minimum, maximum):
            previous = bounds_to_anchor.get(bounds)
            if previous is None or strengths[anchor] > strengths[previous]:
                bounds_to_anchor[bounds] = anchor
    bounds = sorted(bounds_to_anchor)
    diagnostics["boundary_variants_assessed"] = len(bounds)
    if not bounds:
        return [], diagnostics

    rubric_texts = [text for pair in DEFAULT_CONTEXT_RUBRIC.values() for text in pair]
    assessment_texts: list[str] = []
    for left, right in bounds:
        assessment_texts.extend(
            [
                " ".join(unit.text for unit in units[left : right + 1]),
                units[left].text,
                units[right].text,
            ]
        )
    assessed = _embedding_batch(
        backend,
        [*assessment_texts, *rubric_texts],
        "spoken story admissibility",
    )
    rubric_vectors = assessed[len(assessment_texts) :]
    windows: list[SemanticWindow] = []
    rejected = 0
    for index, (left, right) in enumerate(bounds):
        margins = {
            name: cosine(assessed[index * 3 + dimension], rubric_vectors[dimension * 2])
            - cosine(assessed[index * 3 + dimension], rubric_vectors[dimension * 2 + 1])
            for dimension, name in enumerate(DEFAULT_CONTEXT_RUBRIC)
        }
        if any(value <= minimum_context_margin for value in margins.values()):
            rejected += 1
            continue
        anchor = bounds_to_anchor[(left, right)]
        pairwise = [
            cosine(vectors[item - 1], vectors[item]) for item in range(left + 1, right + 1)
        ]
        coherence = sum(pairwise) / len(pairwise) if pairwise else 1.0
        opening = _quality(margins["opening"])
        story = _quality(margins["story"])
        ending = _quality(margins["ending"])
        coherence_quality = max(0.0, min(1.0, (coherence + 1.0) / 2.0))
        score = max(
            0.0,
            min(
                1.0,
                0.35 * strengths[anchor]
                + 0.20 * opening
                + 0.20 * story
                + 0.20 * ending
                + 0.05 * coherence_quality,
            ),
        )
        windows.append(
            SemanticWindow(
                start=round(units[left].start, 3),
                end=round(units[right].end, 3),
                text=" ".join(unit.text for unit in units[left : right + 1]).strip(),
                anchor_time=round((units[anchor].start + units[anchor].end) / 2.0, 3),
                event_label=labels[anchor],
                event_similarity=round(strengths[anchor], 6),
                opening_quality=round(opening, 6),
                story_quality=round(story, 6),
                ending_quality=round(ending, 6),
                coherence=round(coherence_quality, 6),
                score=round(score, 6),
                vector=tuple(float(value) for value in assessed[index * 3]),
            )
        )
    diagnostics["context_rejected_variants"] = rejected
    diagnostics["candidate_count"] = len(windows)
    diagnostics["story_admissibility"] = {
        "opening_complete": True,
        "story_complete": True,
        "ending_complete": True,
        "closed_ending_required": True,
        "question_without_answer_rejected": True,
    }
    return sorted(windows, key=lambda item: (-item.score, item.start, item.end)), diagnostics


def topic_coverage(
    windows: Sequence[SemanticWindow],
    topics: Sequence[dict[str, Any]],
    backend: EmbeddingFn,
    config: dict[str, Any],
) -> tuple[list[dict[str, float]], dict[str, dict[str, Any]]]:
    semantic_cfg = dict(config.get("spoken_content", {}).get("semantic") or {})
    threshold = float(semantic_cfg.get("minimum_topic_similarity", 0.42))
    descriptors: list[str] = []
    descriptor_topics: list[str] = []
    for topic in topics:
        topic_id = str(topic["id"])
        values = [
            str(item).strip()
            for item in (
                topic.get("coverage_descriptions")
                or topic.get("queries")
                or []
            )
            if str(item).strip()
        ]
        headline = str(topic.get("headline") or "").strip()
        if headline:
            values.append(headline)
        if not values:
            raise RuntimeError(f"spoken topic {topic_id} has no semantic coverage descriptions")
        for value in values:
            descriptors.append(value)
            descriptor_topics.append(topic_id)

    if not windows:
        return [], {
            str(topic["id"]): {
                "best_similarity": None,
                "minimum_similarity": threshold,
                "candidate_count": 0,
            }
            for topic in topics
        }

    embedded = _embedding_batch(
        backend,
        [*[window.text for window in windows], *descriptors],
        "campaign topic coverage",
    )
    window_vectors = embedded[: len(windows)]
    descriptor_vectors = embedded[len(windows) :]
    topic_vectors: dict[str, list[list[float]]] = {}
    for topic_id, vector in zip(descriptor_topics, descriptor_vectors, strict=True):
        topic_vectors.setdefault(topic_id, []).append(vector)

    coverage: list[dict[str, float]] = []
    topic_summary: dict[str, dict[str, Any]] = {}
    for topic in topics:
        topic_id = str(topic["id"])
        scores = [
            max(cosine(vector, descriptor) for descriptor in topic_vectors[topic_id])
            for vector in window_vectors
        ]
        best = max(scores) if scores else None
        topic_summary[topic_id] = {
            "best_similarity": round(best, 6) if best is not None else None,
            "minimum_similarity": threshold,
            "candidate_count": sum(score >= threshold for score in scores),
        }

    for vector in window_vectors:
        item: dict[str, float] = {}
        for topic in topics:
            topic_id = str(topic["id"])
            similarity = max(
                cosine(vector, descriptor) for descriptor in topic_vectors[topic_id]
            )
            if similarity >= threshold:
                item[topic_id] = round(similarity, 6)
        coverage.append(item)
    return coverage, topic_summary
