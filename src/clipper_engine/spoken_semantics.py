from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class TopicMatch:
    topic_id: str
    window_index: int
    similarity: float
    runner_up_similarity: float
    margin: float
    score: float


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


def _fit_minimum_duration(
    left: int,
    right: int,
    units: Sequence[SemanticUnit],
    minimum: float,
    maximum: float,
    transcript_duration: float,
) -> tuple[float, float] | None:
    start = units[left].start
    end = units[right].end
    duration = end - start
    if duration > maximum + 1e-3:
        return None
    if duration >= minimum - 1e-3:
        return start, end

    previous_end = units[left - 1].end if left > 0 else 0.0
    next_start = units[right + 1].start if right + 1 < len(units) else transcript_duration
    left_slack = max(0.0, start - previous_end)
    right_slack = max(0.0, next_start - end)
    deficit = minimum - duration

    left_padding = min(left_slack, deficit / 2.0)
    right_padding = min(right_slack, deficit - left_padding)
    remaining = deficit - left_padding - right_padding
    if remaining > 1e-3:
        extra_left = min(left_slack - left_padding, remaining)
        left_padding += extra_left
        remaining -= extra_left
    if remaining > 1e-3:
        extra_right = min(right_slack - right_padding, remaining)
        right_padding += extra_right
        remaining -= extra_right
    if remaining > 1e-3:
        return None

    fitted_start = max(0.0, start - left_padding)
    fitted_end = min(transcript_duration, end + right_padding)
    if fitted_end - fitted_start > maximum + 1e-3:
        return None
    return fitted_start, fitted_end


def _window_bounds(
    anchor: int,
    units: Sequence[SemanticUnit],
    minimum: float,
    maximum: float,
    transcript_duration: float,
    maximum_internal_pause: float,
) -> list[tuple[int, int, float, float]]:
    left_floor = anchor
    while left_floor > 0:
        gap = units[left_floor].start - units[left_floor - 1].end
        if gap > maximum_internal_pause:
            break
        if units[anchor].end - units[left_floor - 1].start > maximum:
            break
        left_floor -= 1

    right_ceiling = anchor
    while right_ceiling + 1 < len(units):
        gap = units[right_ceiling + 1].start - units[right_ceiling].end
        if gap > maximum_internal_pause:
            break
        if units[right_ceiling + 1].end - units[anchor].start > maximum:
            break
        right_ceiling += 1

    result: list[tuple[int, int, float, float]] = []
    for left in range(left_floor, anchor + 1):
        for right in range(anchor, right_ceiling + 1):
            if not _closed_ending(units[right].text):
                continue
            question_indexes = [
                index for index in range(left, right + 1) if "?" in units[index].text
            ]
            if question_indexes and question_indexes[-1] == right:
                continue
            fitted = _fit_minimum_duration(
                left,
                right,
                units,
                minimum,
                maximum,
                transcript_duration,
            )
            if fitted is None:
                continue
            result.append((left, right, fitted[0], fitted[1]))
    return result


def _quality(margin: float) -> float:
    return max(0.0, min(1.0, 0.5 + 1.5 * margin))


def _overlap(left: SemanticWindow, right: SemanticWindow) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0 else 0.0


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
    maximum_internal_pause = float(semantic_cfg.get("maximum_internal_pause_seconds", 4.5))

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
        "semantic_regions_are_hard_window_boundaries": False,
        "maximum_internal_pause_seconds": maximum_internal_pause,
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
        for item in semantic_cfg.get(
            "off_topic_descriptions",
            DEFAULT_OFF_TOPIC_DESCRIPTIONS,
        )
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

    bounds_to_anchor: dict[tuple[int, int, float, float], int] = {}
    for anchor in anchors:
        for bounds in _window_bounds(
            anchor,
            units,
            minimum,
            maximum,
            transcript.duration,
            maximum_internal_pause,
        ):
            previous = bounds_to_anchor.get(bounds)
            if previous is None or strengths[anchor] > strengths[previous]:
                bounds_to_anchor[bounds] = anchor

    bounds = sorted(bounds_to_anchor)
    diagnostics["boundary_variants_assessed"] = len(bounds)
    diagnostics["cross_region_variant_count"] = sum(
        regions[left] != regions[right] for left, right, _, _ in bounds
    )
    if not bounds:
        return [], diagnostics

    rubric_texts = [text for pair in DEFAULT_CONTEXT_RUBRIC.values() for text in pair]
    assessment_texts: list[str] = []
    for left, right, _, _ in bounds:
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
    for index, (left, right, fitted_start, fitted_end) in enumerate(bounds):
        margins = {
            name: cosine(
                assessed[index * 3 + dimension],
                rubric_vectors[dimension * 2],
            )
            - cosine(
                assessed[index * 3 + dimension],
                rubric_vectors[dimension * 2 + 1],
            )
            for dimension, name in enumerate(DEFAULT_CONTEXT_RUBRIC)
        }
        if any(value <= minimum_context_margin for value in margins.values()):
            rejected += 1
            continue

        anchor = bounds_to_anchor[(left, right, fitted_start, fitted_end)]
        pairwise = [cosine(vectors[item - 1], vectors[item]) for item in range(left + 1, right + 1)]
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
                start=round(fitted_start, 3),
                end=round(fitted_end, 3),
                text=" ".join(unit.text for unit in units[left : right + 1]).strip(),
                anchor_time=round(
                    (units[anchor].start + units[anchor].end) / 2.0,
                    3,
                ),
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

    unique_windows: list[SemanticWindow] = []
    for window in sorted(windows, key=lambda item: (-item.score, item.start, item.end)):
        if any(
            abs(window.start - other.start) < 1e-3
            and abs(window.end - other.end) < 1e-3
            and window.text == other.text
            for other in unique_windows
        ):
            continue
        unique_windows.append(window)

    diagnostics["context_rejected_variants"] = rejected
    diagnostics["candidate_count"] = len(unique_windows)
    diagnostics["story_admissibility"] = {
        "opening_complete": True,
        "story_complete": True,
        "ending_complete": True,
        "closed_ending_required": True,
        "question_without_answer_rejected": True,
        "silence_padding_allowed_to_meet_minimum_duration": True,
    }
    return unique_windows, diagnostics


def topic_matches(
    windows: Sequence[SemanticWindow],
    topics: Sequence[dict[str, Any]],
    backend: EmbeddingFn,
    config: dict[str, Any],
) -> tuple[dict[str, list[TopicMatch]], dict[str, dict[str, Any]]]:
    semantic_cfg = dict(config.get("spoken_content", {}).get("semantic") or {})
    threshold = float(semantic_cfg.get("minimum_topic_similarity", 0.52))
    minimum_topic_margin = float(semantic_cfg.get("minimum_topic_margin", 0.05))
    variants_per_topic = int(semantic_cfg.get("variants_per_topic", 5))
    maximum_variant_overlap = float(semantic_cfg.get("maximum_variant_overlap", 0.85))

    descriptors: list[str] = []
    descriptor_topics: list[str] = []
    for topic in topics:
        topic_id = str(topic["id"])
        values = [
            str(item).strip()
            for item in (topic.get("coverage_descriptions") or topic.get("queries") or [])
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

    empty_summary = {
        str(topic["id"]): {
            "best_similarity": None,
            "best_runner_up_similarity": None,
            "best_margin": None,
            "minimum_similarity": threshold,
            "minimum_topic_margin": minimum_topic_margin,
            "candidate_count": 0,
        }
        for topic in topics
    }
    if not windows:
        return {str(topic["id"]): [] for topic in topics}, empty_summary

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

    similarities: dict[str, list[float]] = {}
    for topic in topics:
        topic_id = str(topic["id"])
        similarities[topic_id] = [
            max(cosine(vector, descriptor) for descriptor in topic_vectors[topic_id])
            for vector in window_vectors
        ]

    matches: dict[str, list[TopicMatch]] = {}
    summary: dict[str, dict[str, Any]] = {}
    topic_ids = [str(topic["id"]) for topic in topics]
    for topic_id in topic_ids:
        ranked: list[TopicMatch] = []
        for window_index, similarity in enumerate(similarities[topic_id]):
            runner_up = max(
                (similarities[other][window_index] for other in topic_ids if other != topic_id),
                default=0.0,
            )
            margin = similarity - runner_up
            if similarity < threshold or margin < minimum_topic_margin:
                continue
            discriminative = max(
                0.0,
                min(1.0, margin / max(0.001, minimum_topic_margin + 0.20)),
            )
            score = max(
                0.0,
                min(
                    1.0,
                    0.70 * similarity + 0.20 * discriminative + 0.10 * windows[window_index].score,
                ),
            )
            ranked.append(
                TopicMatch(
                    topic_id=topic_id,
                    window_index=window_index,
                    similarity=round(similarity, 6),
                    runner_up_similarity=round(runner_up, 6),
                    margin=round(margin, 6),
                    score=round(score, 6),
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.similarity,
                -item.margin,
                windows[item.window_index].start,
            )
        )
        selected: list[TopicMatch] = []
        for match in ranked:
            window = windows[match.window_index]
            if any(
                _overlap(window, windows[item.window_index]) > maximum_variant_overlap
                for item in selected
            ):
                continue
            selected.append(match)
            if len(selected) >= variants_per_topic:
                break

        all_scores = similarities[topic_id]
        raw_best_index = max(range(len(all_scores)), key=all_scores.__getitem__)
        raw_best_similarity = all_scores[raw_best_index]
        raw_best_runner_up = max(
            (
                similarities[other][raw_best_index]
                for other in topic_ids
                if other != topic_id
            ),
            default=0.0,
        )
        best_qualified = selected[0] if selected else None
        matches[topic_id] = selected
        summary[topic_id] = {
            "best_similarity": (
                best_qualified.similarity if best_qualified is not None else None
            ),
            "best_runner_up_similarity": (
                best_qualified.runner_up_similarity
                if best_qualified is not None
                else None
            ),
            "best_margin": best_qualified.margin if best_qualified is not None else None,
            "best_raw_similarity": round(raw_best_similarity, 6),
            "best_raw_runner_up_similarity": round(raw_best_runner_up, 6),
            "best_raw_margin": round(raw_best_similarity - raw_best_runner_up, 6),
            "minimum_similarity": threshold,
            "minimum_topic_margin": minimum_topic_margin,
            "candidate_count": len(selected),
            "independent_search": True,
        }

    return matches, summary
