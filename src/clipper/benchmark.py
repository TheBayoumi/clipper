from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import ClipConcept, StoryMoment


@dataclass(frozen=True, slots=True)
class ReferenceMoment:
    reference_id: str
    start: float
    end: float
    semantic_group: str
    acceptable_start_min: float | None = None
    acceptable_start_max: float | None = None
    acceptable_end_min: float | None = None
    acceptable_end_max: float | None = None

    def __post_init__(self) -> None:
        if not self.reference_id or not self.semantic_group or self.end <= self.start:
            raise ValueError("invalid benchmark reference moment")


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    story_moment_recall: float
    clip_concept_precision: float
    semantic_duplicate_rate: float
    boundary_pass_rate: float
    matched_reference_moments: int
    predicted_story_moments: int
    predicted_concepts: int
    duplicate_concepts: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "story_moment_recall": self.story_moment_recall,
            "clip_concept_precision": self.clip_concept_precision,
            "semantic_duplicate_rate": self.semantic_duplicate_rate,
            "boundary_pass_rate": self.boundary_pass_rate,
            "matched_reference_moments": self.matched_reference_moments,
            "predicted_story_moments": self.predicted_story_moments,
            "predicted_concepts": self.predicted_concepts,
            "duplicate_concepts": self.duplicate_concepts,
        }


def _overlap_ratio(start: float, end: float, reference: ReferenceMoment) -> float:
    overlap = max(0.0, min(end, reference.end) - max(start, reference.start))
    duration = max(1e-9, reference.end - reference.start)
    return overlap / duration


def _moment_range(moment: StoryMoment) -> tuple[float, float]:
    return moment.start, moment.end


def _concept_range(concept: ClipConcept) -> tuple[float, float]:
    return concept.source_start, concept.source_end


def _best_reference(
    start: float,
    end: float,
    references: Sequence[ReferenceMoment],
    *,
    minimum_overlap: float,
) -> ReferenceMoment | None:
    candidates = [(reference, _overlap_ratio(start, end, reference)) for reference in references]
    eligible = [item for item in candidates if item[1] >= minimum_overlap]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[1])[0]


def _boundary_pass(start: float, end: float, reference: ReferenceMoment) -> bool:
    start_min = reference.acceptable_start_min
    start_max = reference.acceptable_start_max
    end_min = reference.acceptable_end_min
    end_max = reference.acceptable_end_max
    if start_min is not None and start < start_min:
        return False
    if start_max is not None and start > start_max:
        return False
    if end_min is not None and end < end_min:
        return False
    return not (end_max is not None and end > end_max)


def evaluate_episode(
    references: Sequence[ReferenceMoment],
    story_moments: Sequence[StoryMoment],
    concepts: Sequence[ClipConcept],
    *,
    minimum_overlap: float = 0.5,
) -> BenchmarkMetrics:
    if not references:
        raise ValueError("benchmark references must not be empty")
    if not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be in (0, 1]")

    matched_reference_ids: set[str] = set()
    for moment in story_moments:
        start, end = _moment_range(moment)
        reference = _best_reference(start, end, references, minimum_overlap=minimum_overlap)
        if reference is not None:
            matched_reference_ids.add(reference.reference_id)

    matched_concepts: list[tuple[ClipConcept, ReferenceMoment]] = []
    for concept in concepts:
        start, end = _concept_range(concept)
        reference = _best_reference(start, end, references, minimum_overlap=minimum_overlap)
        if reference is not None:
            matched_concepts.append((concept, reference))

    group_counts: dict[str, int] = {}
    for _concept, reference in matched_concepts:
        group_counts[reference.semantic_group] = group_counts.get(reference.semantic_group, 0) + 1
    duplicate_concepts = sum(max(0, count - 1) for count in group_counts.values())

    boundary_passes = 0
    for concept, reference in matched_concepts:
        start, end = _concept_range(concept)
        if _boundary_pass(start, end, reference):
            boundary_passes += 1

    return BenchmarkMetrics(
        story_moment_recall=len(matched_reference_ids) / len(references),
        clip_concept_precision=(len(matched_concepts) / len(concepts)) if concepts else 0.0,
        semantic_duplicate_rate=(duplicate_concepts / len(concepts)) if concepts else 0.0,
        boundary_pass_rate=(boundary_passes / len(matched_concepts)) if matched_concepts else 0.0,
        matched_reference_moments=len(matched_reference_ids),
        predicted_story_moments=len(story_moments),
        predicted_concepts=len(concepts),
        duplicate_concepts=duplicate_concepts,
    )


def aggregate_metrics(metrics: Iterable[BenchmarkMetrics]) -> BenchmarkMetrics:
    rows = list(metrics)
    if not rows:
        raise ValueError("benchmark metrics must not be empty")
    total_predictions = sum(row.predicted_concepts for row in rows)
    total_moments = sum(row.predicted_story_moments for row in rows)
    matched = sum(row.matched_reference_moments for row in rows)
    duplicates = sum(row.duplicate_concepts for row in rows)
    # Macro-average the human-facing quality rates so one long episode cannot dominate a corpus.
    return BenchmarkMetrics(
        story_moment_recall=sum(row.story_moment_recall for row in rows) / len(rows),
        clip_concept_precision=sum(row.clip_concept_precision for row in rows) / len(rows),
        semantic_duplicate_rate=sum(row.semantic_duplicate_rate for row in rows) / len(rows),
        boundary_pass_rate=sum(row.boundary_pass_rate for row in rows) / len(rows),
        matched_reference_moments=matched,
        predicted_story_moments=total_moments,
        predicted_concepts=total_predictions,
        duplicate_concepts=duplicates,
    )
