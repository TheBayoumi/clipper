from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


REQUIRED_ACCEPTANCE_DOMAINS = frozenset(
    {"gaming", "business", "comedy_conversational", "science_education", "interview_personal"}
)


@dataclass(frozen=True, slots=True)
class BenchmarkThresholds:
    story_moment_recall: float = 0.85
    clip_concept_precision: float = 0.80
    semantic_duplicate_rate: float = 0.10
    boundary_pass_rate: float = 0.90

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("benchmark thresholds must be between 0 and 1")

    @classmethod
    def from_dict(cls, payload: object) -> BenchmarkThresholds:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("benchmark thresholds must be an object")
        allowed = set(asdict(cls()))
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unsupported benchmark threshold: {sorted(unknown)[0]}")
        return cls(**{key: float(payload[key]) for key in payload})


@dataclass(frozen=True, slots=True)
class CorpusBenchmarkResult:
    status: str
    thresholds: BenchmarkThresholds
    aggregate: BenchmarkMetrics
    domains: tuple[str, ...]
    episode_metrics: tuple[tuple[str, str, BenchmarkMetrics], ...]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "clipper-benchmark-result-v1",
            "status": self.status,
            "thresholds": asdict(self.thresholds),
            "aggregate": self.aggregate.to_dict(),
            "domains": list(self.domains),
            "episodes": [
                {"episode_id": episode_id, "domain": domain, "metrics": metrics.to_dict()}
                for episode_id, domain, metrics in self.episode_metrics
            ],
            "failures": list(self.failures),
        }


def _reference_from_dict(payload: object) -> ReferenceMoment:
    if not isinstance(payload, dict):
        raise ValueError("benchmark reference must be an object")
    return ReferenceMoment(
        reference_id=str(payload.get("reference_id") or ""),
        start=float(payload.get("start") or 0.0),
        end=float(payload.get("end") or 0.0),
        semantic_group=str(payload.get("semantic_group") or ""),
        acceptable_start_min=(
            float(payload["acceptable_start_min"])
            if payload.get("acceptable_start_min") is not None
            else None
        ),
        acceptable_start_max=(
            float(payload["acceptable_start_max"])
            if payload.get("acceptable_start_max") is not None
            else None
        ),
        acceptable_end_min=(
            float(payload["acceptable_end_min"])
            if payload.get("acceptable_end_min") is not None
            else None
        ),
        acceptable_end_max=(
            float(payload["acceptable_end_max"])
            if payload.get("acceptable_end_max") is not None
            else None
        ),
    )


def _artifact_ranges(path: Path, *, start_key: str, end_key: str) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"benchmark prediction artifact must be a list: {path}")
    rows: list[tuple[float, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"benchmark prediction row must be an object: {path}")
        start = float(item.get(start_key) or 0.0)
        end = float(item.get(end_key) or 0.0)
        if start < 0 or end <= start:
            raise ValueError(f"benchmark prediction range is invalid: {path}")
        rows.append((start, end))
    return rows


def evaluate_ranges(
    references: Sequence[ReferenceMoment],
    story_ranges: Sequence[tuple[float, float]],
    concept_ranges: Sequence[tuple[float, float]],
    *,
    minimum_overlap: float = 0.5,
) -> BenchmarkMetrics:
    if not references:
        raise ValueError("benchmark references must not be empty")
    if not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be in (0, 1]")
    matched_reference_ids = {
        reference.reference_id
        for start, end in story_ranges
        if (reference := _best_reference(start, end, references, minimum_overlap=minimum_overlap))
        is not None
    }
    matched_concepts = [
        (start, end, reference)
        for start, end in concept_ranges
        if (reference := _best_reference(start, end, references, minimum_overlap=minimum_overlap))
        is not None
    ]
    group_counts: dict[str, int] = {}
    for _start, _end, reference in matched_concepts:
        group_counts[reference.semantic_group] = group_counts.get(reference.semantic_group, 0) + 1
    duplicate_concepts = sum(max(0, count - 1) for count in group_counts.values())
    boundary_passes = sum(
        1 for start, end, reference in matched_concepts if _boundary_pass(start, end, reference)
    )
    return BenchmarkMetrics(
        story_moment_recall=len(matched_reference_ids) / len(references),
        clip_concept_precision=(len(matched_concepts) / len(concept_ranges))
        if concept_ranges
        else 0.0,
        semantic_duplicate_rate=(duplicate_concepts / len(concept_ranges))
        if concept_ranges
        else 0.0,
        boundary_pass_rate=(boundary_passes / len(matched_concepts)) if matched_concepts else 0.0,
        matched_reference_moments=len(matched_reference_ids),
        predicted_story_moments=len(story_ranges),
        predicted_concepts=len(concept_ranges),
        duplicate_concepts=duplicate_concepts,
    )


def _threshold_failures(metrics: BenchmarkMetrics, thresholds: BenchmarkThresholds) -> list[str]:
    failures: list[str] = []
    if metrics.story_moment_recall < thresholds.story_moment_recall:
        failures.append("story_moment_recall_below_target")
    if metrics.clip_concept_precision < thresholds.clip_concept_precision:
        failures.append("clip_concept_precision_below_target")
    if metrics.semantic_duplicate_rate > thresholds.semantic_duplicate_rate:
        failures.append("semantic_duplicate_rate_above_target")
    if metrics.boundary_pass_rate < thresholds.boundary_pass_rate:
        failures.append("boundary_pass_rate_below_target")
    return failures


def evaluate_corpus_manifest(manifest_path: str | Path) -> CorpusBenchmarkResult:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "clipper-benchmark-corpus-v1"
    ):
        raise ValueError("unsupported benchmark corpus manifest")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("benchmark corpus must contain episodes")
    thresholds = BenchmarkThresholds.from_dict(payload.get("thresholds"))
    minimum_overlap = float(payload.get("minimum_overlap") or 0.5)
    metrics: list[BenchmarkMetrics] = []
    rows: list[tuple[str, str, BenchmarkMetrics]] = []
    domains: set[str] = set()
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise ValueError("benchmark episode must be an object")
        episode_id = str(raw.get("episode_id") or "").strip()
        domain = str(raw.get("domain") or "").strip()
        if not episode_id or not domain:
            raise ValueError("benchmark episode requires episode_id and domain")
        raw_references = raw.get("references")
        if not isinstance(raw_references, list) or not raw_references:
            raise ValueError(f"benchmark episode has no references: {episode_id}")
        predictions = raw.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError(f"benchmark episode has no prediction artifacts: {episode_id}")
        story_file = path.parent / str(predictions.get("story_moments") or "")
        concept_file = path.parent / str(predictions.get("concepts") or "")
        if not story_file.is_file() or not concept_file.is_file():
            raise ValueError(f"benchmark prediction artifacts are missing: {episode_id}")
        references = [_reference_from_dict(item) for item in raw_references]
        result = evaluate_ranges(
            references,
            _artifact_ranges(story_file, start_key="start", end_key="end"),
            _artifact_ranges(concept_file, start_key="source_start", end_key="source_end"),
            minimum_overlap=minimum_overlap,
        )
        metrics.append(result)
        rows.append((episode_id, domain, result))
        domains.add(domain)
    missing_domains = sorted(REQUIRED_ACCEPTANCE_DOMAINS - domains)
    aggregate = aggregate_metrics(metrics)
    failures = [f"missing_domain:{domain}" for domain in missing_domains]
    failures.extend(_threshold_failures(aggregate, thresholds))
    return CorpusBenchmarkResult(
        status="PASS" if not failures else "FAIL",
        thresholds=thresholds,
        aggregate=aggregate,
        domains=tuple(sorted(domains)),
        episode_metrics=tuple(rows),
        failures=tuple(failures),
    )
