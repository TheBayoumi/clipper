from __future__ import annotations

import pytest

from clipper.benchmark import BenchmarkMetrics, ReferenceMoment, aggregate_metrics, evaluate_episode
from clipper.models import ClipConcept, EditorialScores, StoryMoment


def _scores() -> EditorialScores:
    return EditorialScores(*([0.8] * 12))


def _moment(moment_id: str, start: float, end: float) -> StoryMoment:
    return StoryMoment(
        moment_id=moment_id,
        video_id="video",
        start=start,
        end=end,
        text=moment_id,
        moment_type="story",
        topic="topic",
        setup="setup",
        payoff="payoff",
        scores=_scores(),
        score=8.0,
        transcript_fingerprint=f"fp-{moment_id}",
    )


def _concept(concept_id: str, start: float, end: float) -> ClipConcept:
    return ClipConcept(
        concept_id=concept_id,
        video_id="video",
        source_start=start,
        source_end=end,
        text=concept_id,
        topic="topic",
        setup="setup",
        payoff="payoff",
        moment_type="story",
        recommended_duration=end - start,
        scores=_scores(),
        score=8.0,
        semantic_cluster=concept_id,
        transcript_fingerprint=f"fp-{concept_id}",
    )


def test_episode_benchmark_metrics_cover_recall_precision_duplicates_and_boundaries() -> None:
    references = [
        ReferenceMoment("r1", 10, 20, "story-a", 10, 12, 19, 22),
        ReferenceMoment("r2", 40, 50, "story-b", 39, 42, 49, 52),
    ]
    metrics = evaluate_episode(
        references,
        [_moment("m1", 10, 20), _moment("m2", 40, 50)],
        [_concept("c1", 11, 20), _concept("c2", 11, 20), _concept("c3", 41, 51)],
    )
    assert metrics.story_moment_recall == 1.0
    assert metrics.clip_concept_precision == 1.0
    assert metrics.semantic_duplicate_rate == pytest.approx(1 / 3)
    assert metrics.boundary_pass_rate == 1.0
    assert metrics.duplicate_concepts == 1


def test_episode_benchmark_rejects_invalid_inputs_and_records_misses() -> None:
    with pytest.raises(ValueError, match="invalid benchmark"):
        ReferenceMoment("", 2, 1, "")
    reference = ReferenceMoment("r", 10, 20, "story")
    with pytest.raises(ValueError, match="references"):
        evaluate_episode([], [], [])
    with pytest.raises(ValueError, match="minimum_overlap"):
        evaluate_episode([reference], [], [], minimum_overlap=0)
    metrics = evaluate_episode([reference], [_moment("miss", 100, 110)], [])
    assert metrics.story_moment_recall == 0.0
    assert metrics.clip_concept_precision == 0.0
    assert metrics.boundary_pass_rate == 0.0


def test_aggregate_metrics_macro_averages_episode_quality() -> None:
    first = BenchmarkMetrics(1.0, 0.8, 0.1, 0.9, 4, 5, 5, 1)
    second = BenchmarkMetrics(0.8, 1.0, 0.0, 1.0, 8, 10, 8, 0)
    aggregate = aggregate_metrics([first, second])
    assert aggregate.story_moment_recall == pytest.approx(0.9)
    assert aggregate.clip_concept_precision == pytest.approx(0.9)
    assert aggregate.semantic_duplicate_rate == pytest.approx(0.05)
    assert aggregate.boundary_pass_rate == pytest.approx(0.95)
    assert aggregate.predicted_concepts == 13
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_metrics([])
