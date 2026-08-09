from __future__ import annotations

import json

import pytest

from clipper.benchmark import (
    BenchmarkMetrics,
    BenchmarkThresholds,
    ReferenceMoment,
    aggregate_metrics,
    evaluate_corpus_manifest,
    evaluate_episode,
    evaluate_ranges,
)
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


def _write_prediction_pair(root, name: str, *, miss: bool = False) -> tuple[str, str]:
    story = root / f"{name}-stories.json"
    concept = root / f"{name}-concepts.json"
    start = 100.0 if miss else 10.0
    story.write_text(json.dumps([{"start": start, "end": start + 10}]), encoding="utf-8")
    concept.write_text(
        json.dumps([{"source_start": start, "source_end": start + 10}]), encoding="utf-8"
    )
    return story.name, concept.name


def _corpus_payload(tmp_path, domains, *, miss_domain: str | None = None):
    episodes = []
    for index, domain in enumerate(domains):
        story, concept = _write_prediction_pair(
            tmp_path, f"episode-{index}", miss=domain == miss_domain
        )
        episodes.append(
            {
                "episode_id": f"episode-{index}",
                "domain": domain,
                "references": [
                    {
                        "reference_id": f"r-{index}",
                        "start": 10,
                        "end": 20,
                        "semantic_group": f"g-{index}",
                        "acceptable_start_min": 9,
                        "acceptable_start_max": 11,
                        "acceptable_end_min": 19,
                        "acceptable_end_max": 21,
                    }
                ],
                "predictions": {"story_moments": story, "concepts": concept},
            }
        )
    return {
        "schema_version": "clipper-benchmark-corpus-v1",
        "thresholds": {
            "story_moment_recall": 0.85,
            "clip_concept_precision": 0.8,
            "semantic_duplicate_rate": 0.1,
            "boundary_pass_rate": 0.9,
        },
        "episodes": episodes,
    }


def test_evaluate_ranges_and_private_corpus_manifest(tmp_path) -> None:
    reference = ReferenceMoment("r", 10, 20, "g", 9, 11, 19, 21)
    ranges = evaluate_ranges([reference], [(10, 20)], [(10, 20)])
    assert ranges.story_moment_recall == 1.0
    assert ranges.boundary_pass_rate == 1.0

    domains = [
        "gaming",
        "business",
        "comedy_conversational",
        "science_education",
        "interview_personal",
    ]
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps(_corpus_payload(tmp_path, domains)), encoding="utf-8")
    result = evaluate_corpus_manifest(manifest)
    assert result.status == "PASS"
    assert result.aggregate.story_moment_recall == 1.0
    assert result.failures == ()
    assert result.to_dict()["schema_version"] == "clipper-benchmark-result-v1"

    failing = tmp_path / "failing.json"
    failing.write_text(
        json.dumps(_corpus_payload(tmp_path, domains[:-1], miss_domain="business")),
        encoding="utf-8",
    )
    failed = evaluate_corpus_manifest(failing)
    assert failed.status == "FAIL"
    assert "missing_domain:interview_personal" in failed.failures
    assert "story_moment_recall_below_target" in failed.failures


def test_benchmark_manifest_rejects_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        BenchmarkThresholds(story_moment_recall=2.0)
    with pytest.raises(ValueError, match="thresholds must be an object"):
        BenchmarkThresholds.from_dict([])
    with pytest.raises(ValueError, match="unsupported benchmark threshold"):
        BenchmarkThresholds.from_dict({"unknown": 1})
    assert BenchmarkThresholds.from_dict(None) == BenchmarkThresholds()

    bad = tmp_path / "bad.json"
    for payload, message in [
        ({}, "unsupported"),
        ({"schema_version": "clipper-benchmark-corpus-v1", "episodes": []}, "episodes"),
        (
            {
                "schema_version": "clipper-benchmark-corpus-v1",
                "episodes": [{"episode_id": "e", "domain": "gaming", "references": []}],
            },
            "no references",
        ),
    ]:
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            evaluate_corpus_manifest(bad)

    ref = ReferenceMoment("r", 1, 2, "g")
    with pytest.raises(ValueError, match="references"):
        evaluate_ranges([], [], [])
    with pytest.raises(ValueError, match="minimum_overlap"):
        evaluate_ranges([ref], [], [], minimum_overlap=0)
    artifact = tmp_path / "rows.json"
    artifact.write_text(json.dumps({"not": "list"}), encoding="utf-8")
    payload = _corpus_payload(
        tmp_path,
        ["gaming", "business", "comedy_conversational", "science_education", "interview_personal"],
    )
    payload["episodes"][0]["predictions"]["story_moments"] = artifact.name
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        evaluate_corpus_manifest(bad)
