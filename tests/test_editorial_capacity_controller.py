from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clipper.autonomous_quality_planner import AutonomousQualityPlanner
from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.editorial_capacity import (
    capacity_target_input_tokens,
    natural_boundary_near,
    natural_split_index,
    shrink_context_around_interval,
    stable_range_stage,
    token_aware_repartition,
)
from clipper.providers.base import (
    EditorialCapacityError,
    InferenceUsage,
    ModelIdentity,
    ProviderResult,
)
from clipper.source_hazards import SourceHazardClassifier


def _timeline(
    count: int,
    *,
    sentence_at: int | None = None,
    speaker_change_at: int | None = None,
    gap_at: int | None = None,
) -> CanonicalTimeline:
    words: list[CanonicalWord] = []
    cursor = 0.0
    for index in range(count):
        if gap_at is not None and index == gap_at:
            cursor += 3.0
        text = f"word-{index}"
        if sentence_at is not None and index == sentence_at:
            text += "."
        speaker = "a" if speaker_change_at is None or index < speaker_change_at else "b"
        words.append(
            CanonicalWord(
                f"video:w{index:07d}:digest",
                text,
                cursor,
                cursor + 0.5,
                speaker,
                0.99,
                "word_exact",
                "test",
            )
        )
        cursor += 0.5
    return CanonicalTimeline("video", "source-hash", tuple(words))


def _usage() -> InferenceUsage:
    return InferenceUsage("test", "now", 0.01)


def test_natural_capacity_split_prefers_source_boundaries_and_validates_ranges() -> None:
    assert natural_split_index(_timeline(1), 0, 1) is None
    assert natural_split_index(_timeline(8, sentence_at=3), 0, 8) == 4
    assert natural_split_index(_timeline(8, speaker_change_at=4), 0, 8) == 4
    assert natural_split_index(_timeline(8, gap_at=5), 0, 8) == 5
    assert natural_split_index(_timeline(8), 0, 8) == 4

    timeline = _timeline(8)
    assert stable_range_stage("semantic_cores", timeline, 1, 7).startswith("semantic_cores:")
    with pytest.raises(ValueError, match="range is invalid"):
        stable_range_stage("semantic_cores", timeline, 4, 4)


def test_token_aware_repartition_uses_observed_context_ratio_in_one_step() -> None:
    timeline = _timeline(100, sentence_at=23)
    details = {
        "reason": "context_exhausted",
        "input_tokens": 4_239_373,
        "context_limit_tokens": 262_144,
        "generation_budget_tokens": 1_456,
    }

    target = capacity_target_input_tokens(details)
    assert target == 260_688
    plan = token_aware_repartition(timeline, 0, 100, details)
    assert plan is not None
    assert plan.observed_input_tokens == 4_239_373
    assert plan.target_input_tokens == 260_688
    assert plan.partition_count == 17
    assert plan.ranges[0][0] == 0
    assert plan.ranges[-1][1] == 100
    assert all(left < right for left, right in plan.ranges)
    assert sum(right - left for left, right in plan.ranges) == 100


def test_token_aware_repartition_uses_dynamic_oom_as_runtime_boundary() -> None:
    timeline = _timeline(40, speaker_change_at=20)
    details = {
        "reason": "cuda_oom_dynamic_cache",
        "input_tokens": 153_725,
        "context_limit_tokens": 262_144,
        "generation_budget_tokens": 1_456,
    }

    assert capacity_target_input_tokens(details) == 76_862
    plan = token_aware_repartition(timeline, 0, 40, details)
    assert plan is not None
    assert plan.partition_count == 3
    assert plan.ranges[0][0] == 0
    assert plan.ranges[-1][1] == 40


def test_token_aware_repartition_prefers_learned_good_boundary() -> None:
    timeline = _timeline(30)
    details = {
        "reason": "history_dynamic_oom_boundary",
        "input_tokens": 160_000,
        "context_limit_tokens": 262_144,
        "generation_budget_tokens": 1_000,
        "largest_good_input_tokens": 61_591,
        "smallest_bad_input_tokens": 153_725,
    }
    assert capacity_target_input_tokens(details) == 61_591
    plan = token_aware_repartition(timeline, 0, 30, details)
    assert plan is not None
    assert plan.partition_count == 3


def test_natural_boundary_near_snaps_to_source_structure_not_midpoint() -> None:
    timeline = _timeline(20, sentence_at=3, speaker_change_at=15)
    assert natural_boundary_near(timeline, 0, 20, 5) == 4


def test_context_shrinking_preserves_required_interval_without_absolute_window() -> None:
    timeline = _timeline(12)
    assert shrink_context_around_interval(timeline, 2, 8, 2, 8) is None
    left_shrunk = shrink_context_around_interval(timeline, 0, 12, 6, 9)
    assert left_shrunk is not None
    assert left_shrunk[0] <= 6 < 9 <= left_shrunk[1]
    right_shrunk = shrink_context_around_interval(timeline, 0, 12, 2, 5)
    assert right_shrunk is not None
    assert right_shrunk[0] <= 2 < 5 <= right_shrunk[1]
    with pytest.raises(ValueError, match="outside context"):
        shrink_context_around_interval(timeline, 3, 8, 1, 5)


class CapacityHazardEditorial:
    identity = ModelIdentity("capacity", "rev", "none", "test", "editor", "schema")

    def __init__(self, maximum_words: int) -> None:
        self.maximum_words = maximum_words
        self.calls: list[int] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        assert task.startswith("source_hazards:")
        raw_words = payload.get("words")
        words = raw_words if isinstance(raw_words, list) else []
        self.calls.append(len(words))
        if len(words) > self.maximum_words:
            raise EditorialCapacityError(
                "synthetic hazard capacity",
                details={"word_count": len(words)},
            )
        if not words:
            return ProviderResult({"segments": []}, self.identity, _usage())
        return ProviderResult(
            {
                "segments": [
                    {
                        "start_word_id": words[0]["word_ref"],
                        "end_word_id": words[-1]["word_ref"],
                        "classification": "editorial_content",
                        "confidence": 0.9,
                        "evidence": ["source-grounded"],
                    }
                ]
            },
            self.identity,
            _usage(),
        )


def test_source_hazards_split_on_capacity_and_fail_closed_only_at_smallest_unit(
    tmp_path: Path,
) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline(12)
    provider = CapacityHazardEditorial(maximum_words=3)
    result = SourceHazardClassifier(provider, DagStore(tmp_path / "adaptive")).classify(
        brief,
        timeline,
        multimodal=None,
    )
    assert result.rejections == ()
    assert provider.calls[0] == len(timeline.words)
    assert any(count <= provider.maximum_words for count in provider.calls)
    assert sum(len(item.source_word_ids) for item in result.hazards) == len(timeline.words)

    one_word = _timeline(1)
    exhausted = CapacityHazardEditorial(maximum_words=0)
    failed = SourceHazardClassifier(exhausted, DagStore(tmp_path / "exhausted")).classify(
        brief,
        one_word,
        multimodal=None,
    )
    assert failed.hazards[0].evidence == ("source_hazard_capacity_exhausted",)
    assert failed.rejections[0]["reasons"] == [
        "policy_uncertain",
        "editorial_capacity_exhausted",
    ]


def test_source_hazard_legacy_cache_is_reused_without_provider_reexecution(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline(20)
    dag = DagStore(tmp_path / "legacy")
    writer = CapacityHazardEditorial(maximum_words=100)
    classifier = SourceHazardClassifier(writer, dag)

    ranges = classifier._legacy_ranges(timeline)
    assert ranges
    for index, (start, end) in enumerate(ranges):
        payload = classifier._payload_for_range(brief, timeline, None, start, end)
        classifier._complete(timeline, brief, f"source_hazards:{index}", payload)

    class NeverCalled(CapacityHazardEditorial):
        def complete_json(
            self,
            *,
            task: str,
            payload: dict[str, Any],
        ) -> ProviderResult[dict[str, Any]]:
            raise AssertionError((task, payload))

    reader = SourceHazardClassifier(NeverCalled(maximum_words=0), dag)
    replay = reader.classify(brief, timeline, multimodal=None)
    assert replay.stage_cache_hits == len(ranges)
    assert replay.stage_executions == 0
    assert replay.hazards


class EnvelopeCapacityEditorial:
    identity = ModelIdentity("capacity", "rev", "none", "test", "editor", "schema")

    def __init__(self) -> None:
        self.context_sizes: list[int] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        if task.startswith("semantic_cores:"):
            words = payload["words"]
            return ProviderResult(
                {
                    "cores": [
                        {
                            "core_id": "ignored",
                            "start_word_id": words[4]["word_ref"],
                            "end_word_id": words[6]["word_ref"],
                            "semantic_summary": "worthwhile idea",
                            "editorial_reason": "source-grounded reason",
                            "confidence": 0.9,
                        }
                    ]
                },
                self.identity,
                _usage(),
            )
        if task.startswith("narrative_envelope:"):
            words = payload["source_context_words"]
            self.context_sizes.append(len(words))
            if len(words) > 12:
                raise EditorialCapacityError("envelope context too large")
            core = payload["core"]
            return ProviderResult(
                {
                    "envelope_id": "ignored",
                    "core_id": core["core_id"],
                    "start_word_id": words[0]["word_ref"],
                    "end_word_id": words[-1]["word_ref"],
                    "required_prior_context": "setup",
                    "required_followup_context": "payoff",
                    "setup_resolved": True,
                    "payoff_resolved": True,
                    "reference_resolution": [],
                    "confidence": 0.9,
                },
                self.identity,
                _usage(),
            )
        if task.startswith("quality_windows:"):
            return ProviderResult(
                {
                    "core_id": payload["core"]["core_id"],
                    "selected_window_id": None,
                    "decision": "REJECT",
                    "quality_score": 0.5,
                    "opening_strategy": "insufficient opening",
                    "rationale": "not strong enough",
                    "confidence": 0.9,
                },
                self.identity,
                _usage(),
            )
        raise AssertionError(task)


def test_narrative_context_shrinks_on_capacity_while_preserving_core(tmp_path: Path) -> None:
    timeline = _timeline(30)
    provider = EnvelopeCapacityEditorial()
    result = AutonomousQualityPlanner(provider, DagStore(tmp_path / "planner")).plan(
        timeline,
        multimodal=None,
        modality_profile=None,
        campaign_context={},
        relevant_policy={},
        min_seconds=2.0,
        max_seconds=12.0,
    )
    assert result.cores
    assert provider.context_sizes[0] == len(timeline.words)
    assert provider.context_sizes[-1] <= 12
    assert all(
        set(core.source_word_ids).issubset(envelope.source_word_ids)
        for core, envelope in zip(result.cores, result.envelopes, strict=True)
    )
