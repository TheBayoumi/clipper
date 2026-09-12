from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clipper.autonomous_quality_planner import (
    AutonomousPlanningError,
    AutonomousQualityPlanner,
    narrative_envelope_from_payload,
    quality_assessment_from_payload,
    semantic_cores_from_payload,
)
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.modality_profile import SourceModalityProfile
from clipper.models import CampaignBrief
from clipper.providers.base import (
    EditorialCapacityError,
    InferenceUsage,
    ModelIdentity,
    ProviderResult,
)
from clipper.source_hazards import SourceHazardClassifier
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.window_solver import enumerate_feasible_windows


def _timeline(word_count: int = 60) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source-hash",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:digest",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker-a",
                0.99,
                "word_exact",
                "test",
            )
            for index in range(word_count)
        ),
    )


def _usage() -> InferenceUsage:
    return InferenceUsage(
        provider="fake",
        started_at="2026-08-21T00:00:00+00:00",
        duration_seconds=0.01,
        input_units=1,
        output_units=1,
        estimated_cost_usd=0.001,
    )


class FakeEditorial:
    identity = ModelIdentity(
        model_id="fake-editor",
        revision="rev-1",
        quantization="none",
        inference_engine="fake",
        prompt_version="editor",
        schema_version="editorial-json",
    )

    def __init__(self, *, zero_cores: bool = False, overlong_envelope: bool = False) -> None:
        self.zero_cores = zero_cores
        self.overlong_envelope = overlong_envelope
        self.calls: list[str] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        self.calls.append(task)
        if task.startswith("semantic_cores:"):
            value: dict[str, Any] = {
                "cores": []
                if self.zero_cores
                else [
                    {
                        "core_id": "model-a",
                        "start_word_id": "w0000010",
                        "end_word_id": "w0000012",
                        "semantic_summary": "first worthwhile event",
                        "editorial_reason": "complete source-grounded idea",
                        "confidence": 0.95,
                    },
                    {
                        "core_id": "model-b",
                        "start_word_id": "w0000035",
                        "end_word_id": "w0000037",
                        "semantic_summary": "second worthwhile event",
                        "editorial_reason": "independent source-grounded idea",
                        "confidence": 0.91,
                    },
                ]
            }
        elif task.startswith("narrative_envelope:"):
            core = payload["core"]
            core_start = float(core["source_start"])
            if self.overlong_envelope:
                start_ref, end_ref = "w0000002", "w0000045"
            elif core_start < 20:
                start_ref, end_ref = "w0000008", "w0000017"
            else:
                start_ref, end_ref = "w0000033", "w0000042"
            value = {
                "envelope_id": "model-envelope",
                "core_id": core["core_id"],
                "start_word_id": start_ref,
                "end_word_id": end_ref,
                "required_prior_context": "setup",
                "required_followup_context": "payoff",
                "setup_resolved": True,
                "payoff_resolved": True,
                "reference_resolution": ["reference resolved"],
                "confidence": 0.94,
            }
        elif task.startswith("quality_windows:"):
            core = payload["core"]
            windows = payload["feasible_windows"]
            value = {
                "core_id": core["core_id"],
                "selected_window_id": windows[0]["window_id"],
                "decision": "PASS",
                "quality_score": 0.9,
                "rationale": "worth publishing",
                "opening_strategy": "open on the first complete source-grounded statement",
                "confidence": 0.93,
            }
        else:  # pragma: no cover - defensive fake-provider guard
            raise AssertionError(task)
        return ProviderResult(value=value, model=self.identity, usage=_usage())


def _core(timeline: CanonicalTimeline) -> SemanticCore:
    return SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=tuple(word.word_id for word in timeline.words[10:13]),
        semantic_summary="summary",
        editorial_reason="reason",
        confidence=0.9,
    )


def _envelope(timeline: CanonicalTimeline, core: SemanticCore) -> NarrativeEnvelope:
    return NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id="envelope",
        source_word_ids=tuple(word.word_id for word in timeline.words[8:18]),
        setup_resolved=True,
        payoff_resolved=True,
        confidence=0.9,
    )


def _planner(tmp_path: Path, provider: FakeEditorial) -> AutonomousQualityPlanner:
    return AutonomousQualityPlanner(provider, DagStore(tmp_path / "dag"))


def _plan(planner: AutonomousQualityPlanner, timeline: CanonicalTimeline):
    return planner.plan(
        timeline,
        multimodal=None,
        modality_profile=None,
        campaign_context={"objective": "find worthwhile moments"},
        relevant_policy={"source_segments": "editorial_only"},
        min_seconds=20.0,
        max_seconds=25.0,
    )


def test_model_word_boundaries_are_grounded_and_near_duplicate_cores_are_deduped() -> None:
    timeline = _timeline()
    payload = {
        "cores": [
            {
                "core_id": "a",
                "start_word_id": "w0000010",
                "end_word_id": "w0000014",
                "semantic_summary": "same event",
                "editorial_reason": "reason",
                "confidence": 0.9,
            },
            {
                "core_id": "b",
                "start_word_id": "w0000010",
                "end_word_id": "w0000014",
                "semantic_summary": "same event duplicate",
                "editorial_reason": "reason",
                "confidence": 0.8,
            },
        ]
    }
    cores = semantic_cores_from_payload(timeline, payload)
    assert len(cores) == 1
    assert cores[0].source_word_ids == tuple(word.word_id for word in timeline.words[10:15])


def test_model_word_boundaries_fail_closed_for_unknown_or_reversed_refs() -> None:
    timeline = _timeline()
    base = {
        "core_id": "a",
        "semantic_summary": "event",
        "editorial_reason": "reason",
        "confidence": 0.9,
    }
    with pytest.raises(AutonomousPlanningError, match="unknown canonical word reference"):
        semantic_cores_from_payload(
            timeline,
            {"cores": [{**base, "start_word_id": "missing", "end_word_id": "w0000012"}]},
        )
    with pytest.raises(AutonomousPlanningError, match="reverse source chronology"):
        semantic_cores_from_payload(
            timeline,
            {"cores": [{**base, "start_word_id": "w0000014", "end_word_id": "w0000012"}]},
        )


def test_narrative_envelope_must_reference_and_contain_supplied_core() -> None:
    timeline = _timeline()
    core = _core(timeline)
    payload = {
        "envelope_id": "model",
        "core_id": core.core_id,
        "start_word_id": "w0000008",
        "end_word_id": "w0000017",
        "required_prior_context": "setup",
        "required_followup_context": "payoff",
        "setup_resolved": True,
        "payoff_resolved": True,
        "reference_resolution": [],
        "confidence": 0.9,
    }
    envelope = narrative_envelope_from_payload(timeline, core, payload)
    assert envelope.contains(core)

    with pytest.raises(AutonomousPlanningError, match="wrong semantic core"):
        narrative_envelope_from_payload(timeline, core, {**payload, "core_id": "other"})
    with pytest.raises(AutonomousPlanningError, match="does not contain"):
        narrative_envelope_from_payload(
            timeline,
            core,
            {**payload, "start_word_id": "w0000020", "end_word_id": "w0000025"},
        )


def test_quality_model_can_only_select_supplied_deterministic_window_id() -> None:
    timeline = _timeline()
    core = _core(timeline)
    envelope = _envelope(timeline, core)
    windows = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20.0,
        max_seconds=25.0,
    )
    assert windows
    valid = {
        "core_id": core.core_id,
        "selected_window_id": windows[0].window_id,
        "decision": "PASS",
        "quality_score": 0.9,
        "rationale": "good",
        "opening_strategy": "open on the first complete source-grounded statement",
        "confidence": 0.9,
    }
    assessment = quality_assessment_from_payload(core, windows, valid)
    assert assessment is not None
    assert assessment.window_id == windows[0].window_id

    with pytest.raises(AutonomousPlanningError, match="supplied legal window ID"):
        quality_assessment_from_payload(
            core,
            windows,
            {**valid, "selected_window_id": "invented-window"},
        )


def test_dynamic_planner_produces_one_output_per_independent_quality_core(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = FakeEditorial()
    result = _plan(_planner(tmp_path, provider), timeline)

    assert len(result.cores) == 2
    assert len(result.quality_moments) == 2
    assert len({moment.core.core_id for moment in result.quality_moments}) == 2
    assert all(moment.delivery_window.duration >= 20.0 for moment in result.quality_moments)
    assert all(
        set(moment.envelope.source_word_ids).issubset(moment.delivery_window.source_word_ids)
        for moment in result.quality_moments
    )
    assert result.stage_executions == 5
    assert sum(task.startswith("semantic_cores:") for task in provider.calls) == 1


def test_zero_worthwhile_cores_is_valid_zero_yield_without_downstream_calls(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = FakeEditorial(zero_cores=True)
    result = _plan(_planner(tmp_path, provider), timeline)

    assert result.cores == ()
    assert result.quality_moments == ()
    assert len(provider.calls) == 1 and provider.calls[0].startswith("semantic_cores:")


def test_source_hazard_cache_identity_uses_full_model_contract(tmp_path: Path) -> None:
    timeline = _timeline()
    brief = CampaignBrief("campaign", "Title", "Objective")
    payload = {"words": [{"text": "evidence"}]}

    first_provider = FakeEditorial()
    first = SourceHazardClassifier(first_provider, DagStore(tmp_path / "hazards"))
    first_identity = first._identity(timeline, brief, "source_hazards:test", payload)

    second_provider = FakeEditorial()
    second_provider.identity = ModelIdentity(
        model_id="different-editor",
        revision="rev-1",
        quantization="int4",
        inference_engine="fake",
        prompt_version="editor",
        schema_version="editorial-json",
    )
    second = SourceHazardClassifier(second_provider, DagStore(tmp_path / "hazards"))
    second_identity = second._identity(timeline, brief, "source_hazards:test", payload)

    assert first_identity.cache_key != second_identity.cache_key


def test_dag_cache_does_not_cross_full_model_identity_with_same_revision(
    tmp_path: Path,
) -> None:
    timeline = _timeline()
    first_provider = FakeEditorial()
    _plan(_planner(tmp_path, first_provider), timeline)

    second_provider = FakeEditorial()
    second_provider.identity = ModelIdentity(
        model_id="different-editor",
        revision="rev-1",
        quantization="int4",
        inference_engine="fake",
        prompt_version="editor",
        schema_version="editorial-json",
    )
    second = _plan(_planner(tmp_path, second_provider), timeline)

    assert second_provider.calls
    assert second.stage_executions > 0


def test_dag_replay_reuses_paid_editorial_outputs_without_new_provider_calls(
    tmp_path: Path,
) -> None:
    timeline = _timeline()
    provider = FakeEditorial()
    first = _plan(_planner(tmp_path, provider), timeline)
    first_call_count = len(provider.calls)
    second = _plan(_planner(tmp_path, provider), timeline)

    assert len(first.quality_moments) == len(second.quality_moments) == 2
    assert len(provider.calls) == first_call_count
    assert second.stage_cache_hits == first_call_count
    assert second.stage_executions == 0


def test_overlong_complete_envelope_stops_before_quality_model_call(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = FakeEditorial(overlong_envelope=True)
    result = _plan(_planner(tmp_path, provider), timeline)

    assert result.quality_moments == ()
    assert all(not task.startswith("quality_windows:") for task in provider.calls)
    assert (
        sum(
            rejection.get("reasons") == ["no_campaign_legal_complete_window"]
            for rejection in result.rejections
        )
        == 2
    )


def test_required_visual_evidence_blocks_before_any_editorial_inference(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = FakeEditorial()
    profile = SourceModalityProfile(
        speech_dependency=0.2,
        visual_dependency=0.8,
        motion_dependency=0.7,
        screen_text_dependency=0.0,
        speaker_identity_dependency=0.0,
        action_dependency=0.7,
        visual_evidence_coverage=0.2,
        confidence=0.9,
    )
    planner = _planner(tmp_path, provider)

    with pytest.raises(RuntimeError, match="requires visual evidence"):
        planner.plan(
            timeline,
            multimodal=None,
            modality_profile=profile,
            campaign_context={},
            relevant_policy={},
            min_seconds=20.0,
            max_seconds=25.0,
        )
    assert provider.calls == []


class CapacityEditorial(FakeEditorial):
    def __init__(self, *, maximum_words: int) -> None:
        super().__init__(zero_cores=True)
        self.maximum_words = maximum_words
        self.word_counts: list[int] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        if task.startswith("semantic_cores:"):
            words = payload.get("words")
            count = len(words) if isinstance(words, list) else 0
            self.calls.append(task)
            self.word_counts.append(count)
            if count > self.maximum_words:
                raise EditorialCapacityError(
                    "synthetic capacity boundary",
                    details={"input_words": count},
                )
            return ProviderResult({"cores": []}, self.identity, _usage())
        return super().complete_json(task=task, payload=payload)


def test_semantic_planner_adaptively_splits_capacity_failures_without_fixed_chunk_size(
    tmp_path: Path,
) -> None:
    timeline = _timeline(60)
    provider = CapacityEditorial(maximum_words=20)
    result = _plan(_planner(tmp_path, provider), timeline)

    assert result.cores == ()
    assert provider.word_counts[0] == len(timeline.words)
    successful = [count for count in provider.word_counts if count <= provider.maximum_words]
    failed = [count for count in provider.word_counts if count > provider.maximum_words]
    assert successful
    assert failed
    assert sum(successful) == len(timeline.words)
    assert all(task.startswith("semantic_cores:") for task in provider.calls)


def test_semantic_planner_fails_closed_when_smallest_source_unit_exceeds_capacity(
    tmp_path: Path,
) -> None:
    timeline = _timeline(1)
    provider = CapacityEditorial(maximum_words=0)
    with pytest.raises(AutonomousPlanningError, match="smallest source-grounded semantic interval"):
        _plan(_planner(tmp_path, provider), timeline)
    assert provider.word_counts == [1]


class QualityCapacityEditorial(FakeEditorial):
    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        if task.startswith("quality_windows:"):
            self.calls.append(task)
            assert payload.get("capacity_repartitionable") is True
            raise EditorialCapacityError(
                "synthetic runtime capacity guard",
                details={
                    "reason": "runtime_input_guard",
                    "input_tokens": 70_000,
                    "runtime_safe_input_tokens": 65_536,
                },
            )
        return super().complete_json(task=task, payload=payload)


def test_quality_windows_fail_closed_on_runtime_capacity_before_retry_loop(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = QualityCapacityEditorial()
    planner = _planner(tmp_path, provider)

    with pytest.raises(
        AutonomousPlanningError,
        match="quality-window evidence exceeds runtime-safe editorial capacity",
    ):
        _plan(planner, timeline)

    quality_calls = [task for task in provider.calls if task.startswith("quality_windows:")]
    assert len(quality_calls) == 1
