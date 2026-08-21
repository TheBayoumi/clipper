from __future__ import annotations

from dataclasses import replace
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
from clipper.models import SourceSpan
from clipper.multimodal_timeline import MultimodalTimeline
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.editorial_prompt import (
    editorial_contract,
    editorial_contract_fingerprint,
    editorial_json_schema,
    editorial_output_budget,
    editorial_task_family,
)
from clipper.quality_moments import (
    QualityMoment,
    WindowQualityAssessment,
    choose_quality_moments,
)
from clipper.stage_contracts import StageContract, StageIdentity, stage_identity
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.window_solver import (
    FeasibleDeliveryWindow,
    enumerate_feasible_windows,
    validate_feasible_window,
)


def _timeline(count: int = 12) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker-a",
                0.95,
                "word_exact",
                "test",
            )
            for index in range(count)
        ),
    )


def _graph() -> tuple[
    CanonicalTimeline,
    SemanticCore,
    NarrativeEnvelope,
    FeasibleDeliveryWindow,
    WindowQualityAssessment,
]:
    timeline = _timeline()
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=tuple(word.word_id for word in timeline.words[4:6]),
        semantic_summary="worthwhile idea",
        editorial_reason="independently useful",
        confidence=0.9,
    )
    envelope = NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id="envelope",
        source_word_ids=tuple(word.word_id for word in timeline.words[3:8]),
        setup_resolved=True,
        payoff_resolved=True,
        confidence=0.9,
    )
    window = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=5.0,
        max_seconds=5.0,
    )[0]
    assessment = WindowQualityAssessment(
        core.core_id,
        window.window_id,
        "PASS",
        0.9,
        "complete and worthwhile",
        0.9,
    )
    return timeline, core, envelope, window, assessment


def test_story_graph_rejects_invalid_identity_timestamps_provenance_and_confidence() -> None:
    timeline, core, envelope, _, _ = _graph()
    core_kwargs = {
        "core_id": "core",
        "video_id": "video",
        "source_hash": "source",
        "source_start": 1.0,
        "source_end": 2.0,
        "source_word_ids": ("word",),
        "semantic_summary": "summary",
        "editorial_reason": "reason",
        "confidence": 0.5,
    }
    for overrides, match in (
        ({"core_id": ""}, "stable source identity"),
        ({"source_start": -1.0}, "timestamps"),
        ({"source_word_ids": ()}, "source word provenance"),
        ({"semantic_summary": ""}, "summary and editorial reason"),
        ({"confidence": 2.0}, "confidence"),
    ):
        with pytest.raises(ValueError, match=match):
            SemanticCore(**{**core_kwargs, **overrides})

    with pytest.raises(ValueError, match="contiguous and chronological"):
        SemanticCore.from_word_ids(
            timeline,
            core_id="skipped",
            source_word_ids=(timeline.words[2].word_id, timeline.words[4].word_id),
            semantic_summary="summary",
            editorial_reason="reason",
            confidence=0.5,
        )

    envelope_kwargs = {
        "envelope_id": "envelope",
        "core_id": core.core_id,
        "video_id": "video",
        "source_hash": "source",
        "source_start": 1.0,
        "source_end": 2.0,
        "source_word_ids": ("word",),
        "required_prior_context": "",
        "required_followup_context": "",
        "setup_resolved": True,
        "payoff_resolved": True,
        "reference_resolution": (),
        "confidence": 0.5,
    }
    for overrides, match in (
        ({"envelope_id": ""}, "envelope_id and core_id"),
        ({"video_id": ""}, "stable source identity"),
        ({"source_start": -1.0}, "timestamps"),
        ({"source_word_ids": ()}, "source word provenance"),
        ({"confidence": -0.1}, "confidence"),
    ):
        with pytest.raises(ValueError, match=match):
            NarrativeEnvelope(**{**envelope_kwargs, **overrides})

    with pytest.raises(ValueError, match="contiguous and chronological"):
        NarrativeEnvelope.from_word_ids(
            timeline,
            core,
            envelope_id="skipped",
            source_word_ids=(timeline.words[2].word_id, timeline.words[4].word_id),
            setup_resolved=True,
            payoff_resolved=True,
            confidence=0.5,
        )
    assert envelope.complete is True
    assert envelope.to_dict()["complete"] is True


def test_feasible_window_constructor_and_solver_fail_closed() -> None:
    timeline, core, envelope, window, _ = _graph()
    window_kwargs = {
        "window_id": "window",
        "core_id": "core",
        "envelope_id": "envelope",
        "video_id": "video",
        "source_hash": "source",
        "source_start": 1.0,
        "source_end": 2.0,
        "source_word_ids": ("word",),
    }
    for overrides, match in (
        ({"window_id": ""}, "stable graph identity"),
        ({"video_id": ""}, "stable source identity"),
        ({"source_start": -1.0}, "timestamps"),
        ({"source_word_ids": ()}, "source word provenance"),
    ):
        with pytest.raises(ValueError, match=match):
            FeasibleDeliveryWindow(**{**window_kwargs, **overrides})

    with pytest.raises(ValueError, match="campaign duration bounds"):
        enumerate_feasible_windows(
            timeline,
            core,
            envelope,
            min_seconds=0,
            max_seconds=5,
        )
    with pytest.raises(ValueError, match="semantic core does not belong"):
        enumerate_feasible_windows(
            timeline,
            replace(core, video_id="other"),
            envelope,
            min_seconds=5,
            max_seconds=5,
        )
    with pytest.raises(ValueError, match="narrative envelope does not belong"):
        enumerate_feasible_windows(
            timeline,
            core,
            replace(envelope, source_hash="other"),
            min_seconds=5,
            max_seconds=5,
        )
    assert (
        enumerate_feasible_windows(
            timeline,
            core,
            replace(envelope, payoff_resolved=False),
            min_seconds=5,
            max_seconds=5,
        )
        == ()
    )
    assert enumerate_feasible_windows(timeline, core, envelope, min_seconds=2, max_seconds=4) == ()

    missing_word_envelope = replace(
        envelope,
        source_word_ids=("missing", *envelope.source_word_ids),
    )
    with pytest.raises(ValueError, match="outside the canonical timeline"):
        enumerate_feasible_windows(
            timeline,
            core,
            missing_word_envelope,
            min_seconds=5,
            max_seconds=6,
        )

    forbidden = SourceSpan(window.source_start + 0.25, window.source_start + 0.75)
    assert (
        enumerate_feasible_windows(
            timeline,
            core,
            envelope,
            min_seconds=5,
            max_seconds=5,
            forbidden_spans=(forbidden,),
        )
        == ()
    )

    short_envelope = NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id="short-envelope",
        source_word_ids=core.source_word_ids,
        setup_resolved=True,
        payoff_resolved=True,
        confidence=0.9,
    )
    padded = enumerate_feasible_windows(
        timeline,
        core,
        short_envelope,
        min_seconds=5,
        max_seconds=6,
    )
    assert padded
    assert all(set(short_envelope.source_word_ids) <= set(item.source_word_ids) for item in padded)


def test_feasible_window_validator_rejects_every_identity_and_containment_violation() -> None:
    _, core, envelope, window, _ = _graph()
    cases = (
        (replace(window, core_id="other"), core, envelope, "graph identity"),
        (replace(window, video_id="other"), core, envelope, "source identity"),
        (window, core, replace(envelope, video_id="other"), "narrative source identity"),
    )
    for candidate, candidate_core, candidate_envelope, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_feasible_window(
                candidate,
                candidate_core,
                candidate_envelope,
                min_seconds=5,
                max_seconds=5,
            )

    with pytest.raises(ValueError, match="duration bounds"):
        validate_feasible_window(window, core, envelope, min_seconds=6, max_seconds=7)
    with pytest.raises(ValueError, match="amputates the start"):
        validate_feasible_window(
            replace(window, source_start=envelope.source_start + 0.5),
            core,
            envelope,
            min_seconds=4,
            max_seconds=6,
        )
    with pytest.raises(ValueError, match="amputates the end"):
        validate_feasible_window(
            replace(window, source_end=envelope.source_end - 0.5),
            core,
            envelope,
            min_seconds=4,
            max_seconds=6,
        )
    with pytest.raises(ValueError, match="complete narrative envelope"):
        validate_feasible_window(
            replace(window, source_word_ids=core.source_word_ids),
            core,
            envelope,
            min_seconds=5,
            max_seconds=5,
        )

    envelope_without_core_words = replace(
        envelope,
        source_word_ids=(envelope.source_word_ids[0], envelope.source_word_ids[-1]),
    )
    with pytest.raises(ValueError, match="semantic core"):
        validate_feasible_window(
            replace(window, source_word_ids=envelope_without_core_words.source_word_ids),
            core,
            envelope_without_core_words,
            min_seconds=5,
            max_seconds=5,
        )
    with pytest.raises(ValueError, match="forbidden"):
        validate_feasible_window(
            window,
            core,
            envelope,
            min_seconds=5,
            max_seconds=5,
            forbidden_spans=(SourceSpan(3.5, 4.0),),
        )
    assert window.to_dict()["duration"] == pytest.approx(5.0)


def test_quality_contracts_reject_cross_graph_or_nonpassing_state() -> None:
    _, core, envelope, window, assessment = _graph()
    assessment_kwargs = {
        "core_id": core.core_id,
        "window_id": window.window_id,
        "decision": "PASS",
        "quality_score": 0.9,
        "rationale": "good",
        "confidence": 0.9,
    }
    for overrides, match in (
        ({"core_id": ""}, "core_id and window_id"),
        ({"decision": "UNKNOWN"}, "unsupported"),
        ({"quality_score": 2.0}, "between 0 and 1"),
        ({"rationale": ""}, "rationale"),
    ):
        with pytest.raises(ValueError, match=match):
            WindowQualityAssessment(**{**assessment_kwargs, **overrides})  # type: ignore[arg-type]

    valid = QualityMoment("quality:core", core, envelope, window, assessment)
    assert valid.to_dict()["quality_moment_id"] == "quality:core"
    cases = (
        ("", core, envelope, window, assessment, "stable identifier"),
        (
            "quality:core",
            core,
            envelope,
            replace(window, core_id="other"),
            assessment,
            "wrong semantic core",
        ),
        (
            "quality:core",
            core,
            envelope,
            replace(window, envelope_id="other"),
            assessment,
            "wrong narrative envelope",
        ),
        (
            "quality:core",
            core,
            envelope,
            replace(window, source_hash="other"),
            assessment,
            "wrong source",
        ),
        (
            "quality:core",
            core,
            envelope,
            replace(window, source_start=envelope.source_start + 0.5),
            assessment,
            "amputates narrative setup",
        ),
        (
            "quality:core",
            core,
            envelope,
            replace(window, source_end=envelope.source_end - 0.5),
            assessment,
            "amputates narrative payoff",
        ),
        (
            "quality:core",
            core,
            envelope,
            replace(window, source_word_ids=core.source_word_ids),
            assessment,
            "omits narrative-envelope evidence",
        ),
        (
            "quality:core",
            core,
            envelope,
            window,
            replace(assessment, core_id="other"),
            "wrong semantic core",
        ),
        (
            "quality:core",
            core,
            envelope,
            window,
            replace(assessment, window_id="other"),
            "wrong delivery window",
        ),
        (
            "quality:core",
            core,
            envelope,
            window,
            replace(assessment, decision="REJECT"),
            "only PASS",
        ),
    )
    for (
        moment_id,
        candidate_core,
        candidate_envelope,
        candidate_window,
        candidate_assessment,
        match,
    ) in cases:
        with pytest.raises(ValueError, match=match):
            QualityMoment(
                moment_id,
                candidate_core,
                candidate_envelope,
                candidate_window,
                candidate_assessment,
            )


def test_quality_selection_detects_duplicate_and_dangling_graph_references() -> None:
    _, core, envelope, window, assessment = _graph()
    with pytest.raises(ValueError, match="duplicate semantic core"):
        choose_quality_moments((core, core), (envelope,), (window,), (assessment,))
    with pytest.raises(ValueError, match="duplicate narrative envelope"):
        choose_quality_moments((core,), (envelope, envelope), (window,), (assessment,))
    with pytest.raises(ValueError, match="duplicate feasible window"):
        choose_quality_moments((core,), (envelope,), (window, window), (assessment,))
    with pytest.raises(ValueError, match="unknown window"):
        choose_quality_moments(
            (core,),
            (envelope,),
            (window,),
            (replace(assessment, window_id="missing"),),
        )
    with pytest.raises(ValueError, match="core/window identity mismatch"):
        choose_quality_moments(
            (core,),
            (envelope,),
            (window,),
            (replace(assessment, core_id="other"),),
        )
    assert (
        choose_quality_moments(
            (core,),
            (envelope,),
            (window,),
            (replace(assessment, decision="REJECT"),),
        )
        == ()
    )
    dangling_window = replace(window, envelope_id="missing")
    dangling_assessment = replace(assessment, window_id=dangling_window.window_id)
    with pytest.raises(ValueError, match="unknown narrative envelope"):
        choose_quality_moments(
            (core,),
            (envelope,),
            (dangling_window,),
            (dangling_assessment,),
        )


def test_stage_identity_validates_dependency_material_and_serializes_policy() -> None:
    contract = StageContract("quality", {"schema": "v1"}, {"logos": "forbid"})
    identity = stage_identity(
        contract,
        source_hash="source",
        dependency_output_hashes=("dependency",),
        model_revision="revision",
        decoding_parameters={"do_sample": False},
    )
    payload = identity.to_dict()
    assert payload["cache_key"] == identity.cache_key
    assert payload["dependency_output_hashes"] == ["dependency"]
    assert identity.relevant_policy_hash
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("", "source", "contract")
    with pytest.raises(ValueError, match="dependency output fingerprints"):
        StageIdentity("stage", "source", "contract", ("",))


def test_editorial_prompt_exposes_every_structured_task_family_and_budget() -> None:
    tasks = {
        "source_hazards:0": "source_hazards",
        "semantic_cores:0": "semantic_cores",
        "narrative_envelope:core": "narrative_envelope",
        "quality_windows:core": "quality_windows",
    }
    for task, family in tasks.items():
        assert editorial_task_family(task) == family
        schema = editorial_json_schema(task)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert editorial_contract(task)
        assert len(editorial_contract_fingerprint(task)) == 64

    assert editorial_output_budget({"task": "source_hazards:0"}) == 2048
    assert editorial_output_budget({"task": "semantic_cores:0"}) == 2048
    assert editorial_output_budget({"task": "narrative_envelope:x"}) == 1536
    assert editorial_output_budget({"task": "quality_windows:x"}) == 1536
    with pytest.raises(ValueError, match="unsupported production editorial task"):
        editorial_output_budget({"task": "other"})
    with pytest.raises(ValueError, match="unsupported production editorial task"):
        editorial_task_family("unsupported")
    with pytest.raises(ValueError, match="unsupported production editorial task"):
        editorial_json_schema("unsupported")


class _EdgeEditorial:
    identity = ModelIdentity("edge-editor", "rev", "none", "test", "editor", "schema")

    def __init__(self, mode: str) -> None:
        self.mode = mode
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
                "cores": [
                    {
                        "core_id": "model-core",
                        "start_word_id": "w0000004",
                        "end_word_id": "w0000005",
                        "semantic_summary": "worthwhile idea",
                        "editorial_reason": "independently useful",
                        "confidence": 0.9,
                    }
                ]
            }
        elif task.startswith("narrative_envelope:"):
            core = payload["core"]
            value = {
                "envelope_id": "model-envelope",
                "core_id": core["core_id"],
                "start_word_id": "w0000003",
                "end_word_id": "w0000007",
                "required_prior_context": "setup",
                "required_followup_context": "payoff",
                "setup_resolved": self.mode != "incomplete",
                "payoff_resolved": True,
                "reference_resolution": [],
                "confidence": 0.9,
            }
        elif task.startswith("quality_windows:"):
            value = {
                "core_id": payload["core"]["core_id"],
                "selected_window_id": None,
                "decision": "REJECT",
                "quality_score": 0.4,
                "rationale": "not strong enough",
                "confidence": 0.9,
            }
        else:  # pragma: no cover - defensive fake
            raise AssertionError(task)
        return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))


class _NonObjectEditorial(_EdgeEditorial):
    def complete_json(self, *, task: str, payload: dict[str, Any]) -> ProviderResult[Any]:
        del task, payload
        return ProviderResult(["not-an-object"], self.identity, InferenceUsage("test", "now", 0.01))


def test_autonomous_payload_parsers_fail_closed_on_malformed_model_output() -> None:
    timeline, core, _, window, _ = _graph()
    with pytest.raises(AutonomousPlanningError, match="cores array"):
        semantic_cores_from_payload(timeline, {})
    with pytest.raises(AutonomousPlanningError, match="entry must be an object"):
        semantic_cores_from_payload(timeline, {"cores": ["bad"]})
    with pytest.raises(AutonomousPlanningError, match="string word references"):
        semantic_cores_from_payload(
            timeline,
            {
                "cores": [
                    {
                        "start_word_id": 4,
                        "end_word_id": "w0000005",
                        "semantic_summary": "summary",
                        "editorial_reason": "reason",
                        "confidence": 0.9,
                    }
                ]
            },
        )
    with pytest.raises(AutonomousPlanningError, match="confidence must be numeric"):
        semantic_cores_from_payload(
            timeline,
            {
                "cores": [
                    {
                        "start_word_id": "w0000004",
                        "end_word_id": "w0000005",
                        "semantic_summary": "summary",
                        "editorial_reason": "reason",
                        "confidence": True,
                    }
                ]
            },
        )

    envelope_payload: dict[str, Any] = {
        "core_id": core.core_id,
        "start_word_id": "w0000003",
        "end_word_id": "w0000007",
        "required_prior_context": "",
        "required_followup_context": "",
        "setup_resolved": True,
        "payoff_resolved": True,
        "reference_resolution": [],
        "confidence": 0.9,
    }
    with pytest.raises(AutonomousPlanningError, match="wrong semantic core"):
        narrative_envelope_from_payload(timeline, core, {**envelope_payload, "core_id": "other"})
    with pytest.raises(AutonomousPlanningError, match="confidence must be numeric"):
        narrative_envelope_from_payload(timeline, core, {**envelope_payload, "confidence": True})
    with pytest.raises(AutonomousPlanningError, match="string array"):
        narrative_envelope_from_payload(
            timeline,
            core,
            {**envelope_payload, "reference_resolution": "bad"},
        )
    with pytest.raises(AutonomousPlanningError, match="does not contain"):
        narrative_envelope_from_payload(
            timeline,
            core,
            {**envelope_payload, "start_word_id": "w0000008", "end_word_id": "w0000009"},
        )

    quality_payload = {
        "core_id": core.core_id,
        "selected_window_id": None,
        "decision": "REJECT",
        "quality_score": 0.4,
        "rationale": "reject",
        "confidence": 0.8,
    }
    with pytest.raises(AutonomousPlanningError, match="wrong semantic core"):
        quality_assessment_from_payload(core, (window,), {**quality_payload, "core_id": "other"})
    with pytest.raises(AutonomousPlanningError, match="unsupported"):
        quality_assessment_from_payload(core, (window,), {**quality_payload, "decision": "MAYBE"})
    with pytest.raises(AutonomousPlanningError, match="must be numeric"):
        quality_assessment_from_payload(core, (window,), {**quality_payload, "quality_score": True})
    with pytest.raises(AutonomousPlanningError, match="unknown window"):
        quality_assessment_from_payload(
            core,
            (window,),
            {**quality_payload, "selected_window_id": "missing"},
        )
    assert quality_assessment_from_payload(core, (window,), quality_payload) is None


def test_autonomous_planner_constructor_source_guards_and_rejection_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 200"):
        AutonomousQualityPlanner(
            _EdgeEditorial("reject"), DagStore(tmp_path / "small"), max_words_per_chunk=100
        )
    with pytest.raises(ValueError, match="chunk overlap"):
        AutonomousQualityPlanner(
            _EdgeEditorial("reject"),
            DagStore(tmp_path / "overlap"),
            max_words_per_chunk=200,
            chunk_overlap_words=200,
        )
    with pytest.raises(ValueError, match="context is too small"):
        AutonomousQualityPlanner(
            _EdgeEditorial("reject"),
            DagStore(tmp_path / "context"),
            max_words_per_chunk=200,
            envelope_context_words=50,
        )

    timeline = _timeline()
    planner = AutonomousQualityPlanner(_EdgeEditorial("reject"), DagStore(tmp_path / "guards"))
    with pytest.raises(ValueError, match="duration bounds"):
        planner.plan(
            timeline,
            multimodal=None,
            modality_profile=None,
            campaign_context={},
            relevant_policy={},
            min_seconds=0,
            max_seconds=5,
        )
    with pytest.raises(AutonomousPlanningError, match="different sources"):
        planner.plan(
            timeline,
            multimodal=MultimodalTimeline("other", "source", timeline.end, ()),
            modality_profile=None,
            campaign_context={},
            relevant_policy={},
            min_seconds=5,
            max_seconds=5,
        )

    incomplete_provider = _EdgeEditorial("incomplete")
    incomplete = AutonomousQualityPlanner(
        incomplete_provider,
        DagStore(tmp_path / "incomplete"),
    ).plan(
        timeline,
        multimodal=None,
        modality_profile=None,
        campaign_context={},
        relevant_policy={},
        min_seconds=5,
        max_seconds=5,
    )
    assert incomplete.quality_moments == ()
    assert incomplete.rejections[0]["reasons"] == ["incomplete_narrative_envelope"]
    assert all(not task.startswith("quality_windows:") for task in incomplete_provider.calls)
    assert incomplete.to_dict()["stage_executions"] == 2

    rejected_provider = _EdgeEditorial("reject")
    rejected = AutonomousQualityPlanner(
        rejected_provider,
        DagStore(tmp_path / "rejected"),
    ).plan(
        timeline,
        multimodal=None,
        modality_profile=None,
        campaign_context={},
        relevant_policy={},
        min_seconds=5,
        max_seconds=5,
    )
    assert rejected.quality_moments == ()
    assert rejected.rejections[-1]["stage"] == "quality_windows"
    assert rejected.rejections[-1]["decision"] == "REJECT"

    non_object = AutonomousQualityPlanner(
        _NonObjectEditorial("reject"),
        DagStore(tmp_path / "non-object"),
    )
    with pytest.raises(AutonomousPlanningError, match="non-object payload"):
        non_object.plan(
            timeline,
            multimodal=None,
            modality_profile=None,
            campaign_context={},
            relevant_policy={},
            min_seconds=5,
            max_seconds=5,
        )
