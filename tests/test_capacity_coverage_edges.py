from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from clipper.autonomous_quality_planner import (
    AutonomousPlanningError,
    AutonomousQualityPlanner,
    quality_assessment_from_payload,
)
from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.editorial_capacity import shrink_context_around_interval
from clipper.providers.base import (
    EditorialCapacityError,
    InferenceUsage,
    ModelIdentity,
    ProviderResult,
    compute_profile,
)
from clipper.providers.local import LocalEditorialProvider, ProviderUnavailable
from clipper.providers.modal import ModalJSONProvider
from clipper.quality_batch import RecordingEditorialProvider, plan_quality_batch
from clipper.source_hazards import SourceHazardClassifier
from clipper.stage_contracts import structural_contract_fingerprint
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.window_solver import enumerate_feasible_windows


def _timeline(count: int = 8) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{i:07d}:digest",
                f"w{i}",
                float(i),
                float(i + 1),
                "speaker",
                0.99,
                "word_exact",
                "test",
            )
            for i in range(count)
        ),
    )


class _Tensor:
    def __init__(self, count: int) -> None:
        self.count = count
        self.shape = (1, count)

    def __getitem__(self, _key: object) -> _Tensor:
        return self

    def numel(self) -> int:
        return self.count


class _Batch(dict[str, _Tensor]):
    def to(self, _device: object) -> _Batch:
        return self


class _Tokenizer:
    def __init__(self, model_max_length: object) -> None:
        self.model_max_length = model_max_length

    def apply_chat_template(self, _messages: object, **_kwargs: object) -> str:
        return "rendered"

    def __call__(self, _rendered: str, **_kwargs: object) -> _Batch:
        return _Batch(input_ids=_Tensor(3))

    def decode(self, _generated: object, **_kwargs: object) -> str:
        return "{}"


class _Model:
    device = "cpu"

    def __init__(self, context: object = None) -> None:
        self.config = SimpleNamespace(max_position_embeddings=context)

    def generate(self, **_kwargs: object) -> list[_Tensor]:
        return [_Tensor(1)]


def test_local_editorial_context_metadata_fail_closed_paths() -> None:
    provider = LocalEditorialProvider()
    provider._model = _Model(context=None)
    provider._tokenizer = _Tokenizer(model_max_length="invalid")
    with pytest.raises(EditorialCapacityError, match="does not expose"):
        provider.complete_json(task="semantic_cores:x", payload={})

    provider._model = _Model(context=3)
    provider._tokenizer = _Tokenizer(model_max_length=128)
    with pytest.raises(EditorialCapacityError, match="exceeds model context") as caught:
        provider.complete_json(task="semantic_cores:x", payload={})
    assert caught.value.details["input_tokens"] == 3
    assert caught.value.details["context_limit_tokens"] == 3


def test_quality_decision_requires_nonempty_opening_strategy() -> None:
    timeline = _timeline(6)
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=(timeline.words[1].word_id, timeline.words[2].word_id),
        semantic_summary="summary",
        editorial_reason="reason",
        confidence=0.9,
    )
    envelope = NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id="envelope",
        source_word_ids=tuple(word.word_id for word in timeline.words[1:5]),
        setup_resolved=True,
        payoff_resolved=True,
        confidence=0.9,
    )
    windows = enumerate_feasible_windows(
        timeline,
        core,
        envelope,  # type: ignore[arg-type]
        min_seconds=2.0,
        max_seconds=5.0,
    )
    assert windows
    payload = {
        "core_id": core.core_id,
        "selected_window_id": windows[0].window_id,
        "decision": "PASS",
        "quality_score": 0.8,
        "opening_strategy": "",
        "rationale": "reason",
        "confidence": 0.9,
    }
    with pytest.raises(AutonomousPlanningError, match="opening strategy"):
        quality_assessment_from_payload(core, windows, payload)


class _EscapeEditorial:
    identity = ModelIdentity("escape", "rev", "none", "test", "editor", "schema")

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            raise EditorialCapacityError("split")
        return ProviderResult(
            {
                "cores": [
                    {
                        "core_id": "escape",
                        "start_word_id": "w0000006",
                        "end_word_id": "w0000007",
                        "semantic_summary": "outside",
                        "editorial_reason": "outside",
                        "confidence": 0.9,
                    }
                ]
            },
            self.identity,
            InferenceUsage("test", "now", 0.0),
        )


def test_semantic_split_rejects_core_escaping_subrange(tmp_path: Path) -> None:
    planner = AutonomousQualityPlanner(_EscapeEditorial(), DagStore(tmp_path / "dag"))
    with pytest.raises(AutonomousPlanningError, match="outside supplied evidence"):
        planner._semantic_cores_adaptive(
            _timeline(8),
            multimodal=None,
            campaign_context={},
            relevant_policy={},
        )


class _AlwaysCapacity:
    identity = ModelIdentity("cap", "rev", "none", "test", "editor", "schema")

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        raise EditorialCapacityError("always full")


def test_envelope_fails_closed_when_core_only_context_still_exceeds_capacity(
    tmp_path: Path,
) -> None:
    timeline = _timeline(4)
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=(timeline.words[1].word_id, timeline.words[2].word_id),
        semantic_summary="summary",
        editorial_reason="reason",
        confidence=0.9,
    )
    planner = AutonomousQualityPlanner(_AlwaysCapacity(), DagStore(tmp_path / "dag"))
    with pytest.raises(AutonomousPlanningError, match="minimum semantic-core context"):
        planner._narrative_envelope_adaptive(
            timeline,
            core,
            multimodal=None,
            campaign_context={},
            relevant_policy={},
        )


def test_disabled_hazard_policy_and_empty_legacy_ranges(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    disabled = replace(brief, acceptance_policy=replace(brief.acceptance_policy, enabled=False))
    classifier = SourceHazardClassifier(_AlwaysCapacity(), DagStore(tmp_path / "haz"))
    result = classifier.classify(disabled, _timeline(2), multimodal=None)
    assert result.hazards == ()
    assert SourceHazardClassifier._legacy_ranges(CanonicalTimeline("v", "s", ())) == ()


def test_context_shrink_required_only_fallback_path() -> None:
    timeline = _timeline(4)
    assert shrink_context_around_interval(timeline, 0, 4, 0, 4) is None


def test_small_runtime_guard_branches() -> None:
    with pytest.raises(ValueError, match="unsupported compute profile"):
        compute_profile("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a name"):
        structural_contract_fingerprint("", CanonicalWord)

    provider = ModalJSONProvider(
        app_name="app",
        identity=ModelIdentity("m", "r", "none", "test"),
        function_name="f",
    )
    with (
        patch(
            "clipper.providers.modal.importlib.import_module", side_effect=ImportError("missing")
        ),
        pytest.raises(ProviderUnavailable, match=r"install clipper\[modal\]"),
    ):
        provider._modal()


class _SuccessEditorial:
    identity = ModelIdentity("success", "rev", "none", "test")

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        return ProviderResult(
            {"ok": True},
            self.identity,
            InferenceUsage("test", "now", 0.0),
        )


def test_recording_provider_emits_success_progress() -> None:
    progress: list[tuple[str, str]] = []
    recorder = RecordingEditorialProvider(
        _SuccessEditorial(),
        progress_callback=lambda task, state: progress.append((task, state)),
    )
    recorder.complete_json(task="x", payload={})
    assert progress == [("x", "running"), ("x", "success")]


def test_narrative_envelope_rejects_result_escaping_shrunk_context(tmp_path: Path) -> None:
    timeline = _timeline(8)
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=(timeline.words[3].word_id, timeline.words[4].word_id),
        semantic_summary="summary",
        editorial_reason="reason",
        confidence=0.9,
    )
    planner = AutonomousQualityPlanner(_AlwaysCapacity(), DagStore(tmp_path / "escape"))
    calls = 0

    def complete(
        _timeline_arg: CanonicalTimeline,
        _stage: str,
        _payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise EditorialCapacityError("shrink")
        return {
            "envelope_id": "escape",
            "core_id": core.core_id,
            "start_word_id": "w0000000",
            "end_word_id": "w0000005",
            "required_prior_context": "setup",
            "required_followup_context": "payoff",
            "setup_resolved": True,
            "payoff_resolved": True,
            "reference_resolution": [],
            "confidence": 0.9,
        }

    planner._complete = complete  # type: ignore[method-assign]
    with pytest.raises(AutonomousPlanningError, match="escaped the supplied adaptive context"):
        planner._narrative_envelope_adaptive(
            timeline,
            core,
            multimodal=None,
            campaign_context={},
            relevant_policy={},
        )


def _duplicate_batch_fixture(unique_concepts: bool) -> tuple[Any, Any, Any]:
    hazard = SimpleNamespace(
        rejections=(),
        hazards=(),
        stage_cache_hits=0,
        stage_executions=0,
        to_dict=lambda: {},
    )
    moments = (SimpleNamespace(quality_moment_id="q1"), SimpleNamespace(quality_moment_id="q2"))
    planning = SimpleNamespace(
        rejections=(),
        cores=(),
        quality_moments=moments,
        stage_cache_hits=0,
        stage_executions=0,
        to_dict=lambda: {},
    )
    counter = {"n": 0}

    def adapt(*_args: Any, **_kwargs: Any) -> Any:
        counter["n"] += 1
        concept_id = f"c{counter['n']}" if unique_concepts else "dup"
        return SimpleNamespace(
            concept=SimpleNamespace(concept_id=concept_id),
            variant=SimpleNamespace(),
            plan=SimpleNamespace(plan_id="dup-plan"),
            boundary_audit=SimpleNamespace(to_dict=lambda _policy: {}),
            policy_audit=SimpleNamespace(to_dict=lambda: {}),
        )

    return hazard, planning, adapt


@pytest.mark.parametrize(
    ("unique_concepts", "message"),
    [
        (False, "duplicate compatibility concept identities"),
        (True, "duplicate compatibility plan identities"),
    ],
)
def test_quality_batch_rejects_duplicate_compatibility_identities(
    tmp_path: Path,
    unique_concepts: bool,
    message: str,
) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline(8)
    hazard, planning, adapt = _duplicate_batch_fixture(unique_concepts)
    with (
        patch("clipper.quality_batch._requires_source_visual_policy", return_value=False),
        patch("clipper.quality_batch.SourceHazardClassifier.classify", return_value=hazard),
        patch("clipper.quality_batch.AutonomousQualityPlanner.plan", return_value=planning),
        patch("clipper.quality_batch.adapt_quality_moment", side_effect=adapt),
        patch("clipper.quality_batch._story_moment", return_value=SimpleNamespace()),
        pytest.raises(RuntimeError, match=message),
    ):
        plan_quality_batch(
            brief,
            {timeline.video_id: timeline},
            {},
            _SuccessEditorial(),
            dag_root=tmp_path / "batch",
        )


def test_package_lazy_pipeline_exports_and_unknown_attribute() -> None:
    import clipper

    assert clipper.PipelineSettings is not None
    assert clipper.run_pipeline is not None
    with pytest.raises(AttributeError):
        _value = clipper.definitely_missing  # type: ignore[attr-defined]
