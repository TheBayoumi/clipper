from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.quality_batch import plan_quality_batch
from clipper.visual import VisualEvidenceSpan, VisualEvent, VisualTimeline


class _QualityEditorial:
    identity = ModelIdentity(
        "quality-test-model",
        "quality-test-revision",
        "none",
        "test",
        "editor",
        "editorial-json",
    )

    def __init__(self, *, sponsor: bool = False, fail_if_called: bool = False) -> None:
        self.sponsor = sponsor
        self.fail_if_called = fail_if_called
        self.tasks: list[str] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        if self.fail_if_called:
            raise AssertionError(f"unexpected uncached editorial call: {task}")
        self.tasks.append(task)
        if task.startswith("source_hazards:"):
            words = payload["words"]
            value: dict[str, Any] = {
                "segments": [
                    {
                        "start_word_id": words[0]["word_ref"],
                        "end_word_id": words[-1]["word_ref"],
                        "classification": "sponsor_read" if self.sponsor else "editorial_content",
                        "confidence": 0.99,
                        "evidence": ["test source classification"],
                    }
                ]
            }
        elif task.startswith("semantic_cores:"):
            words = payload["words"]
            value = {
                "cores": [
                    {
                        "core_id": "model-local-core",
                        "start_word_id": words[10]["word_ref"],
                        "end_word_id": words[12]["word_ref"],
                        "semantic_summary": "one worthwhile complete idea",
                        "editorial_reason": "strong independent test moment",
                        "confidence": 0.95,
                    }
                ]
            }
        elif task.startswith("narrative_envelope:"):
            words = payload["source_context_words"]
            value = {
                "envelope_id": "model-local-envelope",
                "core_id": payload["core"]["core_id"],
                "start_word_id": words[5]["word_ref"],
                "end_word_id": words[24]["word_ref"],
                "required_prior_context": "",
                "required_followup_context": "",
                "setup_resolved": True,
                "payoff_resolved": True,
                "reference_resolution": [],
                "confidence": 0.95,
            }
        elif task.startswith("quality_windows:"):
            windows = payload["feasible_windows"]
            value = {
                "core_id": payload["core"]["core_id"],
                "selected_window_id": windows[0]["window_id"],
                "decision": "PASS",
                "quality_score": 0.94,
                "rationale": "worth publishing",
                "opening_strategy": "open on the first complete source-grounded statement",
                "confidence": 0.96,
            }
        else:
            raise AssertionError(f"unexpected editorial task: {task}")
        return ProviderResult(
            value,
            self.identity,
            InferenceUsage(
                provider="test",
                started_at=datetime.now(UTC).isoformat(),
                duration_seconds=0.01,
                input_units=10,
                output_units=5,
            ),
        )


def _timeline(count: int = 30) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker",
                0.99,
                "word_exact",
                "test",
            )
            for index in range(count)
        ),
    )


def _visual(timeline: CanonicalTimeline) -> VisualTimeline:
    return VisualTimeline(
        timeline.video_id,
        timeline.source_hash,
        (
            VisualEvent(
                0.0,
                timeline.end,
                "scene-1",
                "continuous source visual evidence",
                visible_speakers=("speaker",),
                confidence=0.99,
            ),
        ),
        coverage_spans=(
            VisualEvidenceSpan(0.0, timeline.end, timeline.end / 2, "source_policy"),
        ),
        source_duration=timeline.end,
    )


def test_quality_batch_yield_is_quality_derived_not_campaign_quota(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    provider = _QualityEditorial()
    result = plan_quality_batch(
        brief,
        {timeline.video_id: timeline},
        {timeline.video_id: _visual(timeline)},
        provider,
        dag_root=tmp_path / "dag",
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )

    assert len(result.quality_moments) == 1
    assert len(result.concepts) == 1
    assert len(result.variants) == 1
    assert len(result.plans) == 1
    assert len(result.story_moments) == 1
    assert len(result.plans) == len(result.quality_moments)
    assert result.plans[0].concept_id == result.quality_moments[0].quality_moment_id
    assert result.plans[0].pre_render_eligibility["decision"] == "PASS"
    assert result.source_evidence[timeline.video_id]["status"] == "PASS"
    assert result.stage_executions == 4
    assert result.stage_cache_hits == 0
    assert provider.tasks == [
        "source_hazards:0",
        "semantic_cores:0",
        next(task for task in provider.tasks if task.startswith("narrative_envelope:")),
        next(task for task in provider.tasks if task.startswith("quality_windows:")),
    ]
    assert result.to_dict()["quality_moments"]


def test_quality_batch_reuses_exact_dag_without_new_model_calls(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    dag_root = tmp_path / "dag"
    first = plan_quality_batch(
        brief,
        {timeline.video_id: timeline},
        {timeline.video_id: _visual(timeline)},
        _QualityEditorial(),
        dag_root=dag_root,
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    assert first.stage_executions == 4

    cached = plan_quality_batch(
        brief,
        {timeline.video_id: timeline},
        {timeline.video_id: _visual(timeline)},
        _QualityEditorial(fail_if_called=True),
        dag_root=dag_root,
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    assert len(cached.plans) == 1
    assert cached.stage_executions == 0
    assert cached.stage_cache_hits == 4
    assert cached.model_invocations == ()


def test_branding_policy_missing_visual_evidence_is_failure_not_zero_yield(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    with pytest.raises(RuntimeError, match="quality graph planning failed for every source"):
        plan_quality_batch(
            brief,
            {timeline.video_id: timeline},
            {},
            _QualityEditorial(),
            dag_root=tmp_path / "dag",
            max_words_per_chunk=200,
            chunk_overlap_words=20,
        )



def test_branding_policy_rejects_explicitly_insufficient_source_policy_coverage(
    tmp_path: Path,
) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    visual = VisualTimeline(
        timeline.video_id,
        timeline.source_hash,
        (
            VisualEvent(
                0.0,
                timeline.end,
                "scene",
                "semantic visual event duration must not determine policy coverage",
                confidence=0.9,
            ),
        ),
        coverage_spans=(VisualEvidenceSpan(0.0, 5.0, 2.5, "source_policy"),),
        source_duration=timeline.end,
    )
    with pytest.raises(RuntimeError, match=r"broader source visual evidence coverage: 0\.167"):
        plan_quality_batch(
            brief,
            {timeline.video_id: timeline},
            {timeline.video_id: visual},
            _QualityEditorial(),
            dag_root=tmp_path / "dag",
            max_words_per_chunk=200,
            chunk_overlap_words=20,
        )

def test_forbidden_sponsor_source_produces_legitimate_zero_quality_yield(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    provider = _QualityEditorial(sponsor=True)
    result = plan_quality_batch(
        brief,
        {timeline.video_id: timeline},
        {timeline.video_id: _visual(timeline)},
        provider,
        dag_root=tmp_path / "dag",
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    assert result.plans == ()
    assert result.quality_moments == ()
    assert result.source_evidence[timeline.video_id]["status"] == "PASS"
    assert any(
        item.get("reasons") == ["no_campaign_legal_complete_window"] for item in result.rejections
    )
    assert not any(task.startswith("quality_windows:") for task in provider.tasks)
