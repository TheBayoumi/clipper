from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.models import (
    ClipCandidate,
    EditPlan,
    PipelineManifest,
    TranscriptSegment,
    VideoCandidate,
)
from clipper.pipeline import (
    PipelineSettings,
    _campaign_watermark,
    _copy_render_sidecars,
    _download_asset,
    _normalize_asset_url,
    _record_source_media_metadata,
    _rendered_clip,
    _renderer_for_source,
    _run_id,
    _source_media,
    _speaker_focus_for_source,
    _target_candidates,
    _tracking_transitions,
    _visual_timeline,
    run_pipeline,
)
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.quality_batch import QualityBatchResult
from clipper.quality_moments import QualityMoment, WindowQualityAssessment
from clipper.quality_pipeline import adapt_quality_moment
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.visual import VisualEvent, VisualTimeline
from clipper.visual_ai import VisualReviewIssue, VisualReviewReport
from clipper.window_solver import enumerate_feasible_windows


class FakeSource:
    def __init__(self, media: Path, *, watermark: Path | None = None) -> None:
        self.media = media
        self.watermark = watermark
        self.downloads = 0

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        del video, work_dir
        self.downloads += 1
        return self.media

    def campaign_watermark(self, brief) -> Path | None:
        del brief
        return self.watermark


class FakeTranscription:
    identity = ModelIdentity("fake-asr", "r1", "none", "test", "none", "canonical-v1")

    def __init__(self, *, empty: bool = False) -> None:
        self.calls = 0
        self.empty = empty

    def transcribe(self, source: Path, *, video_id: str, source_hash: str):
        self.calls += 1
        assert source.is_file()
        words = () if self.empty else _words(video_id, 60)
        timeline = CanonicalTimeline(video_id, source_hash, words)
        return ProviderResult(timeline, self.identity, _usage())


class FakeAlignment:
    identity = ModelIdentity("fake-align", "r1", "none", "test", "none", "canonical-v1")

    def __init__(self) -> None:
        self.calls = 0

    def align(self, source: Path, timeline: CanonicalTimeline):
        self.calls += 1
        assert source.is_file()
        return ProviderResult(timeline, self.identity, _usage())


class FakeDiarization:
    identity = ModelIdentity("fake-diarize", "r1", "none", "test", "none", "canonical-v1")

    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, source: Path, timeline: CanonicalTimeline):
        self.calls += 1
        assert source.is_file()
        return ProviderResult(timeline, self.identity, _usage())


class FakeEditorial:
    identity = ModelIdentity("fake-editor", "r1", "none", "test", "editor", "editorial-json")

    def complete_json(self, *, task: str, payload: dict[str, object]):
        raise AssertionError((task, payload))


class FakeVision:
    identity = ModelIdentity("fake-vlm", "r1", "none", "test", "visual", "visual-json")

    def inspect(self, *, task: str, frames: list[Path], context: dict[str, object]):
        raise AssertionError((task, frames, context))


class FakeRenderer:
    def __init__(self, *, fail_calls: set[int] | None = None, omit_sidecars: bool = False) -> None:
        self.calls = 0
        self.fail_calls = fail_calls or set()
        self.omit_sidecars = omit_sidecars
        self.watermarks: list[Path | None] = []

    def render(
        self,
        source_path: Path,
        output_path: Path,
        clip: ClipCandidate,
        segments: list[TranscriptSegment],
        watermark_path: Path | None = None,
        edit_plan: EditPlan | None = None,
    ) -> Path:
        self.calls += 1
        assert source_path.is_file()
        assert clip.duration > 0
        assert segments
        assert edit_plan is not None
        self.watermarks.append(watermark_path)
        if self.calls in self.fail_calls:
            raise RuntimeError(f"render failure {self.calls}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"mp4-{edit_plan.plan_id}".encode())
        if not self.omit_sidecars:
            output_path.with_suffix(".ass").write_text("{\\ko10}word", encoding="utf-8")
            output_path.with_suffix(".caption-audit.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            output_path.with_suffix(".tracking.json").write_text(
                json.dumps({"transitions": []}), encoding="utf-8"
            )
        return output_path


def _usage() -> InferenceUsage:
    return InferenceUsage("test", "2026-08-22T00:00:00Z", 0.01)


def _words(video_id: str, count: int) -> tuple[CanonicalWord, ...]:
    return tuple(
        CanonicalWord(
            f"{video_id}:w{index:07d}:x",
            f"word-{index}",
            float(index),
            float(index + 1),
            "speaker-a",
            0.99,
            "word_exact",
            "test",
        )
        for index in range(count)
    )


def _write_brief(
    path: Path,
    *,
    watermark_url: str | None = None,
    media_url: str | None = None,
    acceptance_policy: dict[str, object] | None = None,
) -> Path:
    target: dict[str, object] = {
        "video_id": "v1",
        "url": "https://www.youtube.com/watch?v=v1",
        "channel_id": "UC1",
    }
    if media_url is not None:
        target["media_url"] = media_url
    payload: dict[str, object] = {
        "campaign_id": "pipeline-contract",
        "title": "Podcast",
        "objective": "Find independently worthwhile complete moments",
        "targets": {"mode": "explicit", "videos": [target]},
        "rights": {"confirmed": True, "authorized_channels": ["UC1"]},
        "content_constraints": {"min_clip_seconds": 20, "max_clip_seconds": 25},
    }
    if watermark_url is not None:
        payload["watermark_url"] = watermark_url
    if acceptance_policy is not None:
        payload["acceptance_policy"] = acceptance_policy
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_visual(
    media_path: Path,
    video: VideoCandidate,
    timeline: CanonicalTimeline,
    provider: FakeVision,
    run_dir: Path,
):
    del media_path, provider, run_dir
    visual = VisualTimeline(
        video.video_id,
        timeline.source_hash,
        (VisualEvent(0.0, timeline.end, "scene-1", "source footage", ("speaker-a",), (), 0.99),),
    )
    return visual, {"model": FakeVision.identity.to_dict(), "usage": {}, "degraded": False}


def _quality_result(
    timelines: dict[str, CanonicalTimeline],
    root: Path,
    *,
    count: int = 1,
    reserve: bool = False,
) -> QualityBatchResult:
    timeline = timelines["v1"]
    brief = load_brief(root / "brief.json")
    moments: list[QualityMoment] = []
    concepts = []
    variants = []
    plans: list[EditPlan] = []
    for index in range(count):
        start = 5 + index * 30
        core = SemanticCore.from_word_ids(
            timeline,
            core_id=f"core-{index}",
            source_word_ids=tuple(word.word_id for word in timeline.words[start + 5 : start + 8]),
            semantic_summary=f"worthwhile idea {index}",
            editorial_reason="independently publishable",
            confidence=0.95,
        )
        envelope = NarrativeEnvelope.from_word_ids(
            timeline,
            core,
            envelope_id=f"envelope-{index}",
            source_word_ids=tuple(word.word_id for word in timeline.words[start : start + 20]),
            setup_resolved=True,
            payoff_resolved=True,
            confidence=0.95,
        )
        window = enumerate_feasible_windows(
            timeline,
            core,
            envelope,
            min_seconds=brief.min_clip_seconds,
            max_seconds=brief.max_clip_seconds,
        )[0]
        assessment = WindowQualityAssessment(
            core.core_id,
            window.window_id,
            "PASS",
            0.94,
            "open on the first complete source-grounded statement",
            "complete and worth publishing",
            0.96,
        )
        moment = QualityMoment(f"quality:{core.core_id}", core, envelope, window, assessment)
        adapted = adapt_quality_moment(brief, timeline, moment, hazards=(), branding=())
        assert adapted is not None
        moments.append(moment)
        concepts.append(adapted.concept)
        variants.append(adapted.variant)
        plans.append(adapted.plan)
        if reserve:
            reserve_plan = replace(
                adapted.plan,
                plan_id=f"{adapted.plan.plan_id}:reserve",
                variant_id=f"{adapted.variant.variant_id}:reserve",
            )
            plans.append(reserve_plan)
    return QualityBatchResult(
        story_moments=(),
        concepts=tuple(concepts),
        variants=tuple(variants),
        plans=tuple(plans),
        quality_moments=tuple(moments),
        rejections=(),
        model_invocations=(),
        boundary_audits=(),
        campaign_policy_audits=(),
        source_hazards=(),
        source_evidence={
            "v1": {
                "semantic_cores": count,
                "modality_profile": {"requires_speaker_identity": True},
            }
        },
        stage_cache_hits=0,
        stage_executions=max(1, count * 3),
        stage_dag_root=root / "dag",
    )


def _empty_quality(root: Path) -> QualityBatchResult:
    return QualityBatchResult((), (), (), (), (), (), (), (), (), (), {}, 0, 1, root / "dag")


def _review_result(decision: str = "PASS", *, issues=()):
    report = VisualReviewReport(decision, "review", 0.99, tuple(issues))
    result = ProviderResult({"decision": decision}, FakeVision.identity, _usage())
    return report, [result]


def _run_with_quality(
    tmp_path: Path,
    quality_factory,
    *,
    renderer: FakeRenderer | None = None,
    render: bool = True,
    qc=None,
    review=None,
    source: FakeSource | None = None,
):
    brief_path = _write_brief(tmp_path / "brief.json")
    media = tmp_path / "source.mkv"
    media.write_bytes(b"authorized-source-master")
    source = source or FakeSource(media)
    transcription = FakeTranscription()
    alignment = FakeAlignment()
    diarization = FakeDiarization()
    editor = FakeEditorial()
    vision = FakeVision()

    def quality_side_effect(brief, timelines, visual_timelines, editorial, *, dag_root):
        del brief, visual_timelines, editorial, dag_root
        return quality_factory(timelines, tmp_path)

    qc_value = qc or {"status": "PASS", "issues": [], "captions": {"alignment": "PASS"}}
    review_value = review or _review_result()
    with (
        patch("clipper.pipeline._visual_timeline", side_effect=_fake_visual),
        patch("clipper.pipeline.plan_quality_batch", side_effect=quality_side_effect),
        patch(
            "clipper.pipeline.run_technical_qc",
            side_effect=qc_value if callable(qc_value) else None,
            return_value=None if callable(qc_value) else qc_value,
        ),
        patch(
            "clipper.pipeline.review_rendered_clip",
            side_effect=review_value if callable(review_value) else None,
            return_value=None if callable(review_value) else review_value,
        ),
    ):
        run_dir = run_pipeline(
            brief_path,
            settings=PipelineSettings(
                artifact_root=tmp_path / "artifacts",
                cache_root=tmp_path / "cache",
            ),
            source_client=source,
            renderer=renderer or FakeRenderer(),
            editorial_provider=editor,
            visual_scout_provider=vision,
            visual_review_provider=vision,
            transcription_provider=transcription,
            alignment_provider=alignment,
            diarization_provider=diarization,
            render=render,
        )
    return run_dir, transcription, alignment, diarization


def test_pipeline_planning_uses_only_explicit_target_and_writes_contract_artifacts(
    tmp_path: Path,
) -> None:
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root),
        render=False,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "SUCCESS"
    assert manifest["status_reason"] == "planning_complete"
    assert manifest["publication_state"] == "PLANNED_NOT_RENDERED"
    assert [item["video_id"] for item in manifest["discovered_videos"]] == ["v1"]
    assert manifest["run_metadata"]["architecture"] == "autonomous-multimodal-quality-graph"
    assert manifest["targets"] == {"eligible_quality_moments": 1}
    assert len(manifest["planned_clips"]) == 1
    assert (run_dir / "canonical" / "v1.json").is_file()
    assert (run_dir / "visual-strategy").is_dir()
    assert json.loads((run_dir / "coverage.json").read_text()) == {
        "explicit_targets": 1,
        "grounded_targets": 1,
        "eligible_quality_moments": 1,
        "accepted_quality_moments": 0,
    }


def test_grounding_cache_reuses_exact_model_and_source_identity(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path / "brief.json")
    media = tmp_path / "source.mkv"
    media.write_bytes(b"same-source")
    source = FakeSource(media)
    t = FakeTranscription()
    a = FakeAlignment()
    d = FakeDiarization()
    settings = PipelineSettings(artifact_root=tmp_path / "runs", cache_root=tmp_path / "cache")

    def quality(*_args, **_kwargs):
        return _empty_quality(tmp_path)

    with (
        patch("clipper.pipeline._visual_timeline", side_effect=_fake_visual),
        patch("clipper.pipeline.plan_quality_batch", side_effect=quality),
        patch("clipper.pipeline._run_id", side_effect=["first", "second"]),
    ):
        for _ in range(2):
            run_pipeline(
                brief_path,
                settings=settings,
                source_client=source,
                editorial_provider=FakeEditorial(),
                visual_scout_provider=FakeVision(),
                transcription_provider=t,
                alignment_provider=a,
                diarization_provider=d,
                render=False,
            )
    assert (t.calls, a.calls, d.calls) == (1, 1, 1)
    second_manifest = json.loads((settings.artifact_root / "second" / "manifest.json").read_text())
    assert second_manifest["cache"]["hits"] == 3


def test_run_id_is_execution_unique_and_traceable() -> None:
    first_execution = "a" * 32
    second_execution = "b" * 32
    first = _run_id("campaign", first_execution)
    second = _run_id("campaign", second_execution)
    assert first != second
    assert first.endswith(first_execution)
    assert second.endswith(second_execution)


def test_pipeline_rejects_partial_grounding_provider_override(tmp_path: Path) -> None:
    brief = _write_brief(tmp_path / "brief.json")
    with pytest.raises(ValueError, match="requires transcription, alignment, and diarization"):
        run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
            source_client=FakeSource(tmp_path / "missing"),
            editorial_provider=FakeEditorial(),
            visual_scout_provider=FakeVision(),
            transcription_provider=FakeTranscription(),
            render=False,
        )


def test_grounding_failure_fails_closed_for_explicit_target(tmp_path: Path) -> None:
    brief = _write_brief(tmp_path / "brief.json")
    media = tmp_path / "source.mkv"
    media.write_bytes(b"source")
    with patch("clipper.pipeline._visual_timeline", side_effect=_fake_visual):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
            source_client=FakeSource(media),
            editorial_provider=FakeEditorial(),
            visual_scout_provider=FakeVision(),
            transcription_provider=FakeTranscription(empty=True),
            alignment_provider=FakeAlignment(),
            diarization_provider=FakeDiarization(),
            render=False,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["status_reason"] == "explicit_target_grounding_failed"
    assert manifest["errors"][0]["stage"] == "source_grounding"


def test_quality_graph_failure_fails_closed(tmp_path: Path) -> None:
    brief = _write_brief(tmp_path / "brief.json")
    media = tmp_path / "source.mkv"
    media.write_bytes(b"source")
    with (
        patch("clipper.pipeline._visual_timeline", side_effect=_fake_visual),
        patch("clipper.pipeline.plan_quality_batch", side_effect=RuntimeError("planner failed")),
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
            source_client=FakeSource(media),
            editorial_provider=FakeEditorial(),
            visual_scout_provider=FakeVision(),
            transcription_provider=FakeTranscription(),
            alignment_provider=FakeAlignment(),
            diarization_provider=FakeDiarization(),
            render=False,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["status_reason"] == "autonomous_quality_graph_failed"


def test_zero_quality_yield_is_success_not_quota_failure(tmp_path: Path) -> None:
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda _timelines, root: _empty_quality(root),
        render=True,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "SUCCESS"
    assert manifest["status_reason"] == "no_quality_moments"
    assert manifest["actual"]["eligible_quality_moments"] == 0
    assert manifest["rendered_clips"] == []


def test_full_quality_yield_requires_technical_and_multimodal_pass(tmp_path: Path) -> None:
    renderer = FakeRenderer()
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root),
        renderer=renderer,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "SUCCESS"
    assert manifest["status_reason"] is None
    assert manifest["publication_state"] == "READY_FOR_HUMAN_REVIEW"
    assert manifest["actual"]["rendered_finalists"] == 1
    assert manifest["funnel"]["technical_qc_pass"] == 1
    assert manifest["funnel"]["editorial_qc_pass"] == 1
    assert len(manifest["submission_shortlist"]) == 1
    assert list((run_dir / "captions").glob("*.ass"))
    assert list((run_dir / "tracking").glob("*.tracking.json"))
    review = json.loads((run_dir / "editorial-review.json").read_text())
    assert review["status"] == "PENDING_HUMAN_REVIEW"
    assert review["required"] is True


def test_technical_qc_rejection_reduces_quality_yield_without_replacement_quota(
    tmp_path: Path,
) -> None:
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root),
        qc={"status": "FAIL", "issues": ["bad caption"]},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["status_reason"] == "eligible_quality_moments_not_rendered"
    assert manifest["funnel"]["render_attempts"] == 1
    assert manifest["funnel"]["technical_qc_pass"] == 0
    assert manifest["submission_shortlist"] == []


def test_multimodal_review_rejection_reduces_quality_yield(tmp_path: Path) -> None:
    issue = VisualReviewIssue(
        "crop_oscillation",
        1.0,
        2.0,
        "HIGH",
        0.95,
        "TRACKING",
        "camera reverses",
    )
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root),
        review=_review_result("REPAIR", issues=(issue,)),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["funnel"]["editorial_review_reject_count"] == 1
    assert manifest["editorial_qc"][0]["decision"] == "REPAIR"


def test_partial_quality_yield_is_degraded_not_backfilled(tmp_path: Path) -> None:
    renderer = FakeRenderer(fail_calls={2})
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root, count=2),
        renderer=renderer,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "DEGRADED"
    assert manifest["status_reason"] == "partial_quality_yield"
    assert manifest["actual"]["eligible_quality_moments"] == 2
    assert manifest["actual"]["rendered_finalists"] == 1
    assert manifest["funnel"]["render_attempts"] == 1
    assert manifest["funnel"]["reserve_promotions"] == 0


def test_reserve_recovery_removes_primary_files_rejected_by_technical_qc(
    tmp_path: Path,
) -> None:
    calls = 0

    def qc(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "FAIL", "issues": ["synthetic rejection"]}
        return {"status": "PASS", "issues": [], "captions": {"alignment": "PASS"}}

    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root, reserve=True),
        renderer=FakeRenderer(),
        qc=qc,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "SUCCESS"
    assert manifest["funnel"]["reserve_promotions"] == 1
    mp4s = list((run_dir / "clips").glob("*.mp4"))
    assert len(mp4s) == 1
    assert "attempt-002" in mp4s[0].name
    assert not list((run_dir / "clips").glob("attempt-001*"))


def test_reserve_variant_can_recover_same_quality_moment_after_primary_failure(
    tmp_path: Path,
) -> None:
    renderer = FakeRenderer(fail_calls={1})
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root, reserve=True),
        renderer=renderer,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "SUCCESS"
    assert manifest["actual"]["eligible_quality_moments"] == 1
    assert manifest["actual"]["rendered_finalists"] == 1
    assert renderer.calls == 2
    assert manifest["funnel"]["reserve_promotions"] == 1


def test_missing_renderer_sidecar_fails_render_acceptance(tmp_path: Path) -> None:
    run_dir, *_ = _run_with_quality(
        tmp_path,
        lambda timelines, root: _quality_result(timelines, root),
        renderer=FakeRenderer(omit_sidecars=True),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert any("renderer omitted required evidence" in item["error"] for item in manifest["errors"])


def test_campaign_watermark_is_copied_before_render(tmp_path: Path) -> None:
    brief_path = _write_brief(
        tmp_path / "brief.json", watermark_url="https://example.test/watermark.png"
    )
    media = tmp_path / "source.mkv"
    media.write_bytes(b"source")
    watermark = tmp_path / "watermark.png"
    watermark.write_bytes(b"png")
    source = FakeSource(media, watermark=watermark)
    renderer = FakeRenderer()

    def quality_side_effect(brief, timelines, visual_timelines, editorial, *, dag_root):
        del brief, visual_timelines, editorial, dag_root
        return _quality_result(timelines, tmp_path)

    with (
        patch("clipper.pipeline._visual_timeline", side_effect=_fake_visual),
        patch("clipper.pipeline.plan_quality_batch", side_effect=quality_side_effect),
        patch("clipper.pipeline.run_technical_qc", return_value={"status": "PASS"}),
        patch("clipper.pipeline.review_rendered_clip", return_value=_review_result()),
    ):
        run_dir = run_pipeline(
            brief_path,
            settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
            source_client=source,
            renderer=renderer,
            editorial_provider=FakeEditorial(),
            visual_scout_provider=FakeVision(),
            visual_review_provider=FakeVision(),
            transcription_provider=FakeTranscription(),
            alignment_provider=FakeAlignment(),
            diarization_provider=FakeDiarization(),
        )
    expected = run_dir / "assets" / "campaign-watermark.png"
    assert renderer.watermarks == [expected]
    assert expected.read_bytes() == b"png"


def test_target_candidates_are_exact_and_channel_is_not_a_discovery_fallback(
    tmp_path: Path,
) -> None:
    brief_path = _write_brief(tmp_path / "brief.json")
    brief = load_brief(brief_path)
    candidates = _target_candidates(brief_path, brief)
    assert [(item.video_id, item.channel_id) for item in candidates] == [("v1", "UC1")]
    assert brief.source_channel_ids == []


def test_source_media_prefers_explicit_media_url(tmp_path: Path) -> None:
    brief_path = _write_brief(
        tmp_path / "brief.json", media_url="https://media.example.test/source.mkv"
    )
    brief = load_brief(brief_path)
    source = Mock()
    expected = tmp_path / "downloaded.source"
    expected.write_bytes(b"source")
    with patch("clipper.pipeline._download_asset", return_value=expected) as download:
        result = _source_media(
            brief,
            source,
            _target_candidates(brief_path, brief)[0],
            tmp_path,
        )
    assert result == expected
    source.download_media.assert_not_called()
    assert download.call_args.kwargs["expected_kind"] == "media"


def test_campaign_asset_url_normalization_and_validation() -> None:
    normalized = _normalize_asset_url("https://drive.google.com/file/d/abc123/view?usp=sharing")
    assert normalized.startswith("https://drive.usercontent.google.com/download?")
    assert "id=abc123" in normalized
    assert "export=download" in normalized
    query_style = _normalize_asset_url("https://drive.google.com/open?id=xyz789")
    assert "id=xyz789" in query_style
    assert _normalize_asset_url("https://example.com/watermark.png") == (
        "https://example.com/watermark.png"
    )
    with pytest.raises(ValueError, match="must use https"):
        _normalize_asset_url("http://example.com/watermark.png")


def _response(content_type: str, chunks: list[bytes]) -> Mock:
    body = Mock()
    body.headers.get_content_type.return_value = content_type
    body.read.side_effect = chunks
    context = Mock()
    context.__enter__ = Mock(return_value=body)
    context.__exit__ = Mock(return_value=False)
    return context


def test_download_asset_accepts_images_and_rejects_bad_payloads(tmp_path: Path) -> None:
    output = tmp_path / "watermark.png"
    with patch("clipper.pipeline.urlopen", return_value=_response("image/png", [b"png", b""])):
        assert _download_asset("https://example.com/watermark.png", output) == output
    assert output.read_bytes() == b"png"

    with (
        patch("clipper.pipeline.urlopen", return_value=_response("text/html", [b"bad", b""])),
        pytest.raises(RuntimeError, match="not an image"),
    ):
        _download_asset("https://example.com/bad", tmp_path / "bad.png")

    with (
        patch("clipper.pipeline.urlopen", return_value=_response("image/png", [b"123", b""])),
        pytest.raises(RuntimeError, match="exceeds"),
    ):
        _download_asset("https://example.com/large", tmp_path / "large.png", max_bytes=2)


def test_download_asset_accepts_binary_media_and_drive_path(tmp_path: Path) -> None:
    output = tmp_path / "source.mkv"
    with patch(
        "clipper.pipeline.urlopen",
        return_value=_response("application/octet-stream", [b"media", b""]),
    ):
        assert (
            _download_asset("https://example.com/source.mkv", output, expected_kind="media")
            == output
        )
    assert output.read_bytes() == b"media"

    drive = tmp_path / "drive.mkv"

    def fake_drive(*, url: str, output: str, quiet: bool):
        assert url.startswith("https://drive.google.com/")
        assert quiet is True
        Path(output).write_bytes(b"drive")
        return output

    with patch("clipper.pipeline.gdown.download", side_effect=fake_drive):
        assert (
            _download_asset(
                "https://drive.google.com/file/d/source/view", drive, expected_kind="media"
            )
            == drive
        )
    assert drive.read_bytes() == b"drive"


def test_source_media_metadata_is_recorded_only_from_valid_sidecar(tmp_path: Path) -> None:
    media = tmp_path / "source.mkv"
    media.write_bytes(b"source")
    manifest = PipelineManifest("campaign")
    _record_source_media_metadata(manifest, "v1", media)
    assert "source_media" not in manifest.run_metadata
    sidecar = media.with_suffix(".source.json")
    sidecar.write_text(json.dumps({"selected": {"height": 2160}}), encoding="utf-8")
    _record_source_media_metadata(manifest, "v1", media)
    assert manifest.run_metadata["source_media"]["v1"]["selected"]["height"] == 2160
    sidecar.write_text("bad-json", encoding="utf-8")
    before = dict(manifest.run_metadata["source_media"])
    _record_source_media_metadata(manifest, "v2", media)
    assert manifest.run_metadata["source_media"] == before


def test_speaker_focus_comes_from_override_or_source_modality() -> None:
    quality = SimpleNamespace(
        source_evidence={"v1": {"modality_profile": {"requires_speaker_identity": True}}}
    )
    assert _speaker_focus_for_source(PipelineSettings(), quality, "v1") is True
    assert (
        _speaker_focus_for_source(PipelineSettings(speaker_focus_override=False), quality, "v1")
        is False
    )
    with patch("clipper.pipeline.FFmpegRenderer") as renderer_cls:
        _renderer_for_source(PipelineSettings(), quality, "v1")
    assert renderer_cls.call_args.kwargs["speaker_focus"] is True


def test_visual_timeline_requires_grounding_and_records_scout_result(tmp_path: Path) -> None:
    media = tmp_path / "source.mkv"
    media.write_bytes(b"source")
    video = VideoCandidate("v1", "T", "UC1", "C", "https://youtube.test/v1", duration_seconds=30)
    empty = CanonicalTimeline("v1", "hash", ())
    with pytest.raises(RuntimeError, match="no source words"):
        _visual_timeline(media, video, empty, FakeVision(), tmp_path)

    timeline = CanonicalTimeline("v1", "hash", _words("v1", 30))
    visual = VisualTimeline("v1", "hash", (VisualEvent(0, 30, "scene", "source", (), (), 0.9),))
    result = ProviderResult({"events": []}, FakeVision.identity, _usage())
    assert not (tmp_path / "visual-scout").exists()
    with patch("clipper.pipeline.scout_visual_timeline", return_value=(visual, result)):
        observed, meta = _visual_timeline(media, video, timeline, FakeVision(), tmp_path)
    assert observed == visual
    assert meta["model"]["model_id"] == "fake-vlm"
    assert (tmp_path / "visual-scout" / "v1.json").is_file()


def test_required_render_evidence_and_tracking_helpers_fail_closed(tmp_path: Path) -> None:
    rendered = tmp_path / "clip.mp4"
    rendered.write_bytes(b"clip")
    plan = SimpleNamespace(plan_id="plan")
    with pytest.raises(RuntimeError, match="renderer omitted required evidence"):
        _copy_render_sidecars(rendered, tmp_path / "run", plan)
    assert _tracking_transitions(rendered) == ()
    rendered.with_suffix(".tracking.json").write_text("bad", encoding="utf-8")
    assert _tracking_transitions(rendered) == ()


def test_rendered_clip_records_exact_output_hash(tmp_path: Path) -> None:
    brief_path = _write_brief(tmp_path / "brief.json")
    brief = load_brief(brief_path)
    timeline = CanonicalTimeline("v1", "hash", _words("v1", 60))
    quality = _quality_result({"v1": timeline}, tmp_path)
    plan = quality.plans[0]
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"final")
    video = _target_candidates(brief_path, brief)[0]
    payload = _rendered_clip(plan, output, video)
    assert payload["render_sha256"]
    assert payload["plan_id"] == plan.plan_id


def test_campaign_watermark_no_policy_returns_none(tmp_path: Path) -> None:
    brief = load_brief(_write_brief(tmp_path / "brief.json"))
    assert _campaign_watermark(brief, Mock(), tmp_path) is None
