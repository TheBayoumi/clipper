from __future__ import annotations

import re
from pathlib import Path


def replace_section(path_name: str, start: str, end: str, replacement: str) -> None:
    path = Path(path_name)
    text = path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"section anchors failed for {path_name}: {start!r} -> {end!r}")
    begin = text.index(start)
    finish = text.index(end, begin)
    path.write_text(text[:begin] + replacement.rstrip() + "\n\n" + text[finish:], encoding="utf-8")


def replace_function(path_name: str, function_name: str, replacement: str) -> None:
    path = Path(path_name)
    text = path.read_text(encoding="utf-8")
    pattern = rf"^def {re.escape(function_name)}\([^\n]*\).*?(?=^def |^class |\Z)"
    updated, count = re.subn(pattern, replacement.rstrip() + "\n\n", text, count=1, flags=re.M | re.S)
    if count != 1:
        raise SystemExit(f"function anchor failed for {path_name}:{function_name}")
    path.write_text(updated, encoding="utf-8")


def migrate_open_model_legacy_planner_tests() -> None:
    replace_section(
        "tests/test_open_models.py",
        "class _PlannerEditorial:\n",
        "def test_managed_modal_endpoint_editorial_provider_uses_proxy_auth_and_json() -> None:\n",
        '''def test_autonomous_editor_bridge_only_validates_progress_configuration(tmp_path: Path) -> None:
    planner = AutonomousEditorialPlanner(
        Mock(),
        Mock(),
        FileCache(tmp_path / "bridge"),
        max_words_per_chunk=900,
        chunk_overlap_words=120,
    )
    assert planner.max_words_per_chunk == 900
    assert planner.chunk_overlap_words == 120
    for removed in (
        "analyze_video",
        "plan_batch",
        "_dedupe_hooks",
        "_plan_context_words",
        "_classify_source_hazards",
    ):
        assert not hasattr(planner, removed)

    with pytest.raises(ValueError, match="at least 200"):
        AutonomousEditorialPlanner(Mock(), Mock(), FileCache(tmp_path / "small"), max_words_per_chunk=199)
    with pytest.raises(ValueError, match="smaller than chunk size"):
        AutonomousEditorialPlanner(
            Mock(),
            Mock(),
            FileCache(tmp_path / "overlap"),
            max_words_per_chunk=200,
            chunk_overlap_words=200,
        )''',
    )

    replace_function(
        "tests/test_open_models.py",
        "test_editorial_prompt_contracts_cover_all_grounded_tasks",
        '''def test_editorial_prompt_contracts_cover_active_adaptive_tasks() -> None:
    tasks = (
        "source_hazards:0",
        "semantic_cores:0",
        "narrative_envelope:core-1",
        "quality_windows:core-1",
    )
    assert editorial_output_budget({"task": "source_hazards:0"}) == 2048
    assert editorial_output_budget({"task": "semantic_cores:0"}) == 2048
    assert editorial_output_budget({"task": "narrative_envelope:core-1"}) == 1536
    assert editorial_output_budget({"task": "quality_windows:core-1"}) == 1536
    for task in tasks:
        contract = editorial_contract(task)
        assert "never manufacture moments to satisfy a count" in contract
        assert "predeclared vocabulary" in contract
    for legacy in (
        "episode_editorial_profile",
        "story_moments:0",
        "clip_concepts",
        "global_concept_comparison",
        "hook_variants:c1",
        "edit_plans:c1",
        "boundary_audit:p1",
    ):
        with pytest.raises(ValueError, match="unsupported production editorial task"):
            editorial_contract(legacy)
    modal_source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "from clipper.providers.editorial_prompt import editorial_contract" in modal_source''',
    )

    replace_section(
        "tests/test_open_models.py",
        "def test_plan_batch_reports_duration_rejections_when_all_model_plans_are_invalid(\n",
        "def test_visual_timeline_roundtrip_and_payload_validation() -> None:\n",
        '''def test_removed_batch_planner_methods_stay_absent(tmp_path: Path) -> None:
    planner = AutonomousEditorialPlanner(Mock(), Mock(), FileCache(tmp_path / "removed-batch"))
    assert not hasattr(planner, "plan_batch")
    assert not hasattr(planner, "analyze_video")''',
    )


def migrate_phase_b_editorial_contract_test() -> None:
    replace_function(
        "tests/test_phase_b_contract_edges.py",
        "test_editorial_prompt_exposes_every_structured_task_family_and_budget",
        '''def test_editorial_prompt_exposes_active_structured_task_families_and_budget() -> None:
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
    for unsupported in ("episode_editorial_profile", "story_moments:0", "edit_plans:c", "unsupported"):
        with pytest.raises(ValueError, match="unsupported production editorial task"):
            editorial_task_family(unsupported)
        with pytest.raises(ValueError, match="unsupported production editorial task"):
            editorial_json_schema(unsupported)''',
    )


def rewrite_pipeline_tests() -> None:
    Path("tests/test_pipeline.py").write_text(
        '''import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.models import CampaignBrief, PipelineManifest, VideoCandidate
from clipper.pipeline import (
    PipelineSettings,
    _campaign_watermark,
    _download_asset,
    _funnel_template,
    _normalize_asset_url,
    _record_source_media_metadata,
    _source_media,
    _target_candidates,
    run_pipeline,
)
from clipper.providers.base import ModelIdentity
from clipper.quality_batch import QualityBatchResult
from clipper.visual import VisualEvent, VisualTimeline


def _brief_path(tmp_path: Path, *, media_url: str | None = None) -> Path:
    target = {
        "video_id": "v1",
        "url": "https://www.youtube.com/watch?v=v1",
        "channel_id": "UC1",
    }
    if media_url is not None:
        target["media_url"] = media_url
    path = tmp_path / "brief.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign",
                "title": "Exact target",
                "objective": "Find worthwhile source-grounded moments",
                "targets": {"mode": "explicit", "videos": [target]},
                "rights": {"confirmed": True, "authorized_channels": ["UC1"]},
                "min_clip_seconds": 8,
                "max_clip_seconds": 45,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_target_candidates_are_exact_and_never_use_discovery(tmp_path: Path) -> None:
    path = _brief_path(tmp_path)
    brief = load_brief(path)
    candidates = _target_candidates(path, brief)
    assert [item.video_id for item in candidates] == ["v1"]
    assert candidates[0].channel_id == "UC1"
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"


def test_pipeline_settings_parse_quality_chunk_controls(monkeypatch) -> None:
    monkeypatch.setenv("CLIPPER_QUALITY_CHUNK_WORDS", "1200")
    monkeypatch.setenv("CLIPPER_QUALITY_CHUNK_OVERLAP_WORDS", "180")
    settings = PipelineSettings.from_env()
    assert settings.quality_chunk_words == 1200
    assert settings.quality_chunk_overlap_words == 180
    assert settings.compute_profile == "balanced"


def test_source_media_prefers_authorized_direct_media_url(tmp_path: Path) -> None:
    path = _brief_path(tmp_path, media_url="https://example.com/master.mp4")
    brief = load_brief(path)
    video = _target_candidates(path, brief)[0]
    expected = tmp_path / "work" / "v1.source"
    source = Mock()
    with patch("clipper.pipeline._download_asset", return_value=expected) as download:
        assert _source_media(brief, source, video, tmp_path / "work") == expected
    download.assert_called_once_with(
        "https://example.com/master.mp4",
        expected,
        max_bytes=10_000_000_000,
        expected_kind="media",
    )
    source.download_media.assert_not_called()


def test_source_media_falls_back_to_source_client(tmp_path: Path) -> None:
    path = _brief_path(tmp_path)
    brief = load_brief(path)
    video = _target_candidates(path, brief)[0]
    source = Mock()
    source.download_media.return_value = tmp_path / "master.mp4"
    assert _source_media(brief, source, video, tmp_path / "work") == tmp_path / "master.mp4"
    source.download_media.assert_called_once_with(video, tmp_path / "work")


def test_campaign_watermark_prefers_fixture_asset_then_download(tmp_path: Path) -> None:
    supplied = tmp_path / "fixture.png"
    supplied.write_bytes(b"fixture")
    brief = CampaignBrief(
        campaign_id="c",
        title="t",
        objective="o",
        allowed_video_ids=["v1"],
        rights_confirmed=True,
        watermark_url="https://example.com/watermark.png",
    )
    source = SimpleNamespace(campaign_watermark=lambda _brief: supplied)
    output = _campaign_watermark(brief, source, tmp_path / "run")
    assert output is not None and output.read_bytes() == b"fixture"

    plain_source = object()
    downloaded = tmp_path / "downloaded.png"
    with patch("clipper.pipeline._download_asset", return_value=downloaded) as fetch:
        assert _campaign_watermark(brief, plain_source, tmp_path / "run2") == downloaded
    fetch.assert_called_once()
    assert _campaign_watermark(
        CampaignBrief("c", "t", "o", allowed_video_ids=["v1"], rights_confirmed=True),
        plain_source,
        tmp_path / "run3",
    ) is None


def test_asset_normalization_and_download_validation(tmp_path: Path) -> None:
    assert _normalize_asset_url("https://drive.google.com/file/d/abc/view") == (
        "https://drive.google.com/uc?export=download&id=abc"
    )
    assert _normalize_asset_url("https://example.com/a.png") == "https://example.com/a.png"
    with pytest.raises(ValueError, match="HTTPS"):
        _normalize_asset_url("http://example.com/a.png")

    body = Mock()
    body.headers.get_content_type.return_value = "image/png"
    body.read.side_effect = [b"png", b""]
    context = Mock()
    context.__enter__ = Mock(return_value=body)
    context.__exit__ = Mock(return_value=False)
    output = tmp_path / "asset.png"
    with patch("clipper.pipeline.urlopen", return_value=context):
        assert _download_asset("https://example.com/a.png", output) == output
    assert output.read_bytes() == b"png"

    bad = Mock()
    bad.headers.get_content_type.return_value = "text/html"
    bad.read.side_effect = [b"html", b""]
    bad_context = Mock()
    bad_context.__enter__ = Mock(return_value=bad)
    bad_context.__exit__ = Mock(return_value=False)
    with (
        patch("clipper.pipeline.urlopen", return_value=bad_context),
        pytest.raises(RuntimeError, match="not an image"),
    ):
        _download_asset("https://example.com/a.png", tmp_path / "bad.png")


def test_source_metadata_recording_is_fail_safe(tmp_path: Path) -> None:
    media = tmp_path / "v1.mp4"
    media.write_bytes(b"media")
    manifest = PipelineManifest("c")
    _record_source_media_metadata(manifest, "v1", media)
    assert "source_media" not in manifest.run_metadata
    media.with_suffix(".source.json").write_text('{"height": 2160}', encoding="utf-8")
    _record_source_media_metadata(manifest, "v1", media)
    assert manifest.run_metadata["source_media"]["v1"]["height"] == 2160
    media.with_suffix(".source.json").write_text("bad", encoding="utf-8")
    _record_source_media_metadata(manifest, "v2", media)
    assert "v2" not in manifest.run_metadata["source_media"]


def _timeline() -> CanonicalTimeline:
    return CanonicalTimeline(
        "v1",
        hashlib.sha256(b"media").hexdigest(),
        (
            CanonicalWord("w1", "useful", 0.0, 0.4, "A", 0.99, "aligned", "test"),
            CanonicalWord("w2", "idea", 0.5, 0.9, "A", 0.99, "aligned", "test"),
        ),
    )


def _empty_quality(tmp_path: Path) -> QualityBatchResult:
    return QualityBatchResult(
        story_moments=(),
        concepts=(),
        variants=(),
        plans=(),
        quality_moments=(),
        rejections=(),
        model_invocations=(),
        boundary_audits=(),
        campaign_policy_audits=(),
        source_hazards=(),
        source_evidence={"v1": {"semantic_cores": 0, "modality_profile": {}}},
        stage_cache_hits=0,
        stage_executions=1,
        stage_dag_root=tmp_path / "dag",
    )


class _Source:
    def __init__(self, media: Path) -> None:
        self.media = media

    def download_media(self, _video: VideoCandidate, _work_dir: Path) -> Path:
        return self.media


class _Provider:
    identity = ModelIdentity("test", "rev", "none", "test", "editor", "schema")


def test_run_pipeline_accepts_zero_quality_yield_and_records_planning_state(tmp_path: Path) -> None:
    path = _brief_path(tmp_path)
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    timeline = _timeline()
    visual = VisualTimeline(
        "v1",
        timeline.source_hash,
        (VisualEvent(0.0, 1.0, "s1", "speaker", ("A",), ("talking",), 0.9),),
    )
    with (
        patch("clipper.pipeline._cached_transcription", return_value=(timeline, {"stage": "t"})),
        patch("clipper.pipeline._cached_alignment", return_value=(timeline, {"stage": "a"})),
        patch("clipper.pipeline._cached_diarization", return_value=(timeline, {"stage": "d"})),
        patch("clipper.pipeline._visual_timeline", return_value=(visual, {"stage": "v"})),
        patch("clipper.pipeline.plan_quality_batch", return_value=_empty_quality(tmp_path)),
    ):
        run_dir = run_pipeline(
            path,
            settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
            source_client=_Source(media),
            editorial_provider=_Provider(),
            visual_scout_provider=_Provider(),
            transcription_provider=object(),
            alignment_provider=object(),
            diarization_provider=object(),
            render=False,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS"
    assert manifest["status_reason"] == "planning_complete"
    assert manifest["targets"]["eligible_quality_moments"] == 0
    assert manifest["actual"]["rendered_finalists"] == 0
    assert manifest["run_metadata"]["architecture"] == "autonomous-multimodal-quality-graph"


def test_run_pipeline_fails_closed_when_exact_target_grounding_fails(tmp_path: Path) -> None:
    path = _brief_path(tmp_path)
    source = Mock()
    source.download_media.side_effect = RuntimeError("source unavailable")
    run_dir = run_pipeline(
        path,
        settings=PipelineSettings(artifact_root=tmp_path / "failed"),
        source_client=source,
        editorial_provider=_Provider(),
        visual_scout_provider=_Provider(),
        transcription_provider=object(),
        alignment_provider=object(),
        diarization_provider=object(),
        render=False,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["status_reason"] == "explicit_target_grounding_failed"
    assert manifest["errors"][0]["video_id"] == "v1"


def test_funnel_template_is_quality_yield_based() -> None:
    funnel = _funnel_template()
    assert funnel["quality_moments"] == 0
    assert funnel["render_plans"] == 0
    assert "candidate_pool_size" not in funnel
    assert "final_render_budget" not in funnel
''',
        encoding="utf-8",
    )


def main() -> None:
    migrate_open_model_legacy_planner_tests()
    migrate_phase_b_editorial_contract_test()
    rewrite_pipeline_tests()


if __name__ == "__main__":
    main()
