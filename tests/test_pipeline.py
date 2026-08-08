import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.models import (
    CampaignBrief,
    ClipCandidate,
    ClipConcept,
    EditorialScores,
    PipelineManifest,
    TranscriptSegment,
    VideoCandidate,
)
from clipper.pipeline import (
    PipelineSettings,
    _campaign_media_candidates,
    _download_asset,
    _normalize_asset_url,
    _record_source_media_metadata,
    run_pipeline,
)
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.visual import VisualEvent, VisualTimeline
from clipper.visual_ai import VisualReviewIssue, VisualReviewReport


@pytest.fixture(autouse=True)
def _pipeline_qc_pass():
    with patch(
        "clipper.pipeline.run_technical_qc",
        return_value={"status": "PASS", "issues": [], "captions": {"alignment": "PASS"}},
    ):
        yield


class FakeSource:
    def __init__(self, subtitle: Path, media: Path) -> None:
        self.subtitle = subtitle
        self.media = media

    def discover(self, _brief: CampaignBrief) -> list[VideoCandidate]:
        return [
            VideoCandidate("allowed", "Good", "UC1", "Channel", "https://youtu.be/allowed"),
            VideoCandidate("blocked", "Bad", "UC2", "Other", "https://youtu.be/blocked"),
        ]

    def download_subtitles(self, _video: VideoCandidate, _work_dir: Path, _language: str) -> Path:
        return self.subtitle

    def download_media(self, _video: VideoCandidate, _work_dir: Path) -> Path:
        return self.media


class FakeRenderer:
    def __init__(self) -> None:
        self.watermark_path: Path | None = None

    def render(
        self,
        _source_path: Path,
        output_path: Path,
        _clip: ClipCandidate,
        _segments: list[TranscriptSegment],
        watermark_path: Path | None = None,
        edit_plan: object | None = None,
    ) -> Path:
        self.watermark_path = watermark_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        output_path.with_suffix(".tracking-preflight.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "initial_issues": [],
                    "repaired_with_stable_fallback": False,
                    "final_issues": [],
                    "final_framing_mode": "speaker_locked_portrait",
                }
            )
        )
        return output_path


def test_pipeline_writes_manifest_and_filters_sources(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    brief.write_text(
        json.dumps(
            {
                "campaign_id": "campaign",
                "title": "Automation",
                "objective": "Explain AI",
                "keywords": ["automation", "business"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "min_clip_seconds": 8,
                "max_clip_seconds": 20,
                "clip_count": 1,
            }
        ),
        encoding="utf-8",
    )
    subtitle = tmp_path / "captions.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\nAutomation can save time for a business.\n",
        encoding="utf-8",
    )
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source")

    run_dir = run_pipeline(
        brief,
        settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
        source_client=FakeSource(subtitle, media),
        renderer=FakeRenderer(),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["discovered_videos"]) == 1
    assert len(manifest["planned_clips"]) == 1
    assert len(manifest["rendered_clips"]) == 1
    assert manifest["errors"][0]["video_id"] == "blocked"


class NoSubtitleSource(FakeSource):
    def download_subtitles(self, _video: VideoCandidate, _work_dir: Path, _language: str):
        return None


class BrokenSubtitleSource(FakeSource):
    def download_subtitles(self, _video: VideoCandidate, _work_dir: Path, _language: str):
        raise RuntimeError("caption failure")


class BrokenRenderer(FakeRenderer):
    def render(self, *_args, **_kwargs):
        raise RuntimeError("render failure")


def _write_pipeline_brief(tmp_path: Path) -> Path:
    brief = tmp_path / "brief-extra.json"
    brief.write_text(
        json.dumps(
            {
                "campaign_id": "extra",
                "title": "Automation",
                "objective": "Explain AI",
                "keywords": ["automation"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "min_clip_seconds": 8,
                "max_clip_seconds": 20,
                "clip_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return brief


def test_pipeline_asr_no_render_and_environment(tmp_path: Path, monkeypatch) -> None:
    brief = _write_pipeline_brief(tmp_path)
    subtitle = tmp_path / "unused.vtt"
    media = tmp_path / "source-extra.mp4"
    media.write_bytes(b"source")
    monkeypatch.setenv("CLIPPER_ARTIFACT_ROOT", str(tmp_path / "env-artifacts"))
    monkeypatch.setenv("CLIPPER_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("CLIPPER_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("CLIPPER_WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("CLIPPER_SPEAKER_FOCUS", "false")
    monkeypatch.setenv("CLIPPER_SPEAKER_ZOOM", "1.18")
    monkeypatch.setenv("CLIPPER_SPEAKER_SAMPLE_FPS", "6")
    monkeypatch.setenv("CLIPPER_SPEAKER_SWITCH_MARGIN", "1.5")
    monkeypatch.setenv("CLIPPER_SOURCE_MAX_HEIGHT", "1440")
    monkeypatch.setenv("CLIPPER_RENDER_PROFILE", "review")
    monkeypatch.setenv("CLIPPER_SPEAKER_MIN_REFRAME_SECONDS", "0.3")
    monkeypatch.setenv("CLIPPER_SPEAKER_MAX_REFRAME_SECONDS", "0.8")
    monkeypatch.setenv("CLIPPER_SPEAKER_SECONDS_PER_CROP", "0.7")
    monkeypatch.setenv("CLIPPER_SPEAKER_HOLD_THRESHOLD", "0.25")
    monkeypatch.setenv("CLIPPER_SPEAKER_REVERSAL_GUARD_SECONDS", "1.4")
    monkeypatch.setenv("CLIPPER_SPEAKER_WINDOW_SECONDS", "0.9")
    monkeypatch.setenv("CLIPPER_SPEAKER_MIN_DETECTION_COVERAGE", "0.4")

    settings = PipelineSettings.from_env()
    assert settings.whisper_model == "tiny"
    assert settings.speaker_focus is False
    assert settings.speaker_zoom == 1.18
    assert settings.speaker_sample_fps == 6.0
    assert settings.speaker_switch_margin == 1.5
    assert settings.source_max_height == 1440
    assert settings.render_profile == "review"
    assert settings.speaker_min_reframe_seconds == 0.3
    assert settings.speaker_max_reframe_seconds == 0.8
    assert settings.speaker_seconds_per_crop == 0.7
    assert settings.speaker_hold_threshold == 0.25
    assert settings.speaker_reversal_guard_seconds == 1.4
    assert settings.speaker_window_seconds == 0.9
    assert settings.speaker_min_detection_coverage == 0.4
    with patch(
        "clipper.pipeline.transcribe_with_faster_whisper",
        return_value=[TranscriptSegment(0, 9, "automation saves time.")],
    ) as transcribe:
        run_dir = run_pipeline(
            brief,
            settings=settings,
            source_client=NoSubtitleSource(subtitle, media),
            render=False,
        )
    assert transcribe.called
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["planned_clips"]) == 1
    assert manifest["rendered_clips"] == []


def test_pipeline_records_processing_and_render_errors(tmp_path: Path) -> None:
    brief = _write_pipeline_brief(tmp_path)
    subtitle = tmp_path / "captions-extra.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\nautomation works.\n",
        encoding="utf-8",
    )
    media = tmp_path / "source-extra.mp4"
    media.write_bytes(b"source")

    broken_source_dir = run_pipeline(
        brief,
        settings=PipelineSettings(artifact_root=tmp_path / "broken-source"),
        source_client=BrokenSubtitleSource(subtitle, media),
        renderer=FakeRenderer(),
    )
    source_manifest = json.loads((broken_source_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any("caption failure" in item["error"] for item in source_manifest["errors"])

    broken_render_dir = run_pipeline(
        brief,
        settings=PipelineSettings(artifact_root=tmp_path / "broken-render"),
        source_client=FakeSource(subtitle, media),
        renderer=BrokenRenderer(),
    )
    render_manifest = json.loads((broken_render_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any("render failure" in item["error"] for item in render_manifest["errors"])


def test_pipeline_records_empty_transcript(tmp_path: Path) -> None:
    brief = _write_pipeline_brief(tmp_path)
    subtitle = tmp_path / "empty.vtt"
    subtitle.write_text("WEBVTT\n", encoding="utf-8")
    media = tmp_path / "source-empty.mp4"
    media.write_bytes(b"source")
    run_dir = run_pipeline(
        brief,
        settings=PipelineSettings(artifact_root=tmp_path / "empty-run"),
        source_client=FakeSource(subtitle, media),
        renderer=FakeRenderer(),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any("no timestamped segments" in item["error"] for item in manifest["errors"])


def test_pipeline_downloads_required_watermark(tmp_path: Path) -> None:
    brief = _write_pipeline_brief(tmp_path)
    payload = json.loads(brief.read_text(encoding="utf-8"))
    payload["watermark_url"] = "https://drive.google.com/file/d/example/view"
    payload["required_hashtags"] = ["#DoubleCoverage"]
    payload["posting_requirements"] = ["Use a dedicated Double Coverage account"]
    brief.write_text(json.dumps(payload), encoding="utf-8")
    subtitle = tmp_path / "captions-watermark.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\nautomation works.\n",
        encoding="utf-8",
    )
    media = tmp_path / "source-watermark.mp4"
    media.write_bytes(b"source")
    renderer = FakeRenderer()

    def fake_download(_url: str, output_path: Path, **_kwargs) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path

    with patch("clipper.pipeline._download_asset", side_effect=fake_download):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "watermark-run"),
            source_client=FakeSource(subtitle, media),
            renderer=renderer,
        )
    assert renderer.watermark_path == run_dir / "assets" / "watermark.png"
    normalized = json.loads((run_dir / "brief.normalized.json").read_text(encoding="utf-8"))
    assert normalized["required_hashtags"] == ["#DoubleCoverage"]
    assert normalized["posting_requirements"]


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


def test_download_asset_accepts_images_and_rejects_bad_payloads(tmp_path: Path) -> None:
    def response(content_type: str, chunks: list[bytes]) -> Mock:
        body = Mock()
        body.headers.get_content_type.return_value = content_type
        body.read.side_effect = chunks
        context = Mock()
        context.__enter__ = Mock(return_value=body)
        context.__exit__ = Mock(return_value=False)
        return context

    output = tmp_path / "watermark.png"
    with patch(
        "clipper.pipeline.urlopen",
        return_value=response("image/png", [b"png-data", b""]),
    ):
        assert _download_asset("https://example.com/watermark.png", output) == output
    assert output.read_bytes() == b"png-data"

    with (
        patch(
            "clipper.pipeline.urlopen",
            return_value=response("text/html", [b"not-an-image", b""]),
        ),
        pytest.raises(RuntimeError, match="not an image"),
    ):
        _download_asset("https://example.com/bad", tmp_path / "bad.png")

    with (
        patch(
            "clipper.pipeline.urlopen",
            return_value=response("image/png", [b"123", b""]),
        ),
        pytest.raises(RuntimeError, match="exceeds"),
    ):
        _download_asset(
            "https://example.com/large.png",
            tmp_path / "large.png",
            max_bytes=2,
        )


def test_campaign_media_candidates_and_direct_media_bypass_youtube(tmp_path: Path) -> None:
    brief_path = _write_pipeline_brief(tmp_path)
    payload = json.loads(brief_path.read_text(encoding="utf-8"))
    payload["allowed_video_ids"] = ["allowed"]
    payload["source_media_urls"] = {
        "allowed": "https://drive.google.com/file/d/campaign-media/view"
    }
    brief_path.write_text(json.dumps(payload), encoding="utf-8")
    brief = CampaignBrief.from_dict(payload)
    candidates = _campaign_media_candidates(brief)
    assert [item.video_id for item in candidates] == ["allowed"]
    assert candidates[0].channel_id == "UC1"

    class DirectSource(FakeSource):
        def discover(self, _brief: CampaignBrief) -> list[VideoCandidate]:
            raise AssertionError("direct campaign media must bypass YouTube discovery")

        def download_subtitles(self, *_args, **_kwargs):
            raise AssertionError("direct campaign media must bypass YouTube subtitles")

        def download_media(self, *_args, **_kwargs):
            raise AssertionError("direct campaign media must bypass YouTube media download")

    subtitle = tmp_path / "unused-direct.vtt"
    media = tmp_path / "unused-direct.mp4"
    downloaded = tmp_path / "downloaded.mp4"
    downloaded.write_bytes(b"campaign-media")

    def fake_download(_url: str, output_path: Path, **kwargs) -> Path:
        assert kwargs["expected_kind"] == "media"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(downloaded.read_bytes())
        return output_path

    with (
        patch("clipper.pipeline._download_asset", side_effect=fake_download),
        patch(
            "clipper.pipeline.transcribe_with_faster_whisper",
            return_value=[TranscriptSegment(0, 9, "automation saves time.")],
        ) as transcribe,
    ):
        run_dir = run_pipeline(
            brief_path,
            settings=PipelineSettings(artifact_root=tmp_path / "direct-run"),
            source_client=DirectSource(subtitle, media),
            renderer=FakeRenderer(),
        )
    assert transcribe.call_args.args[0] == run_dir / "work" / "allowed" / "source.mp4"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["errors"] == []
    assert len(manifest["rendered_clips"]) == 1


def test_download_asset_accepts_binary_media(tmp_path: Path) -> None:
    body = Mock()
    body.headers.get_content_type.return_value = "application/octet-stream"
    body.read.side_effect = [b"media", b""]
    context = Mock()
    context.__enter__ = Mock(return_value=body)
    context.__exit__ = Mock(return_value=False)
    output = tmp_path / "source.mp4"
    with patch("clipper.pipeline.urlopen", return_value=context):
        assert (
            _download_asset(
                "https://example.com/source.mp4",
                output,
                expected_kind="media",
            )
            == output
        )
    assert output.read_bytes() == b"media"

    html_body = Mock()
    html_body.headers.get_content_type.return_value = "text/html"
    html_body.read.side_effect = [b"login", b""]
    html_context = Mock()
    html_context.__enter__ = Mock(return_value=html_body)
    html_context.__exit__ = Mock(return_value=False)
    with (
        patch("clipper.pipeline.urlopen", return_value=html_context),
        pytest.raises(RuntimeError, match="not binary media"),
    ):
        _download_asset(
            "https://example.com/source.mp4",
            tmp_path / "bad-media.mp4",
            expected_kind="media",
        )


def test_download_asset_uses_gdown_for_google_drive_media(tmp_path: Path) -> None:
    output = tmp_path / "drive-source.mp4"

    def fake_download(*, url: str, output: str, quiet: bool) -> str:
        assert url.startswith("https://drive.google.com/file/d/")
        assert quiet is True
        Path(output).write_bytes(b"drive-media")
        return output

    with patch("clipper.pipeline.gdown.download", side_effect=fake_download) as download:
        assert (
            _download_asset(
                "https://drive.google.com/file/d/source-id/view",
                output,
                expected_kind="media",
            )
            == output
        )
    assert download.call_count == 1
    assert output.read_bytes() == b"drive-media"

    def oversized_download(*, url: str, output: str, quiet: bool) -> str:
        del url, quiet
        Path(output).write_bytes(b"123")
        return output

    with (
        patch("clipper.pipeline.gdown.download", side_effect=oversized_download),
        pytest.raises(RuntimeError, match="exceeds"),
    ):
        _download_asset(
            "https://drive.google.com/file/d/source-id/view",
            tmp_path / "too-large.mp4",
            max_bytes=2,
            expected_kind="media",
        )


def test_pipeline_reuses_transcript_and_editorial_cache(tmp_path: Path) -> None:
    brief_path = _write_pipeline_brief(tmp_path)
    subtitle = tmp_path / "cache.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\nautomation saves creator time.\n",
        encoding="utf-8",
    )
    media = tmp_path / "cache.mp4"
    media.write_bytes(b"source")
    settings = PipelineSettings(
        artifact_root=tmp_path / "runs", cache_root=tmp_path / "persistent-cache"
    )
    with patch("clipper.pipeline._run_id", side_effect=["first", "second"]):
        run_pipeline(
            brief_path, settings=settings, source_client=FakeSource(subtitle, media), render=False
        )
        second = run_pipeline(
            brief_path, settings=settings, source_client=FakeSource(subtitle, media), render=False
        )
    manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache"]["hits"] >= 2
    assert manifest["performance"]["wall_seconds"] >= 0
    assert manifest["run_metadata"]["git_commit_sha"]
    assert manifest["run_metadata"]["transcript_hashes"]["allowed"]


def test_pipeline_records_selected_source_format_metadata(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"video")
    metadata = media.with_suffix(".source.json")
    metadata.write_text(
        json.dumps({"selected": {"format_id": "401", "height": 2160}, "max_height": 2160}),
        encoding="utf-8",
    )
    manifest = PipelineManifest("campaign")
    _record_source_media_metadata(manifest, "video", media)
    assert manifest.run_metadata["source_media"]["video"]["selected"]["height"] == 2160

    metadata.write_text("not-json", encoding="utf-8")
    before = dict(manifest.run_metadata["source_media"])
    _record_source_media_metadata(manifest, "broken", media)
    assert manifest.run_metadata["source_media"] == before


def _scores() -> EditorialScores:
    return EditorialScores(8, 8, 8, 8, 5, 7, 4, 7, 8, 9, 8, 8)


def _concept(index: int) -> ClipConcept:
    start = float((index - 1) * 10)
    return ClipConcept(
        f"concept-{index}",
        "allowed",
        start,
        start + 9.0,
        f"automation story {index} made {index * 100} dollars and ended successfully.",
        f"topic-{index}",
        f"automation story {index}",
        "ended successfully.",
        "money_story",
        9.0,
        _scores(),
        8.0 - index * 0.1,
        f"cluster-{index}",
        f"fingerprint-{index}",
    )


def _yield_brief(tmp_path: Path) -> Path:
    path = tmp_path / "yield-brief.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": "yield",
                "title": "Automation",
                "objective": "Produce a resilient batch",
                "keywords": ["automation", "money"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "min_clip_seconds": 8,
                "max_clip_seconds": 20,
                "clip_count": 1,
                "max_clips_per_source": 3,
                "production": {
                    "candidate_pool_size": 36,
                    "concept_count": 3,
                    "variants_per_concept": 1,
                    "final_render_budget": 2,
                },
                "hooks": {"enabled": ["direct"]},
            }
        )
    )
    return path


def _yield_subtitle(tmp_path: Path) -> Path:
    path = tmp_path / "yield.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:09.000\n"
        "automation story one made 100 dollars and ended successfully.\n\n"
        "00:00:10.000 --> 00:00:19.000\n"
        "automation story two made 200 dollars and ended successfully.\n\n"
        "00:00:20.000 --> 00:00:29.000\n"
        "automation story three made 300 dollars and ended successfully.\n"
    )
    return path


class FailFirstRenderer(FakeRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def render(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first render failed")
        return super().render(*args, **kwargs)


def test_render_failure_promotes_reserve_until_target_is_reached(tmp_path: Path) -> None:
    brief = _yield_brief(tmp_path)
    subtitle = _yield_subtitle(tmp_path)
    media = tmp_path / "yield.mp4"
    media.write_bytes(b"source")
    concepts = [_concept(1), _concept(2), _concept(3)]
    with patch("clipper.pipeline.select_distinct_concepts", return_value=concepts):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "render-replace"),
            source_client=FakeSource(subtitle, media),
            renderer=FailFirstRenderer(),
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "DEGRADED"
    assert manifest["actual"]["rendered_finalists"] == 2
    assert manifest["funnel"]["render_attempts"] == 3
    assert manifest["funnel"]["replacement_attempts"] == 1
    assert len(manifest["submission_shortlist"]) == 1
    assert all(item["plan_id"] for item in manifest["submission_shortlist"])


def test_qc_failure_promotes_reserve_and_shortlist_uses_only_qc_passed_clips(
    tmp_path: Path,
) -> None:
    brief = _yield_brief(tmp_path)
    subtitle = _yield_subtitle(tmp_path)
    media = tmp_path / "yield-qc.mp4"
    media.write_bytes(b"source")
    concepts = [_concept(1), _concept(2), _concept(3)]
    qc_results = [
        {"status": "FAIL", "issues": ["first caption mismatch"]},
        {"status": "PASS", "issues": []},
        {"status": "PASS", "issues": []},
    ]
    with (
        patch("clipper.pipeline.select_distinct_concepts", return_value=concepts),
        patch("clipper.pipeline.run_technical_qc", side_effect=qc_results),
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "qc-replace"),
            source_client=FakeSource(subtitle, media),
            renderer=FakeRenderer(),
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "DEGRADED"
    assert manifest["funnel"]["technical_qc_fail"] == 1
    assert manifest["funnel"]["technical_qc_pass"] == 2
    accepted = {item["plan_id"] for item in manifest["rendered_clips"]}
    assert {item["plan_id"] for item in manifest["submission_shortlist"]} <= accepted
    assert not list((run_dir / "clips").glob("attempt-01-*.mp4"))
    assert list((run_dir / "rejected").glob("attempt-01-*.mp4"))


def test_pipeline_fails_when_reserve_pool_cannot_reach_render_target(tmp_path: Path) -> None:
    brief = _yield_brief(tmp_path)
    subtitle = _yield_subtitle(tmp_path)
    media = tmp_path / "yield-fail.mp4"
    media.write_bytes(b"source")
    concepts = [_concept(1), _concept(2), _concept(3)]
    with patch("clipper.pipeline.select_distinct_concepts", return_value=concepts):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "yield-fail"),
            source_client=FakeSource(subtitle, media),
            renderer=BrokenRenderer(),
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["status_reason"] == "render_yield_below_required_target"
    assert manifest["actual"] == {
        "rendered_finalists": 0,
        "submission_shortlist": 0,
        "distinct_finalist_concepts": 0,
        "distinct_shortlist_concepts": 0,
    }
    assert manifest["funnel"]["render_attempts"] == 3
    assert manifest["funnel"]["render_failures"] == 3
    assert (run_dir / "funnel.json").is_file()
    assert (run_dir / "rejections.json").is_file()


class RepairedPreflightRenderer(FakeRenderer):
    def render(self, *args, **kwargs):
        output_path = super().render(*args, **kwargs)
        output_path.with_suffix(".tracking-preflight.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "initial_issues": ["back-and-forth crop oscillation detected"],
                    "repaired_with_stable_fallback": True,
                    "final_issues": [],
                    "final_framing_mode": "stable_portrait_fallback",
                }
            )
        )
        return output_path


def test_pipeline_records_tracking_preflight_repair(tmp_path: Path) -> None:
    brief = _yield_brief(tmp_path)
    subtitle = _yield_subtitle(tmp_path)
    media = tmp_path / "yield-preflight.mp4"
    media.write_bytes(b"source")
    concepts = [_concept(1), _concept(2), _concept(3)]
    with patch("clipper.pipeline.select_distinct_concepts", return_value=concepts):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "preflight"),
            source_client=FakeSource(subtitle, media),
            renderer=RepairedPreflightRenderer(),
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["funnel"]["tracking_preflight_pass"] == 2
    assert manifest["funnel"]["tracking_preflight_repaired"] == 2
    assert manifest["funnel"]["tracking_preflight_fail"] == 0
    canonical_files = list((run_dir / "canonical").glob("*.json"))
    assert len(canonical_files) == 1
    canonical = json.loads(canonical_files[0].read_text())
    assert canonical["schema_version"] == "canonical-timeline-v1"
    assert manifest["run_metadata"]["canonical_timelines"]


class FakeOpenEditorialProvider:
    identity = ModelIdentity("fake-editor", "rev1", "none", "test", "prompt1", "schema1")

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.payloads: dict[str, dict[str, object]] = {}

    def complete_json(
        self, *, task: str, payload: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        self.calls.append(task)
        self.payloads[task] = payload
        usage = InferenceUsage("test", "2026-08-08T00:00:00Z", 0.01)
        if task == "episode_editorial_profile":
            value: dict[str, object] = {
                "summary": "A short explanatory conversation",
                "valuable_moment_characteristics": ["self-contained explanation"],
                "avoid_characteristics": ["unsupported context"],
                "confidence": 0.95,
            }
        elif task.startswith("story_moments:"):
            words = payload["words"]
            assert isinstance(words, list)
            word_ids = [item["word_id"] for item in words if isinstance(item, dict)]
            value = {
                "moments": [
                    {
                        "moment_id": "moment-1",
                        "supporting_word_ids": word_ids,
                        "semantic_summary": "An explanation of saving time",
                        "narrative_structure": "explanation",
                        "required_prior_context": "",
                        "required_followup_context": "",
                        "editorial_reason": "It stands alone as a complete explanation",
                        "confidence": 0.92,
                    }
                ]
            }
        elif task == "clip_concepts":
            moments = payload["moments"]
            assert isinstance(moments, list) and isinstance(moments[0], dict)
            word_ids = moments[0]["supporting_word_ids"]
            value = {
                "concepts": [
                    {
                        "concept_id": "concept-1",
                        "story_moment_ids": ["moment-1"],
                        "supporting_word_ids": word_ids,
                        "semantic_summary": "Complete source-grounded explanation",
                        "standalone_context": "",
                        "narrative_structure": "explanation",
                        "recommended_duration": 9.0,
                        "visual_dependencies": [],
                        "confidence": 0.91,
                    }
                ]
            }
        elif task == "global_concept_comparison":
            value = {"concept_ids": ["concept-1"]}
        elif task.startswith("hook_variants:"):
            concept = payload["concept"]
            assert isinstance(concept, dict)
            word_ids = list(concept["supporting_word_ids"])
            value = {
                "variants": [
                    {
                        "variant_id": "hook-1",
                        "strategy_label": "start on the source explanation",
                        "source_word_ids": word_ids,
                        "overlay_text": None,
                        "rationale": "The source opening is already clear",
                        "confidence": 0.9,
                    }
                ]
            }
        elif task.startswith("edit_plans:"):
            concept = payload["concept"]
            assert isinstance(concept, dict)
            word_ids = list(concept["supporting_word_ids"])
            value = {
                "plans": [
                    {
                        "plan_id": "plan-1",
                        "video_id": "allowed",
                        "concept_id": "concept-1",
                        "variant_id": "hook-1",
                        "source_word_ids": word_ids,
                        "hook_source_word_ids": word_ids,
                        "overlay_text": None,
                        "strategy_label": "source explanation",
                        "caption_platform": "tiktok",
                        "confidence": 0.9,
                    }
                ]
            }
        else:
            raise AssertionError(f"unexpected task {task}")
        return ProviderResult(value, self.identity, usage)


class FakeOpenEmbeddingProvider:
    identity = ModelIdentity("fake-embedding", "rev1", "none", "test", "none", "embedding1")

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> ProviderResult[list[list[float]]]:
        self.calls += 1
        vectors = [[1.0, float(index + 1)] for index, _ in enumerate(texts)]
        return ProviderResult(
            vectors,
            self.identity,
            InferenceUsage("test", "2026-08-08T00:00:00Z", 0.01, input_units=len(texts)),
        )


def test_open_editorial_pipeline_bypasses_all_heuristic_entry_points(tmp_path: Path) -> None:
    brief = tmp_path / "open-brief.json"
    brief.write_text(
        json.dumps(
            {
                "campaign_id": "open-campaign",
                "title": "Any domain",
                "objective": "Find useful standalone moments",
                "keywords": ["required-by-current-brief-schema"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "min_clip_seconds": 8,
                "max_clip_seconds": 20,
                "clip_count": 1,
                "production": {
                    "candidate_pool_size": 10,
                    "concept_count": 1,
                    "variants_per_concept": 1,
                    "final_render_budget": 1,
                    "minimum_distinct_finalist_concepts": 1,
                },
            }
        )
    )
    subtitle = tmp_path / "open.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\n"
        "This explanation stands alone and clearly saves people time today.\n"
    )
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source")
    editorial = FakeOpenEditorialProvider()
    embedding = FakeOpenEmbeddingProvider()
    forbidden = RuntimeError("heuristic path must not run")
    with (
        patch("clipper.pipeline._cached_editorial_analysis", side_effect=forbidden),
        patch("clipper.pipeline.select_distinct_concepts", side_effect=forbidden),
        patch("clipper.pipeline.generate_hook_variants", side_effect=forbidden),
        patch("clipper.pipeline.build_edit_plan", side_effect=forbidden),
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "open-artifacts",
                cache_root=tmp_path / "open-cache",
                editorial_engine="open",
                compute_profile="local-lite",
                editorial_chunk_words=200,
                editorial_chunk_overlap_words=20,
            ),
            source_client=FakeSource(subtitle, media),
            editorial_provider=editorial,
            embedding_provider=embedding,
            render=False,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_metadata"]["editorial_inference"]["engine"] == "open"
    assert manifest["run_metadata"]["editorial_inference"]["degraded"] is False
    assert manifest["funnel"]["story_moments"] == 1
    assert manifest["funnel"]["raw_concepts"] == 1
    assert manifest["funnel"]["hook_variants"] == 1
    assert manifest["funnel"]["edit_plans"] == 1
    assert manifest["edit_plans"][0]["caption_start_word"] == "This"
    assert (run_dir / "open-model" / "model-invocations.json").is_file()
    assert editorial.calls == [
        "episode_editorial_profile",
        "story_moments:0",
        "clip_concepts",
        "global_concept_comparison",
        "hook_variants:concept-1",
        "edit_plans:concept-1",
    ]
    assert embedding.calls == 1


class DummyVisionProvider:
    identity = ModelIdentity("fake-vlm", "rev", "none", "test", "visual", "v1")

    def inspect(self, *, task: str, frames: list[Path], context: dict[str, object]):
        raise AssertionError("pipeline visual review is patched in this test")


def _visual_result(report: VisualReviewReport):
    return (
        report,
        [
            ProviderResult(
                {"decision": report.decision},
                DummyVisionProvider.identity,
                InferenceUsage("test", "now", 0.01, input_units=4),
            )
        ],
    )


def test_visual_editorial_qc_repair_promotes_reserve_before_finalist_acceptance(
    tmp_path: Path,
) -> None:
    brief = _yield_brief(tmp_path)
    subtitle = _yield_subtitle(tmp_path)
    media = tmp_path / "yield-visual.mp4"
    media.write_bytes(b"source")
    concepts = [_concept(1), _concept(2), _concept(3)]
    repair = VisualReviewReport(
        "REPAIR",
        "The crop reverses while the same speaker continues.",
        0.95,
        (
            VisualReviewIssue(
                "crop_oscillation",
                1.0,
                1.8,
                "HIGH",
                0.95,
                "TRACKING",
                "The virtual camera jumps away and back.",
            ),
        ),
    )
    passed = VisualReviewReport("PASS", "The clip is visually coherent.", 0.95)
    with (
        patch("clipper.pipeline.select_distinct_concepts", return_value=concepts),
        patch(
            "clipper.pipeline.review_rendered_clip",
            side_effect=[_visual_result(repair), _visual_result(passed), _visual_result(passed)],
        ) as review_mock,
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "visual-replace",
                visual_review_enabled=True,
            ),
            source_client=FakeSource(subtitle, media),
            renderer=FakeRenderer(),
            visual_review_provider=DummyVisionProvider(),
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "DEGRADED"
    assert manifest["status_reason"] == "recovered_with_replacement_candidates"
    assert manifest["funnel"]["editorial_qc_fail"] == 1
    assert manifest["funnel"]["editorial_qc_pass"] == 2
    assert manifest["funnel"]["render_attempts"] == 3
    assert manifest["funnel"]["replacement_attempts"] == 1
    assert len(manifest["editorial_qc"]) == 3
    assert len(manifest["rendered_clips"]) == 2
    assert all(item["decision"] == "PASS" for item in manifest["editorial_qc"][1:])
    rejected = [item for item in manifest["rejections"] if item.get("stage") == "editorial_qc"]
    assert rejected[0]["reasons"] == ["crop_oscillation"]
    assert rejected[0]["repair_stages"] == ["TRACKING"]
    assert list((run_dir / "rejected").glob("attempt-01-*.mp4"))
    assert review_mock.call_count == 3


def test_visual_review_escalation_is_recorded_in_pipeline_manifest(tmp_path: Path) -> None:
    brief = _yield_brief(tmp_path)
    subtitle = _yield_subtitle(tmp_path)
    media = tmp_path / "yield-visual-escalation.mp4"
    media.write_bytes(b"source")
    concepts = [_concept(1), _concept(2), _concept(3)]
    passed = VisualReviewReport("PASS", "Escalated review agrees.", 0.95, escalated=True)
    with (
        patch("clipper.pipeline.select_distinct_concepts", return_value=concepts),
        patch(
            "clipper.pipeline.review_rendered_clip",
            return_value=_visual_result(passed),
        ),
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "visual-escalation",
                visual_review_enabled=True,
                visual_escalation_enabled=True,
                compute_profile="quality",
            ),
            source_client=FakeSource(subtitle, media),
            renderer=FakeRenderer(),
            visual_review_provider=DummyVisionProvider(),
            visual_escalation_provider=DummyVisionProvider(),
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["funnel"]["visual_review_escalations"] == 2
    assert manifest["funnel"]["editorial_qc_pass"] == 2
    assert manifest["run_metadata"]["visual_inference"]["primary_model"]["model_id"] == "fake-vlm"
    assert (
        manifest["run_metadata"]["visual_inference"]["escalation_model"]["model_id"] == "fake-vlm"
    )


class _OpenGroundingSource(FakeSource):
    def download_subtitles(self, _video: VideoCandidate, _work_dir: Path, _language: str) -> Path:
        raise AssertionError("open grounding must not acquire subtitles")


class _GroundingTranscription:
    identity = ModelIdentity("asr", "rev", "none", "test", "none", "canonical-v1")

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self, source: Path, *, video_id: str, source_hash: str
    ) -> ProviderResult[CanonicalTimeline]:
        self.calls += 1
        assert source.is_file()
        words = tuple(
            CanonicalWord(
                f"{video_id}:w{index:07d}",
                text,
                float(index),
                float(index) + 0.8,
                None,
                0.95,
                "word_exact",
                "fake-asr",
            )
            for index, text in enumerate(
                [
                    "This",
                    "explanation",
                    "stands",
                    "alone",
                    "and",
                    "clearly",
                    "saves",
                    "people",
                    "time",
                    "today",
                ]
            )
        )
        return ProviderResult(
            CanonicalTimeline(video_id, source_hash, words),
            self.identity,
            InferenceUsage("test", "now", 0.01),
        )


class _GroundingAlignment:
    identity = ModelIdentity("align", "rev", "none", "test", "none", "canonical-v1")

    def __init__(self) -> None:
        self.calls = 0

    def align(self, source: Path, timeline: CanonicalTimeline) -> ProviderResult[CanonicalTimeline]:
        self.calls += 1
        assert source.is_file()
        words = tuple(
            CanonicalWord(
                word.word_id,
                word.text,
                word.source_start,
                word.source_end,
                word.speaker_id,
                word.confidence,
                "aligned",
                "fake-asr+alignment",
            )
            for word in timeline.words
        )
        return ProviderResult(
            CanonicalTimeline(timeline.video_id, timeline.source_hash, words),
            self.identity,
            InferenceUsage("test", "now", 0.01),
        )


class _GroundingDiarization:
    identity = ModelIdentity("diar", "rev", "none", "test", "none", "canonical-v1")

    def __init__(self) -> None:
        self.calls = 0

    def diarize(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]:
        self.calls += 1
        assert source.is_file()
        words = tuple(
            CanonicalWord(
                word.word_id,
                word.text,
                word.source_start,
                word.source_end,
                "SPEAKER_00",
                word.confidence,
                word.timing_mode,
                word.transcript_source,
            )
            for word in timeline.words
        )
        return ProviderResult(
            CanonicalTimeline(timeline.video_id, timeline.source_hash, words),
            self.identity,
            InferenceUsage("test", "now", 0.01),
        )


def test_open_grounding_owns_canonical_timeline_and_bypasses_subtitles(tmp_path: Path) -> None:
    brief = tmp_path / "grounded-open.json"
    brief.write_text(
        json.dumps(
            {
                "campaign_id": "open-grounding",
                "title": "Any domain",
                "objective": "Find useful standalone moments",
                "keywords": ["schema-required"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "min_clip_seconds": 8,
                "max_clip_seconds": 20,
                "clip_count": 1,
                "production": {
                    "candidate_pool_size": 10,
                    "concept_count": 1,
                    "variants_per_concept": 1,
                    "final_render_budget": 1,
                    "minimum_distinct_finalist_concepts": 1,
                },
            }
        )
    )
    subtitle = tmp_path / "must-not-read.vtt"
    subtitle.write_text("WEBVTT\n")
    media = tmp_path / "grounding-source.mp4"
    media.write_bytes(b"grounding-source")
    asr = _GroundingTranscription()
    alignment = _GroundingAlignment()
    diarization = _GroundingDiarization()
    run_dir = run_pipeline(
        brief,
        settings=PipelineSettings(
            artifact_root=tmp_path / "grounded-artifacts",
            cache_root=tmp_path / "grounded-cache",
            editorial_engine="open",
            grounding_engine="open",
            compute_profile="local-lite",
            editorial_chunk_words=200,
            editorial_chunk_overlap_words=20,
        ),
        source_client=_OpenGroundingSource(subtitle, media),
        editorial_provider=FakeOpenEditorialProvider(),
        embedding_provider=FakeOpenEmbeddingProvider(),
        transcription_provider=asr,
        alignment_provider=alignment,
        diarization_provider=diarization,
        render=False,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert asr.calls == alignment.calls == diarization.calls == 1
    assert manifest["run_metadata"]["grounding_inference"]["engine"] == "open"
    assert manifest["run_metadata"]["transcript_sources"]["allowed"]["kind"] == "canonical-open"
    assert manifest["run_metadata"]["canonical_timelines"]["allowed"]["speaker_count"] == 1
    assert manifest["run_metadata"]["canonical_timelines"]["allowed"]["timing_modes"] == ["aligned"]
    assert manifest["edit_plans"][0]["caption_start_word"] == "This"
    transcript = json.loads((run_dir / "transcript.json").read_text())
    assert transcript["allowed"][0]["words"][0]["text"] == "This"

    second = run_pipeline(
        brief,
        settings=PipelineSettings(
            artifact_root=tmp_path / "grounded-artifacts-second",
            cache_root=tmp_path / "grounded-cache",
            editorial_engine="open",
            grounding_engine="open",
            compute_profile="local-lite",
            editorial_chunk_words=200,
            editorial_chunk_overlap_words=20,
        ),
        source_client=_OpenGroundingSource(subtitle, media),
        editorial_provider=FakeOpenEditorialProvider(),
        embedding_provider=FakeOpenEmbeddingProvider(),
        transcription_provider=asr,
        alignment_provider=alignment,
        diarization_provider=diarization,
        render=False,
    )
    assert second.is_dir()
    assert asr.calls == alignment.calls == diarization.calls == 1


def test_open_grounding_requires_complete_provider_set(tmp_path: Path) -> None:
    brief = _write_pipeline_brief(tmp_path)
    with pytest.raises(ValueError, match="transcription, alignment, and diarization"):
        run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "bad-grounding",
                grounding_engine="open",
            ),
            transcription_provider=_GroundingTranscription(),
            render=False,
        )


def test_open_editorial_pipeline_consumes_sparse_visual_timeline_when_media_exists(
    tmp_path: Path,
) -> None:
    brief = tmp_path / "visual-open.json"
    brief.write_text(
        json.dumps(
            {
                "campaign_id": "visual-open",
                "title": "Any visual domain",
                "objective": "Find source-grounded stories",
                "keywords": ["schema-required"],
                "source_channel_ids": ["UC1"],
                "rights_confirmed": True,
                "min_clip_seconds": 8,
                "max_clip_seconds": 20,
                "clip_count": 1,
                "production": {
                    "candidate_pool_size": 10,
                    "concept_count": 1,
                    "variants_per_concept": 1,
                    "final_render_budget": 1,
                    "minimum_distinct_finalist_concepts": 1,
                },
            }
        )
    )
    subtitle = tmp_path / "visual-open.vtt"
    subtitle.write_text("WEBVTT\n")
    media = tmp_path / "visual-open.mp4"
    media.write_bytes(b"visual-source")
    editorial = FakeOpenEditorialProvider()
    embedding = FakeOpenEmbeddingProvider()
    asr = _GroundingTranscription()
    alignment = _GroundingAlignment()
    diarization = _GroundingDiarization()
    visual = VisualTimeline(
        "allowed",
        "visual-hash",
        (
            VisualEvent(
                1.0,
                2.0,
                "scene-1",
                "The guest visibly demonstrates an object while speaking.",
                ("SPEAKER_00",),
                ("demonstration",),
                0.95,
            ),
        ),
    )
    visual_result = ProviderResult(
        {"events": []},
        DummyVisionProvider.identity,
        InferenceUsage("test", "now", 0.01, input_units=1),
    )
    with patch(
        "clipper.pipeline.scout_visual_timeline",
        return_value=(visual, visual_result),
    ) as scout:
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "visual-open-artifacts",
                cache_root=tmp_path / "visual-open-cache",
                editorial_engine="open",
                grounding_engine="open",
                compute_profile="local-lite",
                visual_scout_enabled=True,
                editorial_chunk_words=200,
                editorial_chunk_overlap_words=20,
            ),
            source_client=_OpenGroundingSource(subtitle, media),
            editorial_provider=editorial,
            embedding_provider=embedding,
            visual_scout_provider=DummyVisionProvider(),
            transcription_provider=asr,
            alignment_provider=alignment,
            diarization_provider=diarization,
            render=False,
        )
    assert scout.call_count == 1
    profile_visual = editorial.payloads["episode_editorial_profile"]["visual_evidence"]
    story_visual = editorial.payloads["story_moments:0"]["visual_evidence"]
    assert isinstance(profile_visual, list) and profile_visual[0]["scene_id"] == "scene-1"
    assert isinstance(story_visual, list) and story_visual[0]["event_labels"] == ["demonstration"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_metadata"]["visual_inference"]["scout_runs"][0]["event_count"] == 1
    assert (run_dir / "visual" / "allowed.json").is_file()


def test_visual_scout_acquires_full_media_after_vtt_transcript(tmp_path: Path) -> None:
    brief = _write_pipeline_brief(tmp_path)
    subtitle = tmp_path / "visual-scout.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\nAutomation can save time for a business.\n",
        encoding="utf-8",
    )
    media = tmp_path / "visual-source.mp4"
    media.write_bytes(b"visual-source")

    class CountingSource(FakeSource):
        def __init__(self, subtitle_path: Path, media_path: Path) -> None:
            super().__init__(subtitle_path, media_path)
            self.media_downloads = 0

        def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
            self.media_downloads += 1
            return super().download_media(video, work_dir)

    source = CountingSource(subtitle, media)
    timeline = VisualTimeline(
        "allowed",
        "visual-source-hash",
        (VisualEvent(0.5, 1.5, "scene-1", "visible reaction", (), ("reaction",), 0.9),),
    )
    result = ProviderResult(
        {"events": []},
        DummyVisionProvider.identity,
        InferenceUsage("test", "now", 0.01, input_units=1),
    )
    with patch(
        "clipper.pipeline.scout_visual_timeline",
        return_value=(timeline, result),
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "visual-vtt-artifacts",
                visual_scout_enabled=True,
            ),
            source_client=source,
            visual_scout_provider=DummyVisionProvider(),
            render=False,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert source.media_downloads == 1
    assert manifest["run_metadata"]["transcript_sources"]["allowed"]["kind"] == "youtube-vtt"
    assert manifest["run_metadata"]["visual_inference"]["scout_runs"][0]["event_count"] == 1


def test_visual_scout_failure_degrades_without_dropping_source_analysis(tmp_path: Path) -> None:
    brief = _write_pipeline_brief(tmp_path)
    subtitle = tmp_path / "visual-fail.vtt"
    subtitle.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:09.000\nAutomation can save time for a business.\n",
        encoding="utf-8",
    )
    media = tmp_path / "visual-fail.mp4"
    media.write_bytes(b"visual-source")
    with patch("clipper.pipeline.scout_visual_timeline", side_effect=RuntimeError("tail decode")):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(
                artifact_root=tmp_path / "visual-fail-artifacts",
                visual_scout_enabled=True,
            ),
            source_client=FakeSource(subtitle, media),
            visual_scout_provider=DummyVisionProvider(),
            render=False,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["funnel"]["story_moments"] > 0
    assert manifest["run_metadata"]["visual_inference"]["scout_errors"][0]["error"] == "tail decode"
