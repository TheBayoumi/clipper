import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
    assert manifest["actual"] == {"rendered_finalists": 0, "submission_shortlist": 0}
    assert manifest["funnel"]["render_attempts"] == 3
    assert manifest["funnel"]["render_failures"] == 3
    assert (run_dir / "funnel.json").is_file()
    assert (run_dir / "rejections.json").is_file()
