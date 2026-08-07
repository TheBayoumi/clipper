import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from clipper.models import CampaignBrief, ClipCandidate, TranscriptSegment, VideoCandidate
from clipper.pipeline import (
    PipelineSettings,
    _campaign_media_candidates,
    _download_asset,
    _normalize_asset_url,
    run_pipeline,
)


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

    settings = PipelineSettings.from_env()
    assert settings.whisper_model == "tiny"
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
