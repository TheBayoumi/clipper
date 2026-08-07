import json
from pathlib import Path
from unittest.mock import patch

from clipper.models import CampaignBrief, ClipCandidate, TranscriptSegment, VideoCandidate
from clipper.pipeline import PipelineSettings, run_pipeline


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
    def render(
        self,
        _source_path: Path,
        output_path: Path,
        _clip: ClipCandidate,
        _segments: list[TranscriptSegment],
    ) -> Path:
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
