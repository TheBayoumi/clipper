from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from unittest.mock import Mock, patch

import pytest

from clipper.models import ClipCandidate, TranscriptSegment
from clipper.render import FFmpegRenderer, RenderError, build_ffmpeg_command, create_srt


def test_create_srt_rebases_and_clamps_segments(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 10, 20, "text", 1)
    segments = [
        TranscriptSegment(8, 12, "first"),
        TranscriptSegment(15, 22, "second"),
        TranscriptSegment(30, 31, "outside"),
    ]
    path = create_srt(clip, segments, tmp_path / "captions.srt")
    text = path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in text
    assert "00:00:05,000 --> 00:00:10,000" in text
    assert "outside" not in text


def test_build_ffmpeg_command_contains_vertical_and_audio_filters(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 1.25, 31.5, "text", 1)
    command = build_ffmpeg_command("source.mp4", "out.mp4", clip, tmp_path / "x:y.srt")
    joined = " ".join(command)
    assert "scale=1080:1920" in joined
    assert "loudnorm=I=-14" in joined
    assert "libx264" in command
    assert "1.250" in command
    assert "30.250" in command


def test_build_ffmpeg_command_overlays_campaign_watermark(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 20, "text", 1)
    watermark = tmp_path / "watermark.png"
    command = build_ffmpeg_command(
        "source.mp4",
        "out.mp4",
        clip,
        tmp_path / "captions.srt",
        watermark_path=watermark,
    )
    joined = " ".join(command)
    assert str(watermark) in command
    assert "[1:v]scale=180:-1" in joined
    assert "overlay=W-w-48:48" in joined


def test_renderer_requires_ffmpeg() -> None:
    with patch("clipper.render.shutil.which", return_value=None), pytest.raises(RenderError):
        FFmpegRenderer()


def test_renderer_success_and_failures(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 10, "text", 1)
    segments = [TranscriptSegment(0, 10, "caption")]
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    with patch("clipper.render.shutil.which", return_value="/usr/bin/ffmpeg"):
        renderer = FFmpegRenderer()

    output = tmp_path / "ok.mp4"

    def success(*_args, **_kwargs):
        output.write_bytes(b"video")
        return Mock()

    with patch("clipper.render.subprocess.run", side_effect=success):
        assert renderer.render(source, output, clip, segments) == output

    with (
        patch(
            "clipper.render.subprocess.run",
            side_effect=CalledProcessError(1, ["ffmpeg"], stderr="bad render"),
        ),
        pytest.raises(RenderError, match="bad render"),
    ):
        renderer.render(source, tmp_path / "failed.mp4", clip, segments)

    with (
        patch("clipper.render.subprocess.run", side_effect=TimeoutExpired(["ffmpeg"], 900)),
        pytest.raises(RenderError, match="timed out"),
    ):
        renderer.render(source, tmp_path / "timeout.mp4", clip, segments)

    with (
        patch("clipper.render.subprocess.run", return_value=Mock()),
        pytest.raises(RenderError, match="did not create"),
    ):
        renderer.render(source, tmp_path / "missing.mp4", clip, segments)
