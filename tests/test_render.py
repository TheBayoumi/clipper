from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from unittest.mock import Mock, patch

import pytest

from clipper.models import ClipCandidate, TranscriptSegment
from clipper.render import FFmpegRenderer, RenderError, build_ffmpeg_command, create_srt
from clipper_engine.rendering import delivery


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
    assert "scale=360:640" in joined
    assert "gblur=sigma=18" in joined
    assert "FontSize=10" in joined
    assert "MarginV=28" in joined
    assert "loudnorm=I=-14" in joined
    assert ";[captioned]format=yuv420p[v]" in joined
    assert "libx264" in command
    assert "ultrafast" in command
    assert command[command.index("-threads") + 1] == "1"
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


def test_create_srt_strips_youtube_speaker_marker(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 8, "text", 1)
    path = create_srt(
        clip,
        [TranscriptSegment(0, 4, ">> Speaker turn starts here.")],
        tmp_path / "speaker.srt",
    )
    content = path.read_text(encoding="utf-8")
    assert ">>" not in content
    assert "Speaker turn starts here." in content


def test_delivery_profile_allows_vertical_full_frame_composition() -> None:
    source_profile = {
        "width": 3840,
        "height": 2160,
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "16:9",
    }
    config = {
        "output": {
            "width": 1080,
            "height": 1920,
            "full_source_frame": True,
        }
    }
    profile = delivery.profile_for_output(source_profile, config)
    graph = delivery.full_frame_filter(1080, 1920)

    assert profile["width"] == 1080
    assert profile["height"] == 1920
    assert profile["display_aspect_ratio"] == "9:16"
    assert delivery.composition_required(source_profile, config) is True
    assert "force_original_aspect_ratio=decrease" in graph
    assert "flags=lanczos" in graph
    assert "crop=360:640" in graph
    assert "overlay=(W-w)/2:(H-h)/2" in graph



def test_portrait_delivery_uses_calibrated_matte_without_cropping() -> None:
    source_profile = {
        "width": 3840,
        "height": 2160,
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "16:9",
    }
    config = {
        "output": {
            "width": 1080,
            "height": 1920,
            "portrait_layout": {
                "enabled": True,
                "width": 1080,
                "height": 1920,
                "background_hex": "#0F1115",
                "visual_left": 0,
                "visual_top": 656,
                "visual_width": 1080,
                "visual_height": 608,
                "title_top_height": 656,
                "title_y_positions": {"1": [330], "2": [275, 385], "3": [236, 352, 438]},
                "title_font_sizes": {"1": [66], "2": [62, 58], "3": [60, 56, 52]},
            },
        }
    }

    graph, metadata = delivery.composition_filter(source_profile, config)

    assert "scale=1080:608:force_original_aspect_ratio=decrease:flags=lanczos" in graph
    assert "pad=1080:1920:0:656:color=0x0F1115" in graph
    assert "crop=" not in graph
    assert metadata["mode"] == "portrait_matte_full_frame"
    assert metadata["source_foreground_full_frame"] is True
    assert metadata["source_foreground_crop_used"] is False
