import json
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from unittest.mock import Mock, patch

import pytest

from clipper.models import ClipCandidate, TranscriptSegment, TranscriptWord
from clipper.render import FFmpegRenderer, RenderError, build_ffmpeg_command
from clipper.tracking import FaceAnchor, TrackingPlan


def test_build_ffmpeg_command_contains_speaker_locked_vertical_and_audio_filters(
    tmp_path: Path,
) -> None:
    clip = ClipCandidate("v", 1.25, 31.5, "text", 1)
    plan = TrackingPlan(
        1.12,
        1280,
        720,
        (FaceAnchor(0.0, 20.0, 10.0), FaceAnchor(1.0, 20.0, 10.0)),
        True,
    )
    command = build_ffmpeg_command(
        "source.mp4",
        "out.mp4",
        clip,
        tmp_path / "x:y.ass",
        tracking_plan=plan,
    )
    joined = " ".join(command)
    assert "scale=1080:1920" in joined
    assert "crop=360:640" in joined
    assert "gblur=" not in joined
    assert "split=2" not in joined
    assert "[bg]" not in joined
    assert "if(lt(t,1.000)" in joined
    assert "subtitles=" in joined
    assert "loudnorm=I=-14" in joined
    assert ";[captioned]format=yuv420p[v]" in joined
    assert "libx264" in command
    assert "ultrafast" in command
    assert command[command.index("-threads") + 1] == "1"
    assert "1.250" in command
    assert "30.250" in command


def test_build_ffmpeg_command_uses_static_center_zoom_without_tracking(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 20, "text", 1)
    command = build_ffmpeg_command("source.mp4", "out.mp4", clip, tmp_path / "captions.ass")
    joined = " ".join(command)
    assert "x='(iw-ow)/2'" in joined
    assert "y='(ih-oh)/2'" in joined


def test_build_ffmpeg_command_overlays_campaign_watermark(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 20, "text", 1)
    watermark = tmp_path / "watermark.png"
    command = build_ffmpeg_command(
        "source.mp4",
        "out.mp4",
        clip,
        tmp_path / "captions.ass",
        watermark_path=watermark,
    )
    joined = " ".join(command)
    assert str(watermark) in command
    assert "[1:v]scale=180:-1" in joined
    assert "overlay=W-w-48:48" in joined


def test_renderer_requires_ffmpeg_and_valid_speaker_settings() -> None:
    with patch("clipper.render.shutil.which", return_value=None), pytest.raises(RenderError):
        FFmpegRenderer()
    with patch("clipper.render.shutil.which", return_value="/usr/bin/ffmpeg"):
        with pytest.raises(RenderError, match="zoom_factor"):
            FFmpegRenderer(zoom_factor=1.5)
        with pytest.raises(RenderError, match="speaker_sample_fps"):
            FFmpegRenderer(speaker_sample_fps=20)
        with pytest.raises(RenderError, match="speaker_switch_margin"):
            FFmpegRenderer(speaker_switch_margin=4)
        with pytest.raises(RenderError, match="speaker_transition_seconds"):
            FFmpegRenderer(speaker_transition_seconds=2)
        with pytest.raises(RenderError, match="speaker_window_seconds"):
            FFmpegRenderer(speaker_window_seconds=0.1)
        with pytest.raises(RenderError, match="speaker_min_detection_coverage"):
            FFmpegRenderer(speaker_min_detection_coverage=0.01)


def test_renderer_success_writes_ass_and_speaker_evidence(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 10, "text", 1)
    segments = [
        TranscriptSegment(
            0,
            2,
            "caption now",
            (TranscriptWord(0, 0.8, "caption"), TranscriptWord(1.0, 1.8, "now")),
        )
    ]
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    with patch("clipper.render.shutil.which", return_value="/usr/bin/ffmpeg"):
        renderer = FFmpegRenderer()
    output = tmp_path / "ok.mp4"
    plan = TrackingPlan(
        1.12,
        640,
        360,
        (FaceAnchor(0, 10, 5), FaceAnchor(10, 10, 5)),
        True,
        speaker_tracks=1,
        speaker_switches=0,
    )

    def success(*_args, **_kwargs):
        output.write_bytes(b"video")
        return Mock()

    with (
        patch("clipper.render.plan_speaker_crop", return_value=plan),
        patch("clipper.render.subprocess.run", side_effect=success),
    ):
        assert renderer.render(source, output, clip, segments) == output
    assert r"{\ko" in output.with_suffix(".ass").read_text(encoding="utf-8")
    tracking = json.loads(output.with_suffix(".tracking.json").read_text(encoding="utf-8"))
    assert tracking["face_detected"] is True
    assert tracking["zoom_factor"] == 1.12
    assert tracking["speaker_focus"] is True
    assert tracking["framing_mode"] == "speaker_locked_portrait"
    assert tracking["speaker_switches"] == 0


def test_renderer_failures(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 10, "text", 1)
    segments = [TranscriptSegment(0, 10, "caption")]
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    plan = TrackingPlan(1.12, 640, 360)
    with patch("clipper.render.shutil.which", return_value="/usr/bin/ffmpeg"):
        renderer = FFmpegRenderer(speaker_focus=False)

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

    command = build_ffmpeg_command(
        source,
        tmp_path / "unused.mp4",
        clip,
        tmp_path / "captions.ass",
        tracking_plan=plan,
    )
    joined = " ".join(command)
    assert "crop=180:320" in joined
    assert "gblur=" not in joined
