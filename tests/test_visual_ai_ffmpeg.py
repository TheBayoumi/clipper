from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import clipper.visual_ai as visual_ai


def test_media_duration_seconds_uses_ffprobe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"media")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="2995.721\n", stderr="")

    monkeypatch.setattr(visual_ai.subprocess, "run", fake_run)

    assert visual_ai.media_duration_seconds(source) == pytest.approx(2995.721)
    assert calls[0][0] == "ffprobe"
    assert calls[0][-1] == str(source)


def test_media_duration_seconds_rejects_invalid_probe_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"media")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="N/A\n", stderr="")

    monkeypatch.setattr(visual_ai.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffprobe output"):
        visual_ai.media_duration_seconds(source)


def test_extract_video_frames_uses_bounded_ffmpeg_analysis_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"media")
    output_dir = tmp_path / "frames"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(visual_ai.subprocess, "run", fake_run)

    frames = visual_ai.extract_video_frames(source, (0.0, 123.456), output_dir)

    assert len(frames) == 2
    assert all(frame.is_file() for frame in frames)
    assert all(command[0] == "ffmpeg" for command in calls)
    for command in calls:
        filter_value = command[command.index("-vf") + 1]
        assert "force_original_aspect_ratio=decrease" in filter_value
        assert str(visual_ai.VISUAL_SAMPLE_MAX_EDGE) in filter_value
        assert command[command.index("-map") + 1] == "0:v:0"
        assert command[command.index("-frames:v") + 1] == "1"


def test_extract_video_frames_surfaces_decoder_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"media")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="decoder exploded")

    monkeypatch.setattr(visual_ai.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"12\.345s: decoder exploded"):
        visual_ai.extract_video_frames(source, (12.345,), tmp_path / "frames")
