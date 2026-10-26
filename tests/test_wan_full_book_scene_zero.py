"""Scene-zero recovery must pin the approved historical picture and full video."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wan_full_book_scene_zero as scene_zero  # noqa: E402


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_restores_historical_visual_without_claiming_current_qwen_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "new-qwen-hook.png"
    current.write_bytes(b"new Qwen picture")
    audio = tmp_path / "new-narrator.wav"
    audio.write_bytes(b"new narrator")
    old_frame = b"historical approved picture"
    old_video = b"old-approved-video" * 25000
    monkeypatch.setattr(scene_zero, "IMAGE_SHA256", _digest(old_frame))
    monkeypatch.setattr(scene_zero, "VIDEO_SHA256", _digest(old_video))
    monkeypatch.setattr(scene_zero, "verified_wan_output", lambda *args: True)
    monkeypatch.setenv("GH_TOKEN", "synthetic")
    monkeypatch.setenv("GITHUB_REPOSITORY", "test/clipper")
    destination = tmp_path / "rendered" / "scene-00"
    calls: list[list[str]] = []

    def fake_download(command: list[str], *, check: bool) -> None:
        assert check
        calls.append(command)
        target = Path(command[command.index("--dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        pattern = command[command.index("--pattern") + 1]
        if pattern.startswith("full-book-prep-"):
            (target / "frames").mkdir(exist_ok=True)
            (target / "frames" / "hook.png").write_bytes(old_frame)
        else:
            (target / "scene-00").mkdir(exist_ok=True)
            (target / "scene-00" / "wan-s2v-candidate.mp4").write_bytes(old_video)
            (target / "scene-00" / "model-evidence.json").write_text("{}")

    monkeypatch.setattr(scene_zero.subprocess, "run", fake_download)
    scene_zero.restore(current, audio, destination)
    assert scene_zero.verified_scene_zero_visual(destination, current, audio)
    assert (destination / "locked-original-keyframe.png").read_bytes() == old_frame
    assert (destination / "wan-s2v-candidate.mp4").read_bytes() == old_video
    reuse = json.loads((destination / "visual-reuse.json").read_text())
    assert reuse["original_keyframe_sha256"] == _digest(old_frame)
    assert reuse["current_keyframe_sha256"] == _digest(current.read_bytes())
    assert reuse["publish_eligible"] is False
    assert reuse["lip_sync_new_voice"] == "NOT_VERIFIED"
    scene_zero.restore(current, audio, destination)
    assert len(calls) == 2
    current.write_bytes(b"a different unapproved Qwen frame")
    assert not scene_zero.verified_scene_zero_visual(destination, current, audio)
    current.write_bytes(b"new Qwen picture")
    (destination / "locked-original-keyframe.png").write_bytes(b"tampered")
    assert not scene_zero.verified_scene_zero_visual(destination, current, audio)


def test_recovery_rejects_unpinned_historical_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "new.png"
    current.write_bytes(b"new")
    audio = tmp_path / "new.wav"
    audio.write_bytes(b"speech")
    monkeypatch.setenv("GH_TOKEN", "synthetic")
    monkeypatch.setenv("GITHUB_REPOSITORY", "test/clipper")

    def fake_download(command: list[str], *, check: bool) -> None:
        target = Path(command[command.index("--dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "not-original.png").write_bytes(b"untrusted image")

    monkeypatch.setattr(scene_zero.subprocess, "run", fake_download)
    with pytest.raises(ValueError, match="historical scene-zero keyframe absent"):
        scene_zero.restore(current, audio, tmp_path / "scene-00")
