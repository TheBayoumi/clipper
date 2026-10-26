"""Regression coverage for CPU-only clean master without any text overlays."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wan_full_book_episode as episode  # noqa: E402
from wan_shared_shot_cache import WAN_MODEL_ID, WAN_REVISION  # noqa: E402


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_clean_master_renders_only_raw_video_and_locked_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    frames = tmp_path / "artifacts/full-book/frames"
    source = tmp_path / "artifacts/full-book/source"
    frames.mkdir(parents=True)
    source.mkdir(parents=True)
    durations = [9.3175] * 4
    names = ["hook", "lesson_example", "book_intro", "hook"]
    shots, folders = [], []
    for i, name in enumerate(names):
        frame = frames / (name + ".png")
        frame.write_bytes((name + " approved picture").encode())
        audio = source / f"scene-{i:02d}.wav"
        audio.write_bytes((f"scene {i} baritone").encode())
        prompt = f"Approved cinematic scene {i}"
        folder = tmp_path / f"rendered/scene-{i:02d}"
        folder.mkdir(parents=True)
        video = b"clipper-wan" * 34000
        (folder / "wan-s2v-candidate.mp4").write_bytes(video)
        (folder / "model-evidence.json").write_text(
            json.dumps(
                {
                    "schema": "clipper-wan-s2v-modal-v1",
                    "model_id": WAN_MODEL_ID,
                    "model_revision": WAN_REVISION,
                    "image_sha256": sha(frame.read_bytes()),
                    "audio_sha256": sha(audio.read_bytes()),
                    "prompt_sha256": sha(prompt.encode()),
                    "seed": 735 + i,
                    "status": "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED",
                    "publish_eligible": False,
                    "creative_review": "PENDING",
                    "video_sha256": sha(video),
                }
            )
        )
        shots.append(
            {
                "id": f"scene-{i:02d}",
                "keyframe_name": name,
                "audio": audio.name,
                "audio_sha256": sha(audio.read_bytes()),
                "prompt": prompt,
                "seed": 735 + i,
                "duration_seconds": durations[i],
            }
        )
        folders.append(folder)
    full_audio = source / "full-kokoro.wav"
    full_audio.write_bytes(b"complete scene-aligned baritone narration")
    plan = {
        "schema": episode.SCHEMA,
        "seconds": 37.27,
        "fingerprint": "approved-plan",
        "voice": {"style": "original_warm_low_cinematic_baritone"},
        "shots": shots,
    }
    video_commands = []

    def fake_command(command: list[str]) -> str:
        if command[0] == "ffprobe" and "format=duration" in command:
            return json.dumps({"format": {"duration": "9.318"}})
        if command[0] == "ffprobe":
            return json.dumps(
                {
                    "format": {"duration": "37.270"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1080,
                            "height": 1920,
                            "r_frame_rate": "24/1",
                        },
                        {"codec_type": "audio"},
                    ],
                }
            )
        if "-filter_complex" in command:
            video_commands.append(command)
            Path(command[-1]).write_bytes(b"mock-clean-video")
        return ""

    monkeypatch.setattr(episode, "command_run", fake_command)
    folder = tmp_path / "final"
    episode.prepare_concat(plan, folders, folder, captions=None, clean=True)
    assert len(video_commands) == 1
    cmd = video_commands[0]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "subtitles=" not in graph
    assert "drawtext=" not in graph
    assert "ass=" not in graph
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    assert str(full_audio) in cmd
    assert (folder / "atomic-habits-clean-master.mp4").is_file()
    assert not (folder / "atomic-habits-full-candidate.mp4").exists()
    evidence = json.loads((folder / "full-book-evidence.json").read_text())
    assert evidence["final_overlay_mode"] == "NONE"
    assert evidence["subtitle_overlay_applied"] is False
    assert evidence["book_title_overlay_applied"] is False
    assert evidence["captions_sha256"] is None


def test_captions_still_required_for_original_subtitled_mode(
    tmp_path: Path
) -> None:
    with pytest.raises(episode.EpisodeError, match="not a valid"):
        episode.prepare_concat({}, [], tmp_path, captions=None)
