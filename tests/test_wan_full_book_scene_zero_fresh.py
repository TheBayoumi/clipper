"""Reject legacy scene-zero media while preserving the other scene render contract."""

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


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_scene_zero_rejects_old_audio_video_even_if_old_mp4_hash_is_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "artifacts/full-book/frames/hook.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"new approved Qwen keyframe")
    audio = tmp_path / "artifacts/full-book/source/scene-00.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"new original baritone")
    clip_dir = tmp_path / "shot-0"
    clip_dir.mkdir()
    video = clip_dir / "wan-s2v-candidate.mp4"
    video.write_bytes(b"wan-mp4" * 50000)
    prompt = "Cinematic original host speaks the new line"
    old_audio_sha = digest(b"old original dialogue")
    manifest = {
        "schema": "clipper-wan-s2v-modal-v1",
        "model_id": WAN_MODEL_ID,
        "model_revision": WAN_REVISION,
        "image_sha256": digest(image.read_bytes()),
        "audio_sha256": old_audio_sha,
        "prompt_sha256": digest(prompt.encode()),
        "seed": 735,
        "status": "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED",
        "creative_review": "PENDING",
        "publish_eligible": False,
        "video_sha256": digest(video.read_bytes()),
    }
    (clip_dir / "model-evidence.json").write_text(json.dumps(manifest))
    shots = [
        {
            "id": "scene-00",
            "keyframe_name": "hook",
            "audio": "scene-00.wav",
            "prompt": prompt,
            "seed": 735,
            "duration_seconds": 9.0,
        }
    ]
    shots.extend(
        {
            "id": f"scene-0{i}",
            "keyframe_name": "hook",
            "audio": "scene-00.wav",
            "prompt": prompt,
            "seed": 735 + i,
            "duration_seconds": 9.0,
        }
        for i in (1, 2, 3)
    )
    plan = {"schema": episode.SCHEMA, "seconds": 36, "shots": shots}

    with pytest.raises(episode.EpisodeError, match="current-audio Wan shot scene-00"):
        episode.prepare_concat(
            plan,
            [clip_dir, clip_dir, clip_dir, clip_dir],
            tmp_path / "output",
            tmp_path / "captions.ass",
        )

    # Only when the S2V manifest is bound to the *new* audio should assembly
    # advance to probing the rendered footage.
    manifest["audio_sha256"] = digest(audio.read_bytes())
    (clip_dir / "model-evidence.json").write_text(json.dumps(manifest))

    def stop_at_probe(command: list[str]) -> str:
        assert command[0] == "ffprobe"
        raise RuntimeError("CURRENT_VOICE_SHOT_ACCEPTED_FOR_MEDIA_PROBE")

    monkeypatch.setattr(episode, "command_run", stop_at_probe)
    with pytest.raises(RuntimeError, match="CURRENT_VOICE_SHOT_ACCEPTED_FOR_MEDIA_PROBE"):
        episode.prepare_concat(
            plan,
            [clip_dir, clip_dir, clip_dir, clip_dir],
            tmp_path / "output",
            tmp_path / "captions.ass",
        )


def test_scene_zero_separation_does_not_modify_scene_1_2_3_jobs() -> None:
    full = (ROOT / ".github/workflows/wan-full-book-production.yml").read_text()
    only_zero = (ROOT / "scripts/wan_full_book_render_scene_zero.sh").read_text()
    recovery = (ROOT / ".github/workflows/wan-full-book-recovery.yml").read_text()
    assert "matrix:\n        shot: [0, 1, 2, 3]" in full
    assert "fail-fast: false" in full
    assert 'python scripts/wan_full_book_shot_cache.py --shot "$SHOT_INDEX"' in full
    assert 'if [[ "$SHOT_INDEX" == "0" ]]; then' in full
    assert "bash scripts/wan_full_book_render_scene_zero.sh" in full
    assert "--shot 0" in only_zero
    assert "SCENE_ZERO_NEW_VOICE_GPU_SKIPPED_VERIFIED_CACHE" in only_zero
    assert "SCENE_ZERO_CURRENT_QWEN_CURRENT_BARITONE_WAN_VERIFIED" in only_zero
    assert "MODAL_ACCOUNT_SLOT: '3'" in recovery
    assert "full-book-shot-0-fresh-" in recovery
    assert "inputs.mode == 'render_fresh_scene0'" in recovery
