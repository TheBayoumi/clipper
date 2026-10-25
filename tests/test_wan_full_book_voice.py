"""The narrator must be original, source-locked, and independently cached."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wan_full_book_voice as voice  # noqa: E402


def test_original_voice_profile_is_not_an_identifiable_celebrity() -> None:
    assert voice.VOICE_ID == "am_adam"
    assert "original" in voice.VOICE_STYLE
    assert voice.VOICE_MODEL == "hexgrad/Kokoro-82M"
    assert 0.75 <= voice.VOICE_SPEED <= 1.1


def test_cached_voice_reuses_exact_scene_narrations_without_loading_kokoro(tmp_path: Path) -> None:
    rows = [
        {"narration": f"Original carefully sourced narrative about easier habits {i}."}
        for i in range(4)
    ]
    source = {"book": {"title": "Atomic Habits"}, "episode": {"scenes": rows}}
    source_path = tmp_path / "manifest.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "source"
    output.mkdir()
    plan = {
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "shots": [{"narration": row["narration"], "id": f"scene-{i:02d}"} for i, row in enumerate(rows)],
        "seconds": 38,
        "fingerprint": "original",
    }
    (output / "plan.json").write_text(json.dumps(plan))
    cache = tmp_path / "voice-cache"
    cache.mkdir()
    for row in rows:
        identity = {
            "contract": voice.CONTRACT,
            "narration": row["narration"],
            "voice": voice.VOICE_ID,
            "voice_style": voice.VOICE_STYLE,
            "speed": voice.VOICE_SPEED,
            "model": voice.VOICE_MODEL,
            "revision": voice.VOICE_MODEL_REVISION,
        }
        key = voice.sha(json.dumps(identity, sort_keys=True).encode())
        (cache / f"{key}.wav").write_bytes(voice.make_wav(b"\0\0" * (8 * voice.RATE)))
    voice.generate(source_path, output, cache)
    result = json.loads((output / "plan.json").read_text())
    assert 32 <= result["seconds"] <= 34
    assert result["voice"]["celebrity_voice_clone"] is False
    assert result["voice"]["voice_id"] == "am_adam"
    assert result["fingerprint"] != "original"
    assert result["shots"][0]["audio_sha256"] == voice.sha((output / "scene-00.wav").read_bytes())
    assert voice.sha((output / "full-kokoro.wav").read_bytes()) == result["full_audio_sha256"]
