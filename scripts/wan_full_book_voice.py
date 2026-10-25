"""Generate a distinct, original cinematic baritone voice for one sourced book script.

Uses Kokoro's licensed open-weight male timbre; not a celebrity impersonation.
Only narration audio changes. Qwen3 script and any good visual shot are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import wave
from pathlib import Path

VOICE_ID = "am_adam"
VOICE_STYLE = "original_warm_low_cinematic_baritone"
VOICE_MODEL = "hexgrad/Kokoro-82M"
VOICE_MODEL_REVISION = "v1.0"
VOICE_SPEED = 0.98
RATE = 24_000
PAUSE_SECONDS = 0.14
CONTRACT = "clipper-original-book-voice-v1"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def wav_bytes(samples: object) -> bytes:
    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    speech = np.asarray(samples, dtype=np.float32)
    if speech.ndim != 1 or len(speech) < 1000 or not np.isfinite(speech).all():
        raise ValueError("Kokoro returned invalid source speech")
    sf.write(buffer, np.clip(speech, -1, 1), RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def pcm(payload: bytes) -> bytes:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
            raise ValueError("Kokoro output WAV must be mono PCM16 at 24 kHz")
        if wav.getnframes() < RATE:
            raise ValueError("Kokoro segment is too short")
        return wav.readframes(wav.getnframes())


def make_wav(frames: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(frames)
    return buffer.getvalue()


def generate(manifest: Path, output: Path, cache: Path) -> None:
    original = json.loads(manifest.read_text(encoding="utf-8"))
    plan_file = output / "plan.json"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    scenes = original["episode"]["scenes"]
    if original["book"]["title"] != "Atomic Habits" or len(scenes) != len(plan["shots"]):
        raise ValueError("book, source Qwen script, and planned shots disagree")
    if plan["source_manifest_sha256"] != sha(manifest.read_bytes()):
        raise ValueError("Qwen scene manifest fingerprint changed")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    pipeline = None
    full_frames = bytearray()
    previous = 0
    for index, (scene, shot) in enumerate(zip(scenes, plan["shots"], strict=True)):
        if shot["narration"] != scene["narration"]:
            raise ValueError("script changed after editorial lock")
        identity = {
            "contract": CONTRACT,
            "narration": scene["narration"],
            "voice": VOICE_ID,
            "voice_style": VOICE_STYLE,
            "speed": VOICE_SPEED,
            "model": VOICE_MODEL,
            "revision": VOICE_MODEL_REVISION,
        }
        key = sha(json.dumps(identity, sort_keys=True).encode())
        cached = cache / f"{key}.wav"
        try:
            speech_frames = pcm(cached.read_bytes())
            print(f"ORIGINAL_VOICE_SCENE_CACHE_HIT scene={index}")
        except (OSError, ValueError, wave.Error, EOFError):
            if pipeline is None:
                from kokoro import KPipeline

                pipeline = KPipeline(lang_code="a", repo_id=VOICE_MODEL)
            import numpy as np

            parts = [
                np.asarray(audio, dtype=np.float32)
                for _graphemes, _phonemes, audio in pipeline(
                    scene["narration"], voice=VOICE_ID, speed=VOICE_SPEED
                )
            ]
            if not parts:
                raise ValueError("No speech returned from open-source Kokoro voice") from None
            speech_frames = pcm(wav_bytes(np.concatenate(parts)))
            cached.write_bytes(make_wav(speech_frames))
            print(f"ORIGINAL_VOICE_SCENE_GENERATED scene={index}")
        # Silence is added only between separately authored sentences, never
        # inside words. The scene WAV and episode WAV are built from same bytes.
        if index != len(scenes) - 1:
            speech_frames += b"\0\0" * round(PAUSE_SECONDS * RATE)
        segment = make_wav(speech_frames)
        start = previous
        previous += len(speech_frames) // 2
        target = output / f"scene-{index:02d}.wav"
        target.write_bytes(segment)
        full_frames.extend(speech_frames)
        shot.update({
            "audio": target.name,
            "audio_sha256": sha(segment),
            "start_seconds": round(start / RATE, 6),
            "end_seconds": round(previous / RATE, 6),
            "duration_seconds": round((previous - start) / RATE, 6),
            "voice_id": VOICE_ID,
        })
    full_wav = make_wav(bytes(full_frames))
    duration = previous / RATE
    if not 30 <= duration <= 60:
        raise ValueError(f"New baritone voice duration {duration:.3f}s is outside 30-60s")
    (output / "full-kokoro.wav").write_bytes(full_wav)
    plan.update({
        "seconds": round(duration, 6),
        "full_audio_sha256": sha(full_wav),
        "voice": {
            "model": VOICE_MODEL,
            "revision": VOICE_MODEL_REVISION,
            "voice_id": VOICE_ID,
            "style": VOICE_STYLE,
            "speed": VOICE_SPEED,
            "celebrity_voice_clone": False,
        },
        "status": "ORIGINAL_CINEMATIC_BARITONE_AUDIO_LOCKED_NEEDS_VISUAL_REVIEW",
        "publish_eligible": False,
    })
    plan.pop("fingerprint", None)
    plan["fingerprint"] = sha(json.dumps(plan, sort_keys=True).encode())
    plan_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"ORIGINAL_BARITONE_AUDIO_LOCKED voice={VOICE_ID} seconds={duration:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    generate(args.manifest, args.output, args.cache)


if __name__ == "__main__":
    main()
