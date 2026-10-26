"""P3: bind a single unmodified Kokoro WAV to evidence-backed word timings.

This module performs validation only. It cannot invent transcript timings, run
a forced aligner, synthesize speech, render video, or authorize publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

SCHEMA = "clipper-audio-locked-edl-v1"
RATE = 24_000
FPS = 24
VOICE_MODEL = "hexgrad/Kokoro-82M"
VOICE_REVISION = "v1.0"
VOICE_ID = "am_adam"


class AudioTimelineError(ValueError):
    """Audio, text, alignment, or creative contract is not reproducible."""


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def inspect_audio(path: Path) -> tuple[int, str]:
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != RATE
                or reader.getcomptype() != "NONE"
            ):
                raise AudioTimelineError("full narration must be PCM16 mono 24 kHz")
            frames = reader.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioTimelineError("cannot read original Kokoro narration") from exc
    if not 30 <= frames / RATE <= 60:
        raise AudioTimelineError("real narration must last 30-60 seconds")
    return frames, file_sha(path)


def _fingerprinted(payload: dict[str, Any], label: str) -> None:
    original = {key: value for key, value in payload.items() if key != "fingerprint"}
    if payload.get("fingerprint") != digest(original):
        raise AudioTimelineError(f"{label} fingerprint mismatch")


def build_timeline(
    director_request: dict[str, Any],
    director_plan: dict[str, Any],
    audio: Path,
    voice: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """Convert real alignment samples to immutable, gapless picture frame ranges."""
    if director_request.get("schema") != "clipper-ai-director-request-v1":
        raise AudioTimelineError("invalid director request")
    if director_plan.get("schema") != "clipper-ai-director-plan-v1":
        raise AudioTimelineError("invalid director plan")
    _fingerprinted(director_request, "director request")
    _fingerprinted(director_plan, "director plan")
    if (
        director_plan.get("request_fingerprint") != director_request["fingerprint"]
        or director_plan.get("script_sha256") != director_request.get("script_sha256")
        or director_plan.get("episode_id") != director_request.get("episode_id")
        or director_plan.get("publish_eligible") is not False
        or director_request.get("publish_eligible") is not False
    ):
        raise AudioTimelineError("director identity or publication gate mismatch")
    frames, audio_sha = inspect_audio(audio)
    if not isinstance(voice, dict) or set(voice) != {
        "model", "revision", "voice_id", "style", "speed",
        "text_sha256", "wav_sha256",
    }:
        raise AudioTimelineError("complete Kokoro voice provenance is required")
    sections = director_request["narrative_sections"]
    words = [word for section in sections for word in section["narration"].split()]
    text = " ".join(words)
    if (
        voice["model"] != VOICE_MODEL
        or voice["revision"] != VOICE_REVISION
        or voice["voice_id"] != VOICE_ID
        or voice["style"] != "original_warm_low_cinematic_baritone"
        or type(voice["speed"]) not in (float, int)
        or not math.isclose(voice["speed"], 0.98, abs_tol=1e-9)
        or voice["text_sha256"] != digest(text)
        or voice["wav_sha256"] != audio_sha
    ):
        raise AudioTimelineError("narration voice, text, or audio hash changed")
    if not isinstance(alignment, dict) or set(alignment) != {
        "audio_sha256", "script_sha256", "aligner_model",
        "aligner_revision", "words",
    }:
        raise AudioTimelineError("complete forced alignment provenance required")
    if (
        alignment["audio_sha256"] != audio_sha
        or alignment["script_sha256"] != director_request["script_sha256"]
        or not isinstance(alignment["aligner_model"], str)
        or not alignment["aligner_model"].strip()
        or not isinstance(alignment["aligner_revision"], str)
        or not alignment["aligner_revision"].strip()
    ):
        raise AudioTimelineError("forced-alignment identity mismatch")
    positions = alignment["words"]
    if not isinstance(positions, list) or len(positions) != len(words):
        raise AudioTimelineError("forced alignment must cover every spoken word")
    last_end = 0
    for index, (position, word) in enumerate(zip(positions, words, strict=True)):
        if not isinstance(position, dict) or set(position) != {
            "text", "start_sample", "end_sample"
        }:
            raise AudioTimelineError("alignment word schema mismatch")
        start, end = position["start_sample"], position["end_sample"]
        if (
            position["text"] != word
            or type(start) is not int
            or type(end) is not int
            or not last_end <= start < end <= frames
        ):
            raise AudioTimelineError(f"unaligned or reordered word {index}")
        last_end = end
    def boundary(word_index: int) -> int:
        if word_index == 0:
            return 0
        if word_index == len(words):
            return frames
        return (
            positions[word_index - 1]["end_sample"] + positions[word_index]["start_sample"]
        ) // 2

    output_frames = math.ceil(frames * FPS / RATE)
    locked: list[dict[str, Any]] = []
    previous_frame = 0
    for shot in director_plan["shots"]:
        start_word, end_word = shot["start_word"], shot["end_word"]
        if (
            type(start_word) is not int
            or type(end_word) is not int
            or not 0 <= start_word < end_word <= len(words)
            or shot["narration_text"] != " ".join(words[start_word:end_word])
        ):
            raise AudioTimelineError("director shot violates authoritative word spans")
        start_sample, end_sample = boundary(start_word), boundary(end_word)
        start_frame = round(start_sample * FPS / RATE)
        end_frame = (
            output_frames if end_word == len(words) else round(end_sample * FPS / RATE)
        )
        if start_frame != previous_frame or end_frame <= start_frame:
            raise AudioTimelineError("director shots overlap or collapse after FPS quantization")
        locked.append(
            {
                **shot,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "asset_id": shot["id"],
            }
        )
        previous_frame = end_frame
    if not locked or previous_frame != output_frames:
        raise AudioTimelineError("timeline does not cover complete original narration")
    result = {
        "schema": SCHEMA,
        "episode_id": director_request["episode_id"],
        "director_request_fingerprint": director_request["fingerprint"],
        "director_plan_fingerprint": director_plan["fingerprint"],
        "script_sha256": director_request["script_sha256"],
        "voice": voice,
        "voice_cache_key": digest({
            "text": text,
            "model": voice["model"],
            "revision": voice["revision"],
            "voice_id": voice["voice_id"],
            "style": voice["style"],
            "speed": voice["speed"],
        }),
        "audio": {
            "sha256": audio_sha,
            "samples": frames,
            "sample_rate": RATE,
            "duration_seconds": round(frames / RATE, 6),
        },
        "alignment_sha256": digest(alignment),
        "alignment_review": "PENDING_ACOUSTIC_VERIFICATION",
        "output": {"width": 1080, "height": 1920, "fps": FPS, "frames": output_frames},
        "shots": locked,
        "creative_review": "PENDING",
        "publish_eligible": False,
    }
    result["fingerprint"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--director-plan", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_timeline(
        json.loads(args.request.read_text(encoding="utf-8")),
        json.loads(args.director_plan.read_text(encoding="utf-8")),
        args.wav,
        json.loads(args.voice.read_text(encoding="utf-8")),
        json.loads(args.alignment.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "episode_id": plan["episode_id"],
        "shots": len(plan["shots"]),
        "audio_sha256": plan["audio"]["sha256"],
        "stage": "AUDIO_LOCKED_NO_GPU",
    }))


if __name__ == "__main__":
    main()
