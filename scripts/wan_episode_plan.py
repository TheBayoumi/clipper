"""Production-length Clipper episode timeline from measured cached Kokoro WAVs.

This is the 30–60s *production* contract, not the archived 15s Atomic Habits
technical storyboard. A creative director must supply real shot briefs before
any video generation. No filler, fake silence, fixed shot count, or new TTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

SCHEMA = "clipper-wan-episode-timeline-v1"
MIN_EPISODE_SECONDS = 30.0
MAX_EPISODE_SECONDS = 60.0
TARGET_SHOT_SECONDS = 5.0
MAX_SHOT_SECONDS = 6.0
MIN_SHOT_SECONDS = 2.5
OUTPUT_FPS = 24
WAV_RATE = 24000


class ProductionTimelineError(ValueError):
    """Reject incomplete 30–60 second productions rather than padding content."""


def fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def inspect_wav(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProductionTimelineError(f"cached narration WAV missing: {path}")
    try:
        with wave.open(str(path), "rb") as recording:
            channels = recording.getnchannels()
            rate = recording.getframerate()
            sample_width = recording.getsampwidth()
            frames = recording.getnframes()
            compression = recording.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise ProductionTimelineError("cannot read cached narration as WAV") from exc
    if (channels, rate, sample_width, compression) != (1, WAV_RATE, 2, "NONE"):
        raise ProductionTimelineError("cached Kokoro audio must be PCM16 mono 24 kHz")
    if not 2.5 <= frames / rate <= 25:
        raise ProductionTimelineError("cached scene audio length outside supported range")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_frames": frames,
        "sample_rate": rate,
        "duration_seconds": round(frames / rate, 6),
    }


def _scene_beats(frames: int) -> list[tuple[int, int]]:
    """Partition only at audio-frame boundaries; no timeline padding or trimming."""
    seconds = frames / WAV_RATE
    count = max(1, math.ceil(seconds / TARGET_SHOT_SECONDS))
    while seconds / count > MAX_SHOT_SECONDS:
        count += 1
    if seconds / count < MIN_SHOT_SECONDS:
        raise ProductionTimelineError("narration beat would be too brief; revise scene writing")
    borders = [(frames * index) // count for index in range(count + 1)]
    return list(zip(borders[:-1], borders[1:], strict=True))


def build_episode_plan(
    episode: dict[str, Any], audio_paths: list[Path], character_source_sha256: str
) -> dict[str, Any]:
    if not isinstance(episode, dict) or set(episode) != {"title", "scenes"}:
        raise ProductionTimelineError("episode must use the cached Qwen script contract")
    scenes = episode["scenes"]
    if not isinstance(scenes, list) or not 3 <= len(scenes) <= 5:
        raise ProductionTimelineError("production episode requires 3–5 narrative sections")
    if len(audio_paths) != len(scenes):
        raise ProductionTimelineError("every scene needs its exact cached Kokoro narration")
    if len(character_source_sha256) != 64 or any(
        x not in "0123456789abcdef" for x in character_source_sha256
    ):
        raise ProductionTimelineError("character master requires a pinned SHA256")
    measurements = [inspect_wav(p) for p in audio_paths]
    overall_frames = sum(m["sample_frames"] for m in measurements)
    total_seconds = overall_frames / WAV_RATE
    if not MIN_EPISODE_SECONDS <= total_seconds <= MAX_EPISODE_SECONDS:
        raise ProductionTimelineError(
            f"actual Kokoro narration totals {total_seconds:.3f}s; "
            "a production video MUST be 30–60s. Revise the SCRIPT, "
            "not the video speed, padding, or clip repetition."
        )
    shots: list[dict[str, Any]] = []
    offset = 0
    for scene_number, (scene, audio) in enumerate(zip(scenes, measurements, strict=True)):
        if not isinstance(scene, dict) or set(scene) != {"headline", "narration"}:
            raise ProductionTimelineError("scene must preserve headline and narration")
        if not all(isinstance(scene[x], str) and scene[x].strip() for x in scene):
            raise ProductionTimelineError("scene headline or narration missing")
        segments = _scene_beats(audio["sample_frames"])
        for beat_number, (begin, end) in enumerate(segments):
            start_samples, end_samples = offset + begin, offset + end
            shots.append(
                {
                    "id": f"scene-{scene_number:02d}-beat-{beat_number:02d}",
                    "section": scene_number,
                    "beat": beat_number,
                    "headline": scene["headline"],
                    "narration": scene["narration"],
                    "audio_sha256": audio["sha256"],
                    "audio_start_samples": begin,
                    "audio_end_samples": end,
                    "start_frame": round(start_samples * OUTPUT_FPS / WAV_RATE),
                    "end_frame": round(end_samples * OUTPUT_FPS / WAV_RATE),
                    "start_seconds": round(start_samples / WAV_RATE, 6),
                    "end_seconds": round(end_samples / WAV_RATE, 6),
                    # A real story director, not a deterministic keyword heuristic,
                    # must select meaningful action/motion and the video backend.
                    "director_brief": None,
                    "renderer": None,
                    "status": "NEEDS_STORY_DIRECTION",
                    "creative_review": "PENDING",
                    "publish_eligible": False,
                }
            )
        offset += audio["sample_frames"]
    if shots[0]["start_frame"] != 0 or shots[-1]["end_frame"] != round(
        overall_frames * OUTPUT_FPS / WAV_RATE
    ):
        raise ProductionTimelineError("planned shots fail full narration coverage")
    for left, right in zip(shots, shots[1:]):
        if left["end_frame"] != right["start_frame"]:
            raise ProductionTimelineError("shot frame boundaries have overlaps or gaps")
        if left["end_frame"] <= left["start_frame"]:
            raise ProductionTimelineError("shot frame interval is empty")
    result = {
        "schema": SCHEMA,
        "production_policy": "30–60s, narrated, no padding or redundant shots",
        "title": episode["title"],
        "character_source_sha256": character_source_sha256,
        "script_sha256": fingerprint(episode),
        "audio": measurements,
        "output": {
            "width": 1080,
            "height": 1920,
            "fps": OUTPUT_FPS,
            "duration_seconds": round(total_seconds, 6),
            "total_frames": shots[-1]["end_frame"],
        },
        "narrative_section_count": len(scenes),
        "shot_count": len(shots),
        "shots": shots,
        "stage": "AUDIO_LOCKED_AWAITING_AI_STORY_DIRECTION",
        "published": False,
        "publish_eligible": False,
        "accepted_mp4_sha256": None,
    }
    result["fingerprint"] = fingerprint(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument(
        "--audio", type=Path, required=True, nargs="+", help="Cached scene WAVs in script order"
    )
    parser.add_argument("--character-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    plan = build_episode_plan(episode, args.audio, args.character_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "duration_seconds": plan["output"]["duration_seconds"],
                "scene_count": plan["narrative_section_count"],
                "shot_count": plan["shot_count"],
                "fingerprint": plan["fingerprint"],
                "status": plan["stage"],
            }
        )
    )


if __name__ == "__main__":
    main()
