"""Fail-closed, content-addressed contract for Clipper open-weight Wan video shots.

Does not interpret codec success as artistic approval. The original 3D rig
acceptance path remains separate: generative video has no inspectable bones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path

SCHEMA = "clipper-wan-shot-v1"
MODEL_IDS = {
    "s2v": "Wan-AI/Wan2.2-S2V-14B",
    "i2v": "Wan-AI/Wan2.2-I2V-A14B",
}
MODEL_REVISION_S2V = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
MODEL_REVISION_I2V = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
MIN_REFERENCE_WIDTH = 384
MIN_REFERENCE_HEIGHT = 512
MAX_AUDIO_SECONDS = 6.5


class ShotContractError(ValueError):
    """An AI shot lacks verified inputs or output evidence."""


def sha256(path: Path) -> str:
    if not path.is_file():
        raise ShotContractError(f"required file missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_image(path: Path) -> dict[str, object]:
    from PIL import Image

    if not path.is_file() or not 0 < path.stat().st_size <= 20_000_000:
        raise ShotContractError("reference image missing or outside bounded size")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if width < MIN_REFERENCE_WIDTH or height < MIN_REFERENCE_HEIGHT:
            raise ShotContractError(
                f"reference {width}x{height} below Wan technical input floor "
                f"{MIN_REFERENCE_WIDTH}x{MIN_REFERENCE_HEIGHT}"
            )
        if image.mode not in ("RGB", "RGBA"):
            raise ShotContractError("reference must be RGB or RGBA")
    return {
        "sha256": sha256(path),
        "width": width,
        "height": height,
        "creative_reference_quality": (
            "PROVISIONAL_BELOW_1024_NOT_FOR_PRODUCTION_APPROVAL"
            if width < 1024 or height < 1024
            else "MINIMUM_DIMENSIONS_MET_VISUAL_REVIEW_PENDING"
        ),
    }


def inspect_audio(path: Path) -> dict[str, object]:
    if not path.is_file() or not 0 < path.stat().st_size <= 25_000_000:
        raise ShotContractError("audio missing or outside bounded size")
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            width = wav.getsampwidth()
            duration = wav.getnframes() / sample_rate
    except (wave.Error, ZeroDivisionError, EOFError) as exc:
        raise ShotContractError("not a supported PCM WAV") from exc
    if (channels, sample_rate, width) != (1, 24000, 2):
        raise ShotContractError("expected cached Kokoro PCM16 mono 24 kHz WAV")
    if not 1.0 <= duration <= MAX_AUDIO_SECONDS:
        raise ShotContractError("S2V qualification audio must be 1 to 6.5 seconds")
    return {"sha256": sha256(path), "duration_seconds": duration, "sample_rate": sample_rate}


def prepare(
    reference: Path,
    audio: Path,
    prompt: str,
    shot_id: str = "hook",
    seed: int = 724,
    backend: str = "s2v",
) -> dict[str, object]:
    if backend != "s2v":
        raise ShotContractError("only the S2V backend is implemented; I2V not yet qualified")
    if shot_id not in ("hook", "book_intro", "lesson_example"):
        raise ShotContractError("shot must belong to approved Atomic Habits pilot")
    if not 15 <= len(prompt.strip()) <= 800:
        raise ShotContractError("prompt is missing or too long")
    if not 0 <= seed < 2**31:
        raise ShotContractError("seed outside supported range")
    source = inspect_image(reference)
    voice = inspect_audio(audio)
    identity = {
        "schema": SCHEMA,
        "backend": backend,
        "model_id": MODEL_IDS[backend],
        "model_revision": MODEL_REVISION_S2V,
        "shot_id": shot_id,
        "image": source,
        "audio": voice,
        "prompt": prompt.strip(),
        "seed": seed,
        "storyboard": "niches/atomic-habits-approved-storyboard.json",
        "character": "page-to-life-host-cinematic-v1",
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **identity,
        "fingerprint": fingerprint,
        "stage": "INPUTS_QUALIFIED_NOT_RENDERED",
        "creative_review": "PENDING",
        "publish_eligible": False,
    }



def prepare_i2v(
    keyframe: Path,
    prompt: str,
    shot_id: str,
    seed: int = 724,
) -> dict[str, object]:
    """Bind a real starting frame to an action shot; never invent a fake 3D rig."""
    if shot_id not in ("book_intro", "lesson_example"):
        raise ShotContractError("I2V action must belong to approved book/notebook scenes")
    if not 15 <= len(prompt.strip()) <= 1500:
        raise ShotContractError("I2V action prompt is missing or too long")
    if not 0 <= seed < 2**31:
        raise ShotContractError("seed outside supported range")
    image = inspect_image(keyframe)
    payload = {
        "schema": SCHEMA,
        "backend": "i2v",
        "model_id": MODEL_IDS["i2v"],
        "model_revision": MODEL_REVISION_I2V,
        "shot_id": shot_id,
        "image": image,
        "audio": None,
        "prompt": prompt.strip(),
        "seed": seed,
        "storyboard": "niches/atomic-habits-approved-storyboard.json",
        "character": "page-to-life-host-cinematic-v1",
        "keyframe_creative_review": "PENDING",
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "fingerprint": fingerprint,
        "stage": "INPUTS_QUALIFIED_NOT_RENDERED",
        "creative_review": "PENDING",
        "publish_eligible": False,
    }


def audit_mp4(video: Path, request: dict[str, object]) -> dict[str, object]:
    if request.get("schema") != SCHEMA or request.get("stage") != "INPUTS_QUALIFIED_NOT_RENDERED":
        raise ShotContractError("missing qualified request")
    if not video.is_file() or not 25_000 < video.stat().st_size < 300_000_000:
        raise ShotContractError("rendered MP4 missing or implausibly sized")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,width,height,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        frames = [s for s in streams if s.get("codec_type") == "video"]
        if len(frames) != 1:
            raise ShotContractError("must contain exactly one video stream")
        if min(frames[0].get("width", 0), frames[0].get("height", 0)) < 400:
            raise ShotContractError("video resolution below reviewable minimum")
        if not 0.5 <= float(payload["format"]["duration"]) <= 10.0:
            raise ShotContractError("video duration outside qualification window")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
            capture_output=True,
            check=True,
            timeout=180,
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise ShotContractError("video probe or full decode failed") from exc
    return {
        **request,
        "stage": "TECHNICAL_CANDIDATE_AWAITING_VISUAL_REVIEW",
        "video_sha256": sha256(video),
        "video_bytes": video.stat().st_size,
        "width": frames[0]["width"],
        "height": frames[0]["height"],
        "duration_seconds": float(payload["format"]["duration"]),
        "creative_review": "PENDING",
        "identity_consistency": "NOT_ASSESSED",
        "lip_sync": "NOT_ASSESSED",
        "physical_motion": "NOT_ASSESSED",
        "publish_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--shot-id", default="hook")
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.reference, args.audio, args.prompt, args.shot_id, args.seed)
    if args.video is not None:
        manifest = audit_mp4(args.video, manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fingerprint": manifest["fingerprint"], "stage": manifest["stage"]}))


if __name__ == "__main__":
    main()
