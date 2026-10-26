"""P6: CPU-only, renderer-independent assembly from content-verified cached shots.

Never generate missing footage, guess word timings, layer titles/subtitles, or
use clip audio. The full original narration is the sole audio input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from book_audio_timeline import FPS, RATE, SCHEMA, digest, file_sha, inspect_audio

ASSETS_SCHEMA = "clipper-shot-assets-v1"
EVIDENCE_SCHEMA = "clipper-clean-master-evidence-v2"


class AssemblyError(ValueError):
    """Reject missing, stale, or incomplete cached media before FFmpeg starts."""


def run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        raise AssemblyError(f"media command failed: {command[0]}") from exc
    return completed.stdout


def probe(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type,width,height,r_frame_rate,duration,nb_frames",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )
        if not isinstance(payload, dict):
            raise ValueError("invalid ffprobe payload")
        return payload
    except (ValueError, KeyError, TypeError) as exc:
        raise AssemblyError(f"invalid media probe: {path.name}") from exc


def _fingerprinted(plan: dict[str, Any]) -> None:
    if plan.get("schema") != SCHEMA:
        raise AssemblyError("unsupported audio-locked timeline")
    original = {key: value for key, value in plan.items() if key != "fingerprint"}
    if plan.get("fingerprint") != digest(original):
        raise AssemblyError("audio-locked timeline fingerprint mismatch")


def _prepare(
    plan: dict[str, Any], audio_path: Path, assets: dict[str, Any]
) -> tuple[list[tuple[Path, int, int]], int]:
    """Validate all sources and frame coverage before allocating encoding time."""
    _fingerprinted(plan)
    if plan.get("publish_eligible") is not False:
        raise AssemblyError("publication boundary was crossed")
    frames, audio_sha = inspect_audio(audio_path)
    audio = plan["audio"]
    if (
        audio.get("sha256") != audio_sha
        or audio.get("samples") != frames
        or audio.get("sample_rate") != RATE
    ):
        raise AssemblyError("locked narration hash or duration mismatch")
    output = plan["output"]
    if output != {
        "width": 1080,
        "height": 1920,
        "fps": FPS,
        "frames": math.ceil(frames * FPS / RATE),
    }:
        raise AssemblyError("video geometry or frame count disagrees with locked narration")
    if (
        not isinstance(assets, dict)
        or set(assets) != {"schema", "edl_fingerprint", "assets"}
        or assets["schema"] != ASSETS_SCHEMA
        or assets["edl_fingerprint"] != plan["fingerprint"]
        or not isinstance(assets["assets"], list)
    ):
        raise AssemblyError("shot manifest does not belong to the current timeline")
    shots = plan.get("shots")
    records = assets["assets"]
    if not isinstance(shots, list) or not shots or len(records) != len(shots):
        raise AssemblyError("a cached source is required for every directed shot")
    checked: list[tuple[Path, int, int]] = []
    previous = 0
    seen: set[str] = set()
    for shot, asset in zip(shots, records, strict=True):
        if not isinstance(shot, dict) or not isinstance(asset, dict):
            raise AssemblyError("invalid shot or cached asset record")
        if set(asset) != {
            "shot_id",
            "path",
            "sha256",
            "source_start_frame",
            "evidence_path",
            "evidence_sha256",
        }:
            raise AssemblyError("incomplete cached video provenance")
        shot_id = shot.get("id")
        if (
            not isinstance(shot_id, str)
            or not shot_id
            or shot_id in seen
            or asset["shot_id"] != shot_id
        ):
            raise AssemblyError("cached shot ID mismatch")
        seen.add(shot_id)
        start, end = shot.get("start_frame"), shot.get("end_frame")
        source_start = asset["source_start_frame"]
        if (
            type(start) is not int
            or type(end) is not int
            or type(source_start) is not int
            or start != previous
            or not start < end <= output["frames"]
            or source_start < 0
        ):
            raise AssemblyError("shot timing contains a gap, overlap, or invalid source trim")
        previous = end
        if not isinstance(asset["path"], str) or not asset["path"]:
            raise AssemblyError("cached video path required")
        if not isinstance(asset["evidence_path"], str) or not asset["evidence_path"]:
            raise AssemblyError("cached provenance path required")
        video = Path(asset["path"])
        evidence = Path(asset["evidence_path"])
        if not video.is_file() or not evidence.is_file():
            raise AssemblyError(f"missing cached video or evidence for {shot_id}")
        if file_sha(video) != asset["sha256"]:
            raise AssemblyError(f"cached video sha256 mismatch for {shot_id}")
        if file_sha(evidence) != asset["evidence_sha256"]:
            raise AssemblyError(f"cached evidence sha256 mismatch for {shot_id}")
        try:
            record = json.loads(evidence.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssemblyError("unreadable cached model evidence") from exc
        if (
            not isinstance(record, dict)
            or record.get("video_sha256") != asset["sha256"]
            or record.get("publish_eligible") is not False
            or not isinstance(record.get("model_id"), str)
            or not record["model_id"]
            or not isinstance(record.get("model_revision"), str)
            or not record["model_revision"]
        ):
            raise AssemblyError(f"unqualified cached model evidence for {shot_id}")
        metadata = probe(video)
        picture = [s for s in metadata.get("streams", []) if s.get("codec_type") == "video"]
        if len(picture) != 1:
            raise AssemblyError(f"cached shot must have one picture stream: {shot_id}")
        source_duration = float(metadata["format"]["duration"])
        if not math.isfinite(source_duration) or source_duration <= 0:
            raise AssemblyError("invalid cached shot duration")
        required_seconds = (source_start + end - start) / FPS
        if source_duration + 1 / FPS < required_seconds:
            raise AssemblyError(f"cached shot too short; never extend or loop: {shot_id}")
        checked.append((video, source_start, end - start))
    if previous != output["frames"]:
        raise AssemblyError("cached timeline does not cover complete narration")
    return checked, frames


def assemble(
    plan: dict[str, Any],
    audio_path: Path,
    assets: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Render a clean master; all audio comes from the verified unmodified WAV."""
    verified, frames = _prepare(plan, audio_path, assets)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "clean-master.mp4"
    temporary = output_dir / ".clean-master-incomplete.mp4"
    if temporary.exists():
        temporary.unlink()
    inputs: list[str] = []
    graph: list[str] = []
    for index, (video, source_start, count) in enumerate(verified):
        inputs.extend(["-i", str(video)])
        graph.append(
            f"[{index}:v]fps={FPS},"
            f"trim=start_frame={source_start}:end_frame={source_start + count},"
            f"setpts=PTS-STARTPTS,"
            f"scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[v{index}]"
        )
    inputs.extend(["-i", str(audio_path)])
    graph.append(
        "".join(f"[v{index}]" for index in range(len(verified)))
        + f"concat=n={len(verified)}:v=1:a=0[vout]"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[vout]",
        "-map",
        f"{len(verified)}:a:0",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-r",
        str(FPS),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        run(command)
        media = probe(temporary)
        video = [s for s in media.get("streams", []) if s.get("codec_type") == "video"]
        sound = [s for s in media.get("streams", []) if s.get("codec_type") == "audio"]
        if (
            len(video) != 1
            or len(sound) != 1
            or (video[0].get("width"), video[0].get("height")) != (1080, 1920)
            or video[0].get("r_frame_rate") != "24/1"
            or int(video[0].get("nb_frames", 0)) != plan["output"]["frames"]
            or not math.isclose(float(media["format"]["duration"]), frames / RATE, abs_tol=0.12)
        ):
            raise AssemblyError("assembled master failed source duration/geometry QA")
        run(["ffmpeg", "-v", "error", "-i", str(temporary), "-f", "null", "-"])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "episode_id": plan["episode_id"],
        "edl_fingerprint": plan["fingerprint"],
        "original_audio_sha256": file_sha(audio_path),
        "video_sha256": file_sha(target),
        "shot_count": len(verified),
        "picture_frames_requested": plan["output"]["frames"],
        "duration_seconds": float(media["format"]["duration"]),
        "overlay_mode": "NONE",
        "embedded_clip_audio": "DISCARDED",
        "single_authoritative_narration": True,
        "caption_variant": False,
        "creative_review": "PENDING",
        "publish_eligible": False,
    }
    evidence_path = output_dir / "clean-master-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return evidence


def make_legacy_replay(
    plan: dict[str, Any], audio_path: Path, frame_dir: Path, folders: list[Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wrap existing four-shot Wan evidence into the generalized CPU assembler."""
    from wan_shared_shot_cache import verified_wan_output

    if plan.get("schema") != "clipper-full-book-episode-v1":
        raise AssemblyError("unsupported historical Wan shot plan")
    original = {key: value for key, value in plan.items() if key != "fingerprint"}
    if (
        plan.get("fingerprint")
        != hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    ):
        raise AssemblyError("historical full-book plan fingerprint mismatch")
    frames, sha = inspect_audio(audio_path)
    if (
        plan.get("full_audio_sha256") != sha
        or plan.get("voice", {}).get("voice_id") != "am_adam"
        or not isinstance(plan.get("shots"), list)
        or len(plan["shots"]) != len(folders)
    ):
        raise AssemblyError("historical voice, shot count, or audio hash mismatch")
    total_frames = math.ceil(frames * FPS / RATE)
    shots: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    previous = 0
    for index, (shot, folder) in enumerate(zip(plan["shots"], folders, strict=True)):
        video = folder / "wan-s2v-candidate.mp4"
        evidence = folder / "model-evidence.json"
        picture = frame_dir / f"{shot['keyframe_name']}.png"
        voice = audio_path.parent / shot["audio"]
        if (
            not picture.is_file()
            or not voice.is_file()
            or file_sha(voice) != shot.get("audio_sha256")
            or not verified_wan_output(
                video,
                evidence,
                file_sha(picture),
                file_sha(voice),
                shot["prompt"],
                shot["seed"],
            )
        ):
            raise AssemblyError(f"historical Wan shot {index} fails source hash checks")
        end = total_frames if index == len(folders) - 1 else round(float(shot["end_seconds"]) * FPS)
        if end <= previous:
            raise AssemblyError("historical shot duration collapsed")
        shots.append(
            {
                "id": shot["id"],
                "start_frame": previous,
                "end_frame": end,
                "director_role": "LEGACY_APPROVED_SHOT",
            }
        )
        assets.append(
            {
                "shot_id": shot["id"],
                "path": str(video),
                "sha256": file_sha(video),
                "source_start_frame": 0,
                "evidence_path": str(evidence),
                "evidence_sha256": file_sha(evidence),
            }
        )
        previous = end
    edl = {
        "schema": SCHEMA,
        "episode_id": "legacy-atomic-habits-regression",
        "audio": {"sha256": sha, "samples": frames, "sample_rate": RATE},
        "output": {"width": 1080, "height": 1920, "fps": FPS, "frames": total_frames},
        "shots": shots,
        "alignment_review": "LEGACY_SCENE_AUDIO_LOCK_NO_WORD_ALIGNMENT",
        "creative_review": "PENDING",
        "publish_eligible": False,
    }
    edl["fingerprint"] = digest(edl)
    return edl, {
        "schema": ASSETS_SCHEMA,
        "edl_fingerprint": edl["fingerprint"],
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    new = subparsers.add_parser("assemble")
    new.add_argument("--edl", type=Path, required=True)
    new.add_argument("--assets", type=Path, required=True)
    legacy = subparsers.add_parser("legacy-replay")
    legacy.add_argument("--legacy-plan", type=Path, required=True)
    legacy.add_argument("--frames-dir", type=Path, required=True)
    legacy.add_argument("--shot-dirs", type=Path, nargs="+", required=True)
    for child in (new, legacy):
        child.add_argument("--audio", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "legacy-replay":
        edl, sources = make_legacy_replay(
            json.loads(args.legacy_plan.read_text(encoding="utf-8")),
            args.audio,
            args.frames_dir,
            args.shot_dirs,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "audio-locked-edl.json").write_text(
            json.dumps(edl, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "verified-shot-assets.json").write_text(
            json.dumps(sources, indent=2) + "\n", encoding="utf-8"
        )
    else:
        edl = json.loads(args.edl.read_text(encoding="utf-8"))
        sources = json.loads(args.assets.read_text(encoding="utf-8"))
    result = assemble(edl, args.audio, sources, args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
