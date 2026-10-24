"""Review-only 14-second cinematic book-lesson animatic.

Supply three high-resolution approved stills and already-generated narration.
This animates camera framing; it is *not* character performance or lip-sync.
It never invokes a model and never produces a publication-eligible artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image

FRAME = (1080, 1920)
SHOT_DURATIONS = (5.0, 5.3, 4.45)
TRANSITION_SECONDS = 0.4
DURATION_SECONDS = round(sum(SHOT_DURATIONS) - 2 * TRANSITION_SECONDS, 2)
CREATIVE_STATUS = "MOTION_ANIMATIC_REFERENCE_ONLY"


class AnimaticError(ValueError):
    """Asset or rendering contract was not met."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=240,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AnimaticError(f"animatic render command failed: {exc}") from exc


def _frame(source: Path, output: Path) -> None:
    if not source.is_file():
        raise AnimaticError(f"missing approved visual asset: {source}")
    with Image.open(source) as image:
        if image.width < 300 or image.height < 500:
            raise AnimaticError("source art below 300x500 reference minimum")
        if image.width / image.height < 0.45:
            raise AnimaticError("character art is too narrow for portrait framing")
        if image.width / image.height > 0.85:
            raise AnimaticError("character art must be supplied as a portrait shot")
        res = image.convert("RGB").resize(FRAME, Image.Resampling.LANCZOS)
        output.parent.mkdir(parents=True, exist_ok=True)
        res.save(output, "JPEG", quality=94, subsampling=0)


def render(
    artwork: tuple[Path, Path, Path],
    narration: Path,
    output: Path,
    source_run: str,
) -> dict[str, object]:
    if len(artwork) != 3:
        raise AnimaticError("exactly three storyboard beats are required")
    if not narration.is_file():
        raise AnimaticError(f"cached Kokoro narration missing: {narration}")
    if not source_run.isdecimal():
        raise AnimaticError("GitHub source run ID must be numeric")
    frames: list[Path] = []
    clips: list[Path] = []
    for i, (image, seconds) in enumerate(zip(artwork, SHOT_DURATIONS, strict=True)):
        frame_path = output.parent / f"animatic-frame-{i + 1:02d}.jpg"
        _frame(image, frame_path)
        frames.append(frame_path)
        clip = output.parent / f"animatic-clip-{i + 1:02d}.mp4"
        zoom = "1.0 + 0.00020*on" if i != 1 else "1.065 - 0.00026*on"
        vf = (
            "scale=1280:2276:flags=lanczos,"
            f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30,"
            "format=yuv420p"
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-framerate",
                "30",
                "-i",
                str(frame_path),
                "-vf",
                vf,
                "-t",
                str(seconds),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ]
        )
        clips.append(clip)
    first = SHOT_DURATIONS[0] - TRANSITION_SECONDS
    second = SHOT_DURATIONS[0] + SHOT_DURATIONS[1] - 2 * TRANSITION_SECONDS
    graph = (
        f"[0:v][1:v]xfade=transition=fade:duration={TRANSITION_SECONDS}:"
        f"offset={first}[v1];"
        f"[v1][2:v]xfade=transition=fade:duration={TRANSITION_SECONDS}:"
        f"offset={second}[v]"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        command.extend(["-i", str(clip)])
    command.extend(
        [
            "-i",
            str(narration),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-t",
            str(DURATION_SECONDS),
            "-af",
            f"afade=t=out:st={DURATION_SECONDS - 0.17:.2f}:d=0.15",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run(command)
    details = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,width,height",
                "-of",
                "json",
                str(output),
            ],
            text=True,
        )
    )
    duration = float(details["format"]["duration"])
    if abs(duration - DURATION_SECONDS) > 0.15:
        raise AnimaticError("rendered media duration did not match storyboard")
    if not any(
        item.get("codec_type") == "video" and (item.get("width"), item.get("height")) == FRAME
        for item in details["streams"]
    ):
        raise AnimaticError("missing 1080x1920 video stream")
    if not any(item.get("codec_type") == "audio" for item in details["streams"]):
        raise AnimaticError("missing cached Kokoro narration stream")
    _run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"])
    manifest: dict[str, object] = {
        "status": CREATIVE_STATUS,
        "publication_eligible": False,
        "creative_review": "PENDING",
        "actual_character_animation": False,
        "audio_model": "Kokoro from source GitHub Action",
        "source_run_id": source_run,
        "fps": 30,
        "resolution": list(FRAME),
        "duration_seconds": duration,
        "full_decode": True,
        "video_sha256": _sha256(output),
        "audio_sha256": _sha256(narration),
        "visual_asset_sha256": [_sha256(item) for item in artwork],
        "limitations": [
            "Camera moves on stills; no facial/body acting or lip-sync",
            "Story source images require independent rights and visual review",
            "This artifact must not be treated as publishable animation",
        ],
    }
    (output.parent / "animatic-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", action="append", type=Path, required=True)
    parser.add_argument("--kokoro-wav", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.shot) != 3:
        parser.error("Supply exactly three --shot images")
    print(
        json.dumps(
            render(tuple(args.shot), args.kokoro_wav, args.output, args.source_run),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
