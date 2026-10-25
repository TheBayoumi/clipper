"""Preserve user-approved scene 0 picture while replacing the overall voiceover.

The original synced dialogue cannot be reused as new voice audio. This is a
video-only B-roll reuse, never a claim of verified lip synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from wan_shared_shot_cache import verified_wan_output

SOURCE_RUN = 40174311275
VIDEO_SHA256 = "6ba8c31509d75a808d31f29af9f909f9dc04c76628346cf368b35d8ab4a286b1"
IMAGE_SHA256 = "7f03c145af1c1c2f789aba11beadddccdc2649d8fc6bb613f454aa61ea778b67"
OLD_AUDIO_SHA256 = "c2791ea34ec1c4d84e940ff23fd6ac7403d50a2594c494f5379cecf12d774925"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_scene_zero_visual(folder: Path, reference: Path, new_audio: Path) -> bool:
    video = folder / "wan-s2v-candidate.mp4"
    original = folder / "model-evidence.json"
    record = folder / "visual-reuse.json"
    if not reference.is_file() or sha(reference) != IMAGE_SHA256:
        return False
    if not verified_wan_output(
        video, original, IMAGE_SHA256, OLD_AUDIO_SHA256, _original_prompt(), 735
    ):
        return False
    if sha(video) != VIDEO_SHA256:
        return False
    if not record.is_file() or not new_audio.is_file():
        return False
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        return (
            data["source_run"] == SOURCE_RUN
            and data["visual_sha256"] == VIDEO_SHA256
            and data["original_voice_sha256"] == OLD_AUDIO_SHA256
            and data["current_voice_sha256"] == sha(new_audio)
            and data["reuse_type"] == "VIDEO_ONLY_ORIGINAL_NARRATION_MUTED"
            and data["publish_eligible"] is False
        )
    except (ValueError, KeyError, OSError, TypeError):
        return False


def _original_prompt() -> str:
    from wan_full_book_episode import PROMPTS

    return PROMPTS[0]


def restore(reference: Path, new_audio: Path, destination: Path) -> None:
    if verified_scene_zero_visual(destination, reference, new_audio):
        print("SCENE_ZERO_VISUAL_PRESERVED_VERIFIED", VIDEO_SHA256)
        return
    if sha(reference) != IMAGE_SHA256:
        raise ValueError("scene zero Qwen keyframe differs; visual lock requires original frame")
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GitHub Actions read token missing")
    destination.mkdir(parents=True, exist_ok=True)
    source_root = destination.parent / "approved-scene-zero"
    subprocess.run(
        [
            "gh", "run", "download", str(SOURCE_RUN),
            "--repo", os.environ["GITHUB_REPOSITORY"],
            "--pattern", f"full-book-shot-0-{SOURCE_RUN}",
            "--dir", str(source_root),
        ],
        check=True,
    )
    sources = list(source_root.rglob("wan-s2v-candidate.mp4"))
    if len(sources) != 1:
        raise ValueError("expected exactly one user-accepted scene-zero video")
    source = sources[0]
    evidence = source.with_name("model-evidence.json")
    if not verified_wan_output(
        source, evidence, IMAGE_SHA256, OLD_AUDIO_SHA256, _original_prompt(), 735
    ) or sha(source) != VIDEO_SHA256:
        raise ValueError("approved original scene-zero clip failed provenance checks")
    import shutil

    shutil.copyfile(source, destination / source.name)
    shutil.copyfile(evidence, destination / evidence.name)
    reuse = {
        "source_run": SOURCE_RUN,
        "visual_sha256": VIDEO_SHA256,
        "original_voice_sha256": OLD_AUDIO_SHA256,
        "current_voice_sha256": sha(new_audio),
        "reuse_type": "VIDEO_ONLY_ORIGINAL_NARRATION_MUTED",
        "lip_sync_new_voice": "NOT_VERIFIED",
        "creative_review": "PENDING",
        "publish_eligible": False,
    }
    (destination / "visual-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    print("SCENE_ZERO_VISUAL_PRESERVED_VERIFIED", VIDEO_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.reference, args.audio, args.output)


if __name__ == "__main__":
    main()
