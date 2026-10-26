"""Build episode-two clean master exclusively from source-bound cached Wan shots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import book_audio_timeline as timeline
import book_timeline_assemble as editor
from wan_shared_shot_cache import verified_wan_output


def build_assets(root: Path) -> dict[str, Any]:
    source = root / "source"
    edl = json.loads((source / "audio-locked-edl.json").read_text(encoding="utf-8"))
    renders = json.loads((source / "render-jobs.json").read_text(encoding="utf-8"))
    if renders["edl_fingerprint"] != edl["fingerprint"]:
        raise editor.AssemblyError("episode-two video requests belong to a different EDL")
    if len(edl["shots"]) != len(renders["jobs"]):
        raise editor.AssemblyError("partial video render cannot be assembled")
    results: list[dict[str, Any]] = []
    for shot, job in zip(edl["shots"], renders["jobs"], strict=True):
        if shot["id"] != job["id"]:
            raise editor.AssemblyError("render source and audio lock disagree")
        index = job["index"]
        if type(index) is not int or index < 0:
            raise editor.AssemblyError("render index must be a nonnegative integer")
        folder = root / "rendered" / f"shot-{index:02d}"
        video = folder / "wan-s2v-candidate.mp4"
        evidence = folder / "model-evidence.json"
        frame = source / "frames" / job["frame"]
        audio = source / job["audio"]
        if (
            not frame.is_file()
            or not audio.is_file()
            or job["audio_sha256"] != timeline.file_sha(audio)
            or not verified_wan_output(
                video,
                evidence,
                timeline.file_sha(frame),
                timeline.file_sha(audio),
                job["prompt"],
                job["seed"],
            )
        ):
            raise editor.AssemblyError(
                f"shot {shot['id']} is not a verified current-input Wan asset"
            )
        results.append(
            {
                "shot_id": shot["id"],
                "path": str(video),
                "sha256": timeline.file_sha(video),
                "source_start_frame": 0,
                "evidence_path": str(evidence),
                "evidence_sha256": timeline.file_sha(evidence),
            }
        )
    return {
        "schema": editor.ASSETS_SCHEMA,
        "edl_fingerprint": edl["fingerprint"],
        "assets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    source = args.root / "source"
    edl = json.loads((source / "audio-locked-edl.json").read_text(encoding="utf-8"))
    assets = build_assets(args.root)
    out = args.root / "final"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "verified-shot-assets.json"
    manifest.write_text(json.dumps(assets, indent=2) + "\n", encoding="utf-8")
    result = editor.assemble(edl, source / "full-kokoro.wav", assets, out)
    print(
        json.dumps(
            {
                "episode_id": result["episode_id"],
                "mp4": str(out / "clean-master.mp4"),
                "duration_seconds": result["duration_seconds"],
                "shot_count": result["shot_count"],
                "sha256": result["video_sha256"],
                "publish_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
