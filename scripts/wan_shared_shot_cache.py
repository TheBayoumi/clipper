"""Verified shot-level reuse of immutable GitHub Actions artifacts across Modal workspaces.

No GPU, no workspace-specific Modal Volume reads. Only data from prior completed
runs of the same repository/branch and cinematic workflow is considered. Never
trust a filename without recomputing input/output hashes and model provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from wan_keyframe_contract import prepare, verified_local_keyframe

WAN_MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
WAN_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
WORKFLOW = "wan-cinematic-production.yml"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_wan_output(
    video: Path,
    manifest: Path,
    image_sha256: str,
    audio_sha256: str,
    prompt: str,
    seed: int,
) -> bool:
    """Reject stale, forged, truncated or differently prompted video artifacts."""
    if not video.is_file() or not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return False
        required = {
            "schema": "clipper-wan-s2v-modal-v1",
            "model_id": WAN_MODEL_ID,
            "model_revision": WAN_REVISION,
            "image_sha256": image_sha256,
            "audio_sha256": audio_sha256,
            "prompt_sha256": sha(prompt.encode("utf-8")),
            "seed": seed,
            "status": "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED",
            "publish_eligible": False,
            "creative_review": "PENDING",
        }
        if any(record.get(key) != value for key, value in required.items()):
            return False
        if not 300_000 <= video.stat().st_size <= 180_000_000:
            return False
        return sha(video.read_bytes()) == record.get("video_sha256")
    except (OSError, ValueError, TypeError, KeyError):
        return False


def local_keyframes_complete(args: argparse.Namespace) -> bool:
    root = Path(args.output) / "keyframes"
    shots = [x.strip() for x in args.shot_ids.split(",")]
    if not shots or len(set(shots)) != len(shots):
        raise ValueError("invalid shot IDs")
    return all(
        verified_local_keyframe(
            root / f"{shot}.png",
            root / f"{shot}.json",
            prepare(Path(args.reference), Path(args.storyboard), shot),
        )
        for shot in shots
    )


def local_wan_complete(args: argparse.Namespace) -> bool:
    root = Path(args.output)
    return verified_wan_output(
        root / "wan-s2v-candidate.mp4",
        root / "model-evidence.json",
        sha(Path(args.reference).read_bytes()),
        sha(Path(args.audio).read_bytes()),
        args.prompt,
        args.seed,
    )


def list_prior_runs(repo: str, branch: str, limit: int) -> list[int]:
    """Only completed runs, in this workflow and branch, may be reused."""
    command = [
        "gh", "run", "list", "--repo", repo, "--workflow", WORKFLOW,
        "--branch", branch, "--limit", str(limit),
        "--json", "databaseId,status",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=40)
    runs = json.loads(result.stdout)
    this_run = os.environ.get("GITHUB_RUN_ID", "")
    return [
        int(run["databaseId"])
        for run in runs
        if run.get("status") == "completed" and str(run.get("databaseId")) != this_run
    ]


def copy_verified_keyframes(archive: Path, args: argparse.Namespace) -> int:
    key_root = Path(args.output) / "keyframes"
    key_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for shot in [x.strip() for x in args.shot_ids.split(",")]:
        request = prepare(Path(args.reference), Path(args.storyboard), shot)
        frame = key_root / f"{shot}.png"
        manifest = key_root / f"{shot}.json"
        if verified_local_keyframe(frame, manifest, request):
            continue
        for source in archive.rglob(f"{shot}.png"):
            if source.parent.name != "keyframes":
                continue
            candidate_manifest = source.with_suffix(".json")
            if verified_local_keyframe(source, candidate_manifest, request):
                shutil.copy2(source, frame)
                shutil.copy2(candidate_manifest, manifest)
                copied += 1
                break
    return copied


def copy_verified_wan(archive: Path, args: argparse.Namespace) -> int:
    root = Path(args.output)
    if local_wan_complete(args):
        return 0
    image_sha256 = sha(Path(args.reference).read_bytes())
    audio_sha256 = sha(Path(args.audio).read_bytes())
    for source in archive.rglob("wan-s2v-candidate.mp4"):
        candidate_manifest = source.with_name("model-evidence.json")
        if verified_wan_output(
            source, candidate_manifest, image_sha256, audio_sha256, args.prompt, args.seed
        ):
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, root / "wan-s2v-candidate.mp4")
            shutil.copy2(candidate_manifest, root / "model-evidence.json")
            return 1
    return 0


def restore(args: argparse.Namespace) -> None:
    """Restore independently validated completed shots, including from failed runs."""
    if args.stage == "keyframes" and local_keyframes_complete(args):
        print("ALL_KEYFRAMES_ALREADY_VERIFIED")
        return
    if args.stage == "speaking" and local_wan_complete(args):
        print("WAN_VIDEO_ALREADY_VERIFIED")
        return
    repo = os.environ.get("GITHUB_REPOSITORY")
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    if not repo or not branch or not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GitHub repository, branch, and GH_TOKEN required for cross-run reuse")
    for run_id in list_prior_runs(repo, branch, args.max_runs):
        with tempfile.TemporaryDirectory(prefix="clipper-shot-cache-") as temp:
            archive = Path(temp)
            command = [
                "gh", "run", "download", str(run_id), "--repo", repo,
                "--pattern", "clipper-cinematic-*", "--dir", str(archive),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            if result.returncode:
                print(f"SHOT_CACHE_ARTIFACT_UNAVAILABLE run={run_id}")
                continue
            if args.stage == "keyframes":
                count = copy_verified_keyframes(archive, args)
                print(f"SHOT_CACHE_RESTORED_KEYFRAMES run={run_id} count={count}")
                if local_keyframes_complete(args):
                    return
            else:
                count = copy_verified_wan(archive, args)
                print(f"SHOT_CACHE_RESTORED_WAN run={run_id} count={count}")
                if local_wan_complete(args):
                    return
    print(f"SHOT_CACHE_MISS stage={args.stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("restore", "check"))
    parser.add_argument("--stage", required=True, choices=("keyframes", "speaking"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--storyboard", default="niches/atomic-habits-approved-storyboard.json")
    parser.add_argument("--shot-ids", default="hook")
    parser.add_argument("--audio", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--max-runs", type=int, default=15)
    args = parser.parse_args()
    if args.stage == "speaking" and (not args.audio or not args.prompt):
        parser.error("speaking stage requires --audio and --prompt")
    if args.operation == "restore":
        restore(args)
    else:
        ok = (
            local_keyframes_complete(args)
            if args.stage == "keyframes"
            else local_wan_complete(args)
        )
        print(f"SHOT_CACHE_CHECK stage={args.stage} complete={ok}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
