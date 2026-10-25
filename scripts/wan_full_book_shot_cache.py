"""Recover one immutable Wan episode shot from previous same-branch GH artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from wan_shared_shot_cache import list_prior_runs, sha, verified_wan_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, choices=range(4), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, default=12)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    shot = plan["shots"][args.shot]
    video = args.output / "wan-s2v-candidate.mp4"
    manifest = args.output / "model-evidence.json"
    values = (
        sha(args.reference.read_bytes()),
        sha(args.audio.read_bytes()),
        shot["prompt"],
        shot["seed"],
    )
    if verified_wan_output(video, manifest, *values):
        print("FULL_BOOK_SHOT_VERIFIED_LOCAL", args.shot)
        return
    repo = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ["GITHUB_REF_NAME"]
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN required for immutable cross-run cache")
    for run_id in list_prior_runs(repo, branch, args.max_runs):
        with tempfile.TemporaryDirectory(prefix="wan-book-shot-cache-") as root:
            folder = Path(root)
            result = subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    str(run_id),
                    "--repo",
                    repo,
                    "--pattern",
                    f"full-book-shot-{args.shot}-*",
                    "--dir",
                    str(folder),
                ],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if result.returncode:
                continue
            for candidate in folder.rglob("wan-s2v-candidate.mp4"):
                record = candidate.with_name("model-evidence.json")
                if verified_wan_output(candidate, record, *values):
                    args.output.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, video)
                    shutil.copy2(record, manifest)
                    print("FULL_BOOK_SHOT_RESTORED", args.shot, run_id)
                    return
    print("FULL_BOOK_SHOT_CACHE_MISS", args.shot)


if __name__ == "__main__":
    main()
