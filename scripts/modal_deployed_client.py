"""CPU-side Modal client: invoke deployed, versioned apps without modal run.

The active GitHub workflow provisions/deploys a function only for a missing
code-addressed version. All GPU work is remote; never import GPU libraries here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

from wan_keyframe_contract import prepare, verified_local_keyframe

QWEN_FUNCTION = "generate_keyframe"
WAN_FUNCTION = "generate_s2v"
WAN_MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
WAN_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_keyframes(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.reference).read_bytes()
    shots = [shot.strip() for shot in args.shot_ids.split(",")]
    if not shots or len(shots) != len(set(shots)):
        raise ValueError("empty or repeated keyframe request")
    function = None
    for shot in shots:
        request = prepare(Path(args.reference), Path(args.storyboard), shot)
        png_file = output / f"{shot}.png"
        manifest = png_file.with_suffix(".json")
        if verified_local_keyframe(png_file, manifest, request):
            print(json.dumps({"shot": shot, "cache": "REUSED_LOCAL_VERIFIED"}))
            continue
        if function is None:
            function = modal.Function.from_name(args.app, QWEN_FUNCTION)
        evidence: dict[str, Any] = {**request, "status": "INCOMPLETE"}
        try:
            response = function.remote(source, request)
            png = response.pop("png")
            png_file.write_bytes(png)
            if sha(png) != response["image_sha256"]:
                raise RuntimeError("Modal returned keyframe SHA mismatch")
            evidence.update(
                response,
                status="FRAME_GENERATED_TECHNICAL_CANDIDATE",
                stage="KEYFRAME_AWAITING_VISUAL_REVIEW",
                creative_review="PENDING",
                publish_eligible=False,
                output_path=str(png_file),
            )
        except Exception as exc:
            evidence.update(
                status="FRAME_GENERATION_FAILED",
                error_type=type(exc).__name__,
                error_reason=str(exc)[:1000],
            )
            raise
        finally:
            manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"shot": shot, "status": evidence["status"]}))


def render_speaking(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(args.reference).read_bytes()
    audio = Path(args.audio).read_bytes()
    source_sha, audio_sha = sha(reference), sha(audio)
    if not 0 < len(reference) <= 20_000_000 or not 40_000 <= len(audio) <= 25_000_000:
        raise ValueError("invalid source or narration bytes")
    if not 0 <= args.seed < 2**31 or not args.prompt.strip():
        raise ValueError("invalid Wan prompt or seed")

    # Cache verification is also used by the restore step before Modal starts.
    from wan_shared_shot_cache import verified_wan_output

    candidate = output / "wan-s2v-candidate.mp4"
    manifest = output / "model-evidence.json"
    if verified_wan_output(
        candidate, manifest, source_sha, audio_sha, args.prompt, args.seed
    ):
        print("WAN_SHOT_CACHE_HIT_LOCAL_VERIFIED")
        return

    evidence: dict[str, Any] = {
        "schema": "clipper-wan-s2v-modal-v1",
        "model_id": WAN_MODEL_ID,
        "model_revision": WAN_REVISION,
        "creative_review": "PENDING",
        "publish_eligible": False,
        "status": "INCOMPLETE",
    }
    try:
        function = modal.Function.from_name(args.app, WAN_FUNCTION)
        response = function.remote(
            reference, audio, source_sha, audio_sha, args.prompt, args.seed
        )
        video = response.pop("video")
        if sha(video) != response["video_sha256"]:
            raise RuntimeError("Modal returned video SHA mismatch")
        if response.get("model_id") != WAN_MODEL_ID or response.get("model_revision") != WAN_REVISION:
            raise RuntimeError("Wan model provenance mismatch")
        candidate.write_bytes(video)
        evidence.update(response, status="CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED")
    except Exception as exc:
        evidence.update(
            status="GENERATION_FAILED",
            error_type=type(exc).__name__,
            error_reason=str(exc)[:1200],
        )
        raise
    finally:
        manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": evidence["status"], "model": WAN_MODEL_ID}), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    keys = sub.add_parser("keyframes")
    keys.add_argument("--app", required=True)
    keys.add_argument("--reference", required=True)
    keys.add_argument("--storyboard", required=True)
    keys.add_argument("--shot-ids", required=True)
    keys.add_argument("--output", required=True)
    speech = sub.add_parser("speaking")
    speech.add_argument("--app", required=True)
    speech.add_argument("--reference", required=True)
    speech.add_argument("--audio", required=True)
    speech.add_argument("--output", required=True)
    speech.add_argument("--prompt", required=True)
    speech.add_argument("--seed", type=int, default=724)
    args = parser.parse_args()
    if args.mode == "keyframes":
        render_keyframes(args)
    else:
        render_speaking(args)


if __name__ == "__main__":
    main()
