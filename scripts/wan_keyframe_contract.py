"""Reproducible open-model keyframe requests from Clipper's approved storyboard.

Pure CPU planning; actual reference-conditioned image inference runs on Modal.
Generated keyframes are candidates and never constitute creative approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wan_shot_contract import inspect_image
from wan_storyboard_plan import build_plan

SCHEMA = "clipper-cinematic-keyframe-v1"
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
MODEL_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
SHOTS = ("hook", "book_intro", "lesson_example")
BASE_SEED = 724

SCENE_INSTRUCTIONS = {
    "hook": (
        "Vertical 9:16 medium close-up, chest-and-head prominent, speaking to camera. "
        "Seated at a real desk in a warmly lit lived-in evening bedroom, lamp and shelves "
        "visible in softly blurred background; smartphone alarm on the desk near one hand. "
        "Neutral-to-curious expressive face with lips slightly parted mid-sentence. "
        "One natural hand poised close to the phone; believable finger anatomy. "
        "Use a three-quarter eye-level cinematic camera, not a plain gray backdrop."
    ),
    "book_intro": (
        "Vertical 9:16 cinematic medium shot in a warm book-lined creative studio. "
        "The SAME host has one hand placed naturally on a generic unbranded physical "
        "hardcover book on the desk, preparing to lift it; the other hand rests casually. "
        "Camera has clean line of sight to face, both hands and book. No printed title; "
        "book title is added later by Clipper as a separate overlay."
    ),
    "lesson_example": (
        "Vertical 9:16 intimate over-the-shoulder desk shot in the same bedroom. "
        "The SAME host is carefully repositioning an open notebook beside the phone "
        "with physically plausible hand contact, focusing on making the first habit easy. "
        "Maintain exact outfit and styling; natural hands, consistent furniture and lighting."
    ),
}


class KeyframeContractError(ValueError):
    """Source identity or requested cinematic shot is not valid."""


def prepare(reference: Path, storyboard: Path, shot_id: str, seed: int = BASE_SEED) -> dict:
    if shot_id not in SHOTS:
        raise KeyframeContractError("shot must belong to the approved three-shot storyboard")
    if not 0 <= seed < 2**31:
        raise KeyframeContractError("invalid deterministic seed")
    image = inspect_image(reference)
    board = json.loads(storyboard.read_text(encoding="utf-8"))
    plan = build_plan(board, image["sha256"])
    shot = next(x for x in plan["shots"] if x["id"] == shot_id)
    prompt = (
        "REFERENCE-IMAGE CHARACTER EDIT: Use the attached reference image as the "
        "ONLY identity source for our recurring young adult male animated presenter. "
        "Preserve exact recognizable facial identity, expressive large brown eyes, "
        "black rectangular eyeglasses, voluminous dark-brown wavy curly hair, "
        "charcoal hooded sweatshirt with drawstrings and matching dark trousers. "
        "Create an entirely professional polished 3D-styled cinematic animation film "
        "frame with clear facial detail, realistically textured fabric, appealing "
        "stylized proportions, soft key/rim lighting and dimensional depth. "
        "Do not make the character photorealistic or change face, glasses or outfit. "
        "This is one video SHOT START FRAME, not a character reference sheet. "
        f"{SCENE_INSTRUCTIONS[shot_id]} "
        f"Story action: {shot['prompt']}. "
        "Professional film frame, crisp subject, physically plausible hands and props, "
        "no captions, no watermarks, no labels, no split screens, no extra people."
    )
    identity = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_reference": image,
        "source_image_sha256": image["sha256"],
        "storyboard_plan_sha256": plan["plan_sha256"],
        "shot_id": shot_id,
        "prompt": prompt,
        "seed": seed,
        "render_width": 832,
        "render_height": 1472,
        "steps": 40,
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity.update(
        stage="KEYFRAME_REQUEST_NOT_RENDERED",
        creative_review="PENDING",
        publish_eligible=False,
    )
    return identity


def verified_local_keyframe(frame: Path, manifest: Path, request: dict) -> bool:
    """Only reuse a completed frame whose provenance and actual bytes match."""
    if not frame.is_file() or not manifest.is_file():
        return False
    try:
        evidence = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            return False
        for key in ("fingerprint", "source_image_sha256", "model_id", "model_revision"):
            if evidence.get(key) != request.get(key):
                return False
        if (
            evidence.get("status") != "FRAME_GENERATED_TECHNICAL_CANDIDATE"
            or evidence.get("publish_eligible") is not False
            or evidence.get("creative_review") != "PENDING"
        ):
            return False
        if not 100_000 <= frame.stat().st_size <= 20_000_000:
            return False
        inspected = inspect_image(frame)
        return (
            inspected["sha256"] == evidence.get("image_sha256")
            and inspected["width"] == request.get("render_width")
            and inspected["height"] == request.get("render_height")
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--storyboard",
        type=Path,
        default=Path("niches/atomic-habits-approved-storyboard.json"),
    )
    parser.add_argument("--shot-id", choices=SHOTS, required=True)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.reference, args.storyboard, args.shot_id, args.seed)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["fingerprint"])


if __name__ == "__main__":
    main()
