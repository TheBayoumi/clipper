"""Fail-closed storyboard-to-video acceptance for the Atomic Habits pilot.

A passing codec check is not evidence of a visually approved video.
The precise design board is an image reference; it is NOT a rigged GLB.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

PILOT_SHOTS = ("hook", "book_intro", "lesson_example")
STOCK_ASSET_TOKENS = (
    "male_casualsuit01",
    "business-male-01.glb",
    "generic_talking_head",
    "static_card",
)


class CreativeGateError(ValueError):
    """Required creative artifact, animation, or evidence is missing."""


def _required(value: object, kind: type, label: str) -> Any:
    if not isinstance(value, kind):
        raise CreativeGateError(f"{label} must be a {kind.__name__}")
    return value


def read_contract(path: Path) -> dict[str, Any]:
    data = _required(json.loads(path.read_text(encoding="utf-8")), dict, "contract")
    if data.get("schema_version") != "storyboard-production-v1":
        raise CreativeGateError("unexpected storyboard contract version")
    character = _required(data.get("character"), dict, "character")
    if character.get("outfit_lock") != "charcoal_hoodie_not_business_suit":
        raise CreativeGateError("approved dark-hoodie character identity was changed")
    for trait in ("hair", "glasses", "face", "clothing"):
        if not character.get(trait):
            raise CreativeGateError(f"missing character feature: {trait}")
    if character.get("status") == "approved_rig" and not character.get("approved_glb_sha256"):
        raise CreativeGateError("approved character requires a pinned asset hash")
    references = _required(data.get("visual_references"), list, "visual_references")
    if len(references) < 2 or any(len(str(ref.get("sha256", ""))) != 64 for ref in references):
        raise CreativeGateError("approved visual references require immutable SHA256s")
    pilot = _required(data.get("pilot"), dict, "pilot")
    if pilot.get("duration_seconds") != 15:
        raise CreativeGateError("pilot must be 15 seconds")
    if pilot.get("resolution") != [1080, 1920] or pilot.get("fps") != 24:
        raise CreativeGateError("pilot video must be 1080x1920 at 24 fps")
    shots = _required(pilot.get("shots"), list, "pilot.shots")
    if tuple(s.get("id") for s in shots) != PILOT_SHOTS:
        raise CreativeGateError("pilot shot IDs/order cannot be silently replaced")
    prev = 0
    for shot in shots:
        if shot.get("start") != prev or shot.get("end", -1) <= prev:
            raise CreativeGateError("shot timing gap or overlap")
        prev = shot["end"]
        if not shot.get("meaningful_motion") or not shot.get("required_prop_interaction"):
            raise CreativeGateError("every scene must show physical acting with props")
        for key in ("set", "camera", "emotion", "action", "narrative"):
            if not shot.get(key):
                raise CreativeGateError(f"{shot['id']} missing {key}")
    if prev != 15:
        raise CreativeGateError("the pilot must cover the complete 15-second timeline")
    if len({s["set"] for s in shots}) != len(shots):
        raise CreativeGateError("one repetitive studio backdrop is not accepted")
    return data


def verify_physical_contact(
    expected: dict[str, Any], actual: dict[str, Any], master_hash: str
) -> None:
    """Check evaluated skeletal/prop transforms, not a claimed interaction flag.

    Distances are world-space meters; this is numeric validation, not visual approval.
    """
    proof = _required(actual.get("physical_contact_evidence"), dict, "physical evidence")
    if proof.get("evaluation_source") != "evaluated_3d_scene":
        raise CreativeGateError("physical contact was not measured from evaluated 3D transforms")
    if proof.get("character_sha256") != master_hash:
        raise CreativeGateError("physical contact evidence belongs to another character")
    if proof.get("prop_id") != expected["required_prop_interaction"]:
        raise CreativeGateError("wrong physical interaction prop")

    def vec(key: str) -> tuple[float, float, float]:
        value = _required(proof.get(key), list, key)
        if len(value) != 3 or any(type(n) not in (float, int) for n in value):
            raise CreativeGateError(f"{key} must be a numeric 3D position")
        if any(not math.isfinite(float(v)) for v in value):
            raise CreativeGateError(f"{key} contains non-finite coordinates")
        return tuple(float(v) for v in value)

    start_hand, contact_hand = vec("hand_start"), vec("hand_at_contact")
    start_prop, contact_prop = vec("prop_start"), vec("prop_at_contact")
    end_prop = vec("prop_end")
    initial_distance = math.dist(start_hand, start_prop)
    contact_distance = math.dist(contact_hand, contact_prop)
    if contact_distance > 0.08 or initial_distance - contact_distance < 0.05:
        raise CreativeGateError("hand never made a measurable reach/contact with the prop")
    if expected["id"] != "hook" and math.dist(start_prop, end_prop) < 0.04:
        raise CreativeGateError("book/notebook never underwent a real scene transformation")


def audit_render(
    contract: dict[str, Any],
    manifest_path: Path,
    video: Path,
    master_glb: Path | None = None,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise CreativeGateError(
            "missing storyboard-specific render manifest; generic rig proof is insufficient"
        )
    manifest = _required(
        json.loads(manifest_path.read_text(encoding="utf-8")), dict, "render manifest"
    )
    if manifest.get("storyboard_schema") != "storyboard-production-v1":
        raise CreativeGateError("renderer did not consume approved storyboard contract")
    character = contract["character"]
    if manifest.get("character_id") != character["character_id"]:
        raise CreativeGateError("different character from the approved storyboards")
    asset_hash = str(manifest.get("master_character_sha256", ""))
    if len(asset_hash) != 64 or asset_hash != character.get("approved_glb_sha256"):
        raise CreativeGateError("rigged master asset is missing or unapproved")
    if character.get("status") != "approved_rig":
        raise CreativeGateError("character still lacks an independently approved master rig")
    if master_glb is None or not master_glb.is_file():
        raise CreativeGateError("approved master GLB must be provided for structural inspection")
    source = Path(__file__).with_name("character_rig_contract.py")
    spec = importlib.util.spec_from_file_location("clipper_approved_rig_runtime", source)
    if spec is None or spec.loader is None:
        raise CreativeGateError("versioned GLB rig contract is missing")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    try:
        rig_evidence = verifier.audit_rig(master_glb, asset_hash)
    except (verifier.RigContractError, OSError) as exc:
        raise CreativeGateError(f"master GLB failed real rig inspection: {exc}") from exc
    if any(token in json.dumps(manifest).lower() for token in STOCK_ASSET_TOKENS):
        raise CreativeGateError("stock-looking stand-in avatar or generic motion detected")
    shots = _required(manifest.get("shots"), list, "rendered shots")
    if len(shots) != 3:
        raise CreativeGateError("three individual pilot shots must be rendered")
    for expected, actual in zip(contract["pilot"]["shots"], shots, strict=True):
        for field in ("id", "start", "end", "required_prop_interaction"):
            if actual.get(field) != expected[field]:
                raise CreativeGateError(f"{expected['id']} is missing approved {field}")
        if float(actual.get("moving_joint_delta", 0)) <= 0.02:
            raise CreativeGateError(f"{expected['id']} has no real articulated movement")
        if not actual.get("prop_interaction_verified") or not actual.get("set_approved"):
            raise CreativeGateError(f"{expected['id']} did not enact the storyboard")
        if actual.get("character_id") != character["character_id"]:
            raise CreativeGateError("character identity varies across shots")
        verify_physical_contact(expected, actual, asset_hash)
    if manifest.get("audio") != "Kokoro" or not manifest.get("voice_timing_verified"):
        raise CreativeGateError("Kokoro timing evidence is missing")
    if manifest.get("viseme_blendshapes_verified") is not True:
        raise CreativeGateError("jaw-only timing is not an approved facial-viseme rig")
    if not video.is_file():
        raise CreativeGateError("missing final 15-second MP4")
    actual_sha256 = hashlib.sha256(video.read_bytes()).hexdigest()
    if manifest.get("mp4_sha256") != actual_sha256:
        raise CreativeGateError("rendered MP4 hash does not match review evidence")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    media = json.loads(probe.stdout)
    duration = float(media["format"]["duration"])
    if not 14.8 <= duration <= 15.1:
        raise CreativeGateError(f"expected a 15s pilot; got {duration:.3f}s")
    streams = media["streams"]
    if not any(
        stream.get("codec_type") == "video"
        and (stream.get("width"), stream.get("height")) == (1080, 1920)
        for stream in streams
    ):
        raise CreativeGateError("1080x1920 video missing")
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise CreativeGateError("Kokoro soundtrack missing")
    for stream in streams:
        if stream.get("codec_type") == "video":
            try:
                fps = Fraction(stream.get("r_frame_rate", "0"))
            except (ValueError, ZeroDivisionError) as exc:
                raise CreativeGateError("invalid video frame rate") from exc
            if fps != 24:
                raise CreativeGateError(f"storyboard requires native 24 fps; got {fps}")
            count = stream.get("nb_frames")
            if count is not None and int(count) != 360:
                raise CreativeGateError("15-second video does not contain 360 frames")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    return {
        "technical_storyboard_gate": "PASS",
        "master_rig_structure": rig_evidence["structure_gate"],
        "master_rig_sha256": rig_evidence["glb_sha256"],
        "video_sha256": actual_sha256,
        "duration_seconds": duration,
        "human_visual_approval": "PENDING",
        "publish_eligible": False,
        "caveat": "Structured shot metadata does not substitute for human visual review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("niches/atomic-habits-approved-storyboard.json"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--master-glb", type=Path)
    args = parser.parse_args()
    contract = read_contract(args.contract)
    if (args.manifest is None) != (args.video is None):
        parser.error("--manifest and --video must be supplied together")
    if args.manifest is None:
        print(
            json.dumps(
                {
                    "storyboard_contract": "VALID",
                    "character_asset_status": contract["character"]["status"],
                    "pilot_shots": [shot["id"] for shot in contract["pilot"]["shots"]],
                    "creative_acceptance": "NOT_YET_VERIFIED",
                    "publish_eligible": False,
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(audit_render(contract, args.manifest, args.video, args.master_glb), indent=2)
        )


if __name__ == "__main__":
    main()
