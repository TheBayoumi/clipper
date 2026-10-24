"""Build immutable, AI-video-specific shots from the approved Atomic Habits board.

Do not falsify 3D-rig transforms from pixels. Video generation consumes this
plan; the original GLB-only creative gate remains a separate backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SHOTS = ("hook", "book_intro", "lesson_example")
BACKENDS = ("s2v", "i2v", "i2v")
SCHEMA = "clipper-wan-storyboard-v1"


class StoryboardPlanError(ValueError):
    """The storyboard or character identity diverges from the approved source."""


def build_plan(board: dict[str, object], source_sha: str) -> dict[str, object]:
    if board.get("schema_version") != "storyboard-production-v1":
        raise StoryboardPlanError("unsupported storyboard contract")
    character = board.get("character")
    pilot = board.get("pilot")
    if not isinstance(character, dict) or not isinstance(pilot, dict):
        raise StoryboardPlanError("missing character or pilot")
    if character.get("character_id") != "page-to-life-host-cinematic-v1":
        raise StoryboardPlanError("unexpected identity")
    if character.get("outfit_lock") != "charcoal_hoodie_not_business_suit":
        raise StoryboardPlanError("identity outfit changed")
    if (pilot.get("duration_seconds"), pilot.get("resolution"), pilot.get("fps")) != (
        15,
        [1080, 1920],
        24,
    ):
        raise StoryboardPlanError("invalid approved pilot media contract")
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise StoryboardPlanError("source image requires pinned SHA256")
    original = board.get("pilot", {}).get("shots")
    if not isinstance(original, list) or len(original) != 3:
        raise StoryboardPlanError("expected three locked storyboard shots")
    seconds = 0
    plan = []
    for expected, backend, scene in zip(SHOTS, BACKENDS, original, strict=True):
        if not isinstance(scene, dict) or scene.get("id") != expected:
            raise StoryboardPlanError("shot identity/order differs from approved board")
        if scene.get("start") != seconds:
            raise StoryboardPlanError("shot timings have gaps or overlaps")
        seconds = scene.get("end")
        if not isinstance(seconds, int) or seconds <= scene["start"]:
            raise StoryboardPlanError("invalid shot timing")
        requirements = (
            "set",
            "action",
            "camera",
            "emotion",
            "narrative",
            "required_prop_interaction",
        )
        if not all(isinstance(scene.get(k), str) and scene[k].strip() for k in requirements):
            raise StoryboardPlanError("missing action-specific creative directions")
        prompt = (
            "Adult-facing cinematic stylized 3D animation; maintain exactly the approved "
            "same young adult male host: voluminous tousled dark-brown curly hair, "
            "black rectangular glasses, expressive brown eyes, charcoal hoodie "
            "with hood and drawstrings, dark trousers and white shoes. "
            f"Location: {scene['set']}. Camera: {scene['camera']}. "
            f"Emotion: {scene['emotion']}. Onscreen movement: {scene['action']}. "
            "Clean physical interaction, believable hand anatomy and persistent props. "
            "No lookalike substitutions, extra people, floating objects, or baked-in captions."
        )
        plan.append(
            {
                "id": expected,
                "renderer": backend,
                "start_seconds": scene["start"],
                "end_seconds": seconds,
                "voice_line": scene["narrative"],
                "required_prop_interaction": scene["required_prop_interaction"],
                "prompt": prompt,
                "status": "NOT_RENDERED",
                "visual_review": "PENDING",
                "publish_eligible": False,
            }
        )
    if seconds != 15:
        raise StoryboardPlanError("shot plan is not exactly 15 seconds")
    payload = {
        "schema": SCHEMA,
        "book": board.get("book"),
        "character_id": character["character_id"],
        "character_source_sha256": source_sha,
        "output": {"width": 1080, "height": 1920, "fps": 24, "duration_seconds": 15},
        "shots": plan,
        "publish_eligible": False,
        "accepted_video_sha256": None,
    }
    payload["plan_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storyboard", type=Path, default=Path("niches/atomic-habits-approved-storyboard.json")
    )
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(json.loads(args.storyboard.read_text()), args.source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(plan["plan_sha256"])


if __name__ == "__main__":
    main()
