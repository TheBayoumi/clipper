"""Fail-closed creative review for generated book shots; no generation or publishing.

VisionProvider observations are evidence, not human approval. Frame-only review
cannot independently certify speech synchronization. Every accepted shot requires
a separate human decision bound to the immutable video and shot contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wan_shot_contract import CRITERION_VERSIONS

from clipper.providers.base import VisionProvider

SCHEMA = "clipper-book-shot-creative-review-v1"
OBSERVATION_SCHEMA = "clipper-book-shot-vision-observation-v1"
TASK = "book_shot_creative_review_v1"
MIN_CONFIDENCE = 0.85


class CreativeReviewError(ValueError):
    """A shot has not cleared content-bound creative and human review."""


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    if not path.is_file():
        raise CreativeReviewError(f"missing review input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def shot_fingerprint(shot: dict[str, Any], plan_fingerprint: str) -> str:
    return digest({"shot": shot, "plan_fingerprint": plan_fingerprint})


def expected_text(shot: dict[str, Any]) -> list[str]:
    requested = shot.get("required_on_screen_text")
    if requested is None:
        return ["ATOMIC HABITS", "JAMES CLEAR"] if shot.get("keyframe_name") == "book_intro" else []
    if (
        not isinstance(requested, list)
        or not all(isinstance(value, str) and value.strip() for value in requested)
    ):
        raise CreativeReviewError("invalid required on-screen text contract")
    return requested


def contract(
    shot: dict[str, Any],
    plan_fingerprint: str,
    video_sha256: str,
    reference_sha256: str,
    previous_video_sha256: str | None,
) -> dict[str, Any]:
    if (
        not isinstance(shot.get("id"), str)
        or not shot["id"]
        or not _hash(video_sha256)
        or not _hash(reference_sha256)
        or (previous_video_sha256 is not None and not _hash(previous_video_sha256))
    ):
        raise CreativeReviewError("invalid shot or content-hash review identity")
    narration = shot.get("narration_text", shot.get("narration", ""))
    action = shot.get("character_action", shot.get("prompt", ""))
    setting = shot.get("setting", shot.get("prompt", ""))
    return {
        "schema": SCHEMA,
        "shot_id": shot["id"],
        "video_sha256": video_sha256,
        "reference_sha256": reference_sha256,
        "shot_fingerprint": shot_fingerprint(shot, plan_fingerprint),
        "previous_video_sha256": previous_video_sha256,
        "criterion_versions": dict(CRITERION_VERSIONS),
        "narration_sha256": digest(narration),
        "action_sha256": digest(action),
        "setting_sha256": digest(setting),
        "text_policy": {"expected_text": expected_text(shot)},
        "publish_eligible": False,
    }


def _observation(
    raw: object, name: str, frame_hashes: set[str], expected_titles: list[str], has_previous: bool
) -> dict[str, Any]:
    pending = {
        "status": "PENDING_HUMAN",
        "confidence": 0.0,
        "evidence": "insufficient VLM evidence",
    }
    if not isinstance(raw, dict):
        return pending
    status, confidence = raw.get("status"), raw.get("confidence")
    evidence, frame = raw.get("evidence"), raw.get("frame_sha256")
    if (
        status not in {"PASS", "FAIL", "PENDING_HUMAN"}
        or type(confidence) not in (float, int)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or not isinstance(evidence, str)
        or not evidence.strip()
        or not isinstance(frame, str)
        or frame not in frame_hashes
    ):
        return pending
    if status == "PASS" and confidence < MIN_CONFIDENCE:
        status = "PENDING_HUMAN"
    if name == "speech_synchronization" and status == "PASS":
        status = "PENDING_HUMAN"  # Still frames cannot verify phoneme/audio alignment.
    if name == "setting_continuity" and has_previous and not raw.get("prior_shot_compared"):
        status = "PENDING_HUMAN"
    if name == "on_screen_titles" and status == "PASS":
        seen = raw.get("observed_text")
        if not isinstance(seen, list) or any(not isinstance(x, str) for x in seen):
            status = "PENDING_HUMAN"
        elif seen != expected_titles:
            status = "FAIL"  # Text absence is checkable and never a silent exemption.
    return {"status": status, "confidence": float(confidence), "evidence": evidence[:1000]}


def review_candidate(
    *,
    provider: VisionProvider,
    video: Path,
    reference: Path,
    frames: list[Path],
    shot: dict[str, Any],
    plan_fingerprint: str,
    previous_video_sha256: str | None = None,
    previous_frames: list[Path] | None = None,
) -> dict[str, Any]:
    """Record bounded structured VisionProvider evidence; never auto-approve."""
    if len(frames) < 3:
        raise CreativeReviewError("at least three sampled video frames required")
    current = [file_sha(frame) for frame in frames]
    preceding = [file_sha(frame) for frame in (previous_frames or [])]
    base = contract(
        shot, plan_fingerprint, file_sha(video), file_sha(reference), previous_video_sha256
    )
    context = {
        **base,
        "narration": shot.get("narration_text", shot.get("narration", "")),
        "action": shot.get("character_action", shot.get("prompt", "")),
        "setting": shot.get("setting", shot.get("prompt", "")),
        "current_frame_hashes": current,
        "previous_frame_hashes": preceding,
        "criteria": dict(CRITERION_VERSIONS),
    }
    result = provider.inspect(
        task=TASK, frames=[*(previous_frames or []), *frames], context=context
    )
    if (
        result.model.model_id != provider.identity.model_id
        or result.model.revision != provider.identity.revision
    ):
        raise CreativeReviewError("VisionProvider model identity mismatch")
    payload = result.value
    raw = (
        payload.get("criteria")
        if isinstance(payload, dict)
        and payload.get("schema") == OBSERVATION_SCHEMA
        and isinstance(payload.get("criteria"), dict)
        else {}
    )
    observations = {
        name: _observation(
            raw.get(name),
            name,
            set(current + preceding),
            base["text_policy"]["expected_text"],
            previous_video_sha256 is not None,
        )
        for name in CRITERION_VERSIONS
    }
    if result.degraded:
        observations = {
            name: {
                "status": "PENDING_HUMAN",
                "confidence": 0.0,
                "evidence": "degraded VisionProvider output",
            }
            for name in CRITERION_VERSIONS
        }
    return {
        **base,
        "sampled_frame_sha256": current,
        "previous_sampled_frame_sha256": preceding,
        "provider": {
            "name": type(provider).__name__,
            "model_id": result.model.model_id,
            "model_revision": result.model.revision,
        },
        "automated_criteria": observations,
        "status": (
            "FAILED"
            if any(x["status"] == "FAIL" for x in observations.values())
            else "PENDING_HUMAN"
        ),
        "human_approval": None,
    }


def verify_review(
    review: Path,
    *,
    video: Path,
    reference_sha256: str,
    shot: dict[str, Any],
    plan_fingerprint: str,
    previous_video_sha256: str | None,
    review_sha256: str | None = None,
) -> None:
    """Block assembly unless every criterion and the external human decision match."""
    if not review.is_file():
        raise CreativeReviewError(f"shot {shot.get('id')} missing human-reviewed evidence")
    if review_sha256 is not None and file_sha(review) != review_sha256:
        raise CreativeReviewError("creative review evidence SHA256 mismatch")
    if review.stat().st_size > 200_000:
        raise CreativeReviewError("creative review evidence exceeds bounded size")
    try:
        record = json.loads(review.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CreativeReviewError("unreadable creative review evidence") from exc
    if not isinstance(record, dict):
        raise CreativeReviewError("creative review record is not an object")
    base = contract(
        shot, plan_fingerprint, file_sha(video), reference_sha256, previous_video_sha256
    )
    if any(record.get(key) != value for key, value in base.items()):
        raise CreativeReviewError("creative review bound to stale or contradictory shot input")
    frames = record.get("sampled_frame_sha256")
    preceding = record.get("previous_sampled_frame_sha256")
    if (
        not isinstance(frames, list)
        or len(frames) < 3
        or not all(_hash(item) for item in frames)
        or not isinstance(preceding, list)
        or not all(_hash(item) for item in preceding)
        or (previous_video_sha256 is not None and not preceding)
    ):
        raise CreativeReviewError("missing verified sampled-frame hashes")
    provider = record.get("provider")
    if not isinstance(provider, dict) or any(
        not isinstance(provider.get(key), str) or not provider[key].strip()
        for key in ("name", "model_id", "model_revision")
    ):
        raise CreativeReviewError("review provider/model revision provenance missing")
    observations = record.get("automated_criteria")
    if not isinstance(observations, dict) or set(observations) != set(CRITERION_VERSIONS):
        raise CreativeReviewError("creative assessment criteria missing")
    for name, entry in observations.items():
        if not isinstance(entry, dict) or entry.get("status") not in {
            "PASS",
            "FAIL",
            "PENDING_HUMAN",
        }:
            raise CreativeReviewError(f"{name} has no valid automated assessment")
        if entry["status"] == "FAIL":
            raise CreativeReviewError(f"{name} failed: reject shot and hold for human review")
        confidence = entry.get("confidence")
        if (
            type(confidence) not in (float, int)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(entry.get("evidence"), str)
            or not entry["evidence"].strip()
            or (entry["status"] == "PASS" and confidence < MIN_CONFIDENCE)
            or (name == "speech_synchronization" and entry["status"] == "PASS")
        ):
            raise CreativeReviewError(f"{name} assessment is untrusted")
    approval = record.get("human_approval")
    if not isinstance(approval, dict) or approval.get("status") != "APPROVED":
        raise CreativeReviewError("mandatory human creative approval is PENDING")
    if (
        not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or not isinstance(approval.get("approved_at"), str)
        or not approval["approved_at"].strip()
        or approval.get("video_sha256") != base["video_sha256"]
        or approval.get("shot_fingerprint") != base["shot_fingerprint"]
        or approval.get("criterion_versions") != base["criterion_versions"]
        or approval.get("observation_sha256")
        != digest({key: value for key, value in record.items() if key != "human_approval"})
    ):
        raise CreativeReviewError("human approval does not bind this immutable reviewed shot")
    decisions = approval.get("criteria")
    if not isinstance(decisions, dict) or set(decisions) != set(CRITERION_VERSIONS):
        raise CreativeReviewError("human review missing required per-criterion decisions")
    for name, decision in decisions.items():
        if (
            not isinstance(decision, dict)
            or decision.get("status") != "PASS"
            or not isinstance(decision.get("evidence"), str)
            or not decision["evidence"].strip()
        ):
            raise CreativeReviewError(f"{name} has not passed mandatory human review")
