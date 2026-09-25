"""Reusable source-restricted visual-state montage planning and contract checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import media_contract as media
from .profiles import CampaignProfile

SCHEMA_VERSION = 2


class MontageRejection(ValueError):
    """Structured candidate rejection, never a generic story-quality boolean."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def to_json(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def profile_sha256(profile: CampaignProfile) -> str:
    payload = json.dumps(profile.config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rate(profile: CampaignProfile) -> Fraction:
    try:
        value = Fraction(str(profile.config["output"]["fps"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise MontageRejection("invalid_fps", "fps must be a positive rational") from exc
    if value <= 0:
        raise MontageRejection("invalid_fps", "fps must be positive")
    return value


def _frames(seconds: int | float, fps: Fraction) -> int:
    count = Fraction(str(seconds)) * fps
    if count.denominator != 1:
        raise MontageRejection("off_frame_grid", f"{seconds}s is not frame-exact at {fps}")
    return int(count)


def _thumbnail(source: Path, frame: int, fps: Fraction) -> npt.NDArray[np.uint8]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{frame}),scale=96:54:flags=area,format=gray",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode or len(completed.stdout) != 96 * 54:
        raise MontageRejection(
            "visual_evidence_unavailable",
            f"unable to decode source frame {frame}: "
            f"{completed.stderr.decode(errors='replace')[-350:]}",
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(54, 96)


def build_plan(
    source: Path,
    profile: CampaignProfile,
    certified_manifest: dict[str, Any] | None = None,
    comparison_mode: str = "wipe",
    approved_text_index: int | None = None,
) -> dict[str, Any]:
    """Construct a deterministic, visually justified legal montage before rendering."""
    config = profile.config
    if config.get("kind") != "announcement":
        raise MontageRejection(
            "wrong_profile", "announcement montage requires announcement profile"
        )
    if not profile.capability("announcement_montage").get("enabled", False):
        raise MontageRejection("capability_disabled", "announcement montage is disabled")
    if not source.is_file():
        raise MontageRejection("missing_source", str(source))

    contract = media.inspect_source(source, verify_timeline=True)
    fps = rate(profile)
    if contract.video.timing.nominal_rate != fps:
        raise MontageRejection(
            "source_fps_mismatch", f"{contract.video.timing.nominal_rate} != {fps}"
        )
    output = config["output"]
    if (contract.video.width, contract.video.height) != (
        int(output["width"]),
        int(output["height"]),
    ):
        raise MontageRejection("source_geometry_mismatch", "source geometry differs from profile")
    if (contract.audio.sample_rate, contract.audio.channels) != (
        int(output["audio_sample_rate"]),
        int(output["audio_channels"]),
    ):
        raise MontageRejection("source_audio_mismatch", "source audio differs from profile")

    digest = sha256(source)
    certified = certified_manifest is not None
    if certified_manifest is not None:
        if certified_manifest.get("sha256") != digest:
            raise MontageRejection("source_hash_mismatch", "source certification does not match")
        if certified_manifest.get("derivative_type") != "source":
            raise MontageRejection("source_proxy", "proxy material cannot be campaign-qualified")
        if certified_manifest.get("review_url") != profile.source_review_url:
            raise MontageRejection("wrong_review", "source is not from the approved review")
    filename = (
        str(certified_manifest.get("file_name") or certified_manifest.get("title") or "")
        if certified_manifest is not None
        else source.name
    )
    states = config["editorial"]["verified_visual_states"].get(filename)
    if not isinstance(states, dict):
        raise MontageRejection(
            "uncalibrated_source", f"no reviewed before/after frames for {filename!r}"
        )

    frame_count = int(media.video_profile(source, count_frames=True)["frame_count"])
    if frame_count != int(states["source_frames"]):
        raise MontageRejection(
            "source_frames_changed",
            f"expected {states['source_frames']}, decoded {frame_count}; recalibration required",
        )
    before, after = int(states["before_frame"]), int(states["after_frame"])
    if not 0 <= before < after < frame_count:
        raise MontageRejection("state_order_invalid", "verified frames are not ordered")
    roi = config["editorial"].get("visual_state_roi", [0.0, 0.0, 1.0, 1.0])
    if (
        not isinstance(roi, list)
        or len(roi) != 4
        or not 0 <= roi[0] < roi[2] <= 1
        or not 0 <= roi[1] < roi[3] <= 1
    ):
        raise MontageRejection(
            "roi_invalid", "visual state ROI must be a valid normalized rectangle"
        )
    first = _thumbnail(source, before, fps).astype(np.float32)
    second = _thumbnail(source, after, fps).astype(np.float32)
    x0, y0, x1, y1 = (
        int(roi[0] * 96),
        int(roi[1] * 54),
        max(1, int(roi[2] * 96)),
        max(1, int(roi[3] * 54)),
    )
    delta = np.abs(first[y0:y1, x0:x1] - second[y0:y1, x0:x1]) / 255.0
    visual_difference = float(np.mean(delta))
    changed_fraction = float(np.mean(delta > 0.06))
    threshold = float(config["editorial"].get("minimum_state_difference", 0.08))
    fraction_threshold = float(config["editorial"].get("minimum_changed_pixel_fraction", 0.35))
    if visual_difference < threshold or changed_fraction < fraction_threshold:
        raise MontageRejection(
            "no_visual_state_change",
            f"measured change {visual_difference:.4f}/{threshold:.4f}; "
            f"changed pixels {changed_fraction:.4f}/{fraction_threshold:.4f}",
        )

    editorial = config["editorial"]
    target = _frames(editorial["preferred_output_seconds"], fps)
    low = _frames(editorial["minimum_output_seconds"], fps)
    high = _frames(editorial["maximum_output_seconds"], fps)
    if not low <= target <= high:
        raise MontageRejection("invalid_duration_config", "preferred duration out of bounds")
    allowed_modes = editorial.get("comparison_modes", ["wipe"])
    if comparison_mode not in allowed_modes:
        raise MontageRejection(
            "invalid_comparison_mode", f"{comparison_mode} not in {allowed_modes}"
        )
    comparison_frames = _frames(editorial["comparison_seconds"], fps)
    hook = {
        "start_frame": int(states["hook_start_frame"]),
        "frames": int(states["hook_frames"]),
    }
    shots = [dict(shot) for shot in states["ending_shots"]]
    if not shots or [shot["state"] for shot in shots] != ["before", "after", "before", "after"]:
        raise MontageRejection(
            "invalid_switch_pattern", "ending requires calibrated A/B/A/B payoff"
        )
    for shot in shots:
        start_frame = int(shot["start_frame"])
        shot_frames = int(shot["frames"])
        if shot_frames <= 0 or start_frame < 0 or start_frame + shot_frames > frame_count:
            raise MontageRejection("ending_window_invalid", "ending shot lies outside source")
        if shot["state"] == "before" and start_frame + shot_frames >= after:
            raise MontageRejection(
                "ending_window_invalid", "before shot includes after-state frames"
            )
        if shot["state"] == "after" and start_frame < after:
            raise MontageRejection(
                "ending_window_invalid", "after shot precedes verified after-state"
            )
    if hook["frames"] <= 0 or not 0 <= hook["start_frame"] < frame_count:
        raise MontageRejection("hook_window_invalid", "transformation hook is outside source")
    if hook["start_frame"] + hook["frames"] > frame_count:
        raise MontageRejection("hook_window_invalid", "transformation hook overruns source")
    ending_frames = sum(int(shot["frames"]) for shot in shots)
    if hook["frames"] + frame_count + comparison_frames + ending_frames != target:
        raise MontageRejection(
            "no_admissible_montage",
            "calibrated hook, original, comparison and switch must land on exact target",
        )
    if approved_text_index is None:
        selected_text = str(editorial["selected_text"])
    else:
        if not 1 <= approved_text_index <= len(editorial["approved_text"]):
            raise MontageRejection("unapproved_text", "text index outside approved choices")
        selected_text = str(editorial["approved_text"][approved_text_index - 1])
    if selected_text not in editorial["approved_text"]:
        raise MontageRejection("unapproved_text", "on-screen text must be an exact approved line")
    if "Carry Forward" in selected_text:
        raise MontageRejection("prohibited_on_screen_copy", "Carry Forward is caption-only")
    title_lines = editorial["title_lines"].get(selected_text, [])
    if not isinstance(title_lines, list) or " ".join(title_lines) != selected_text:
        raise MontageRejection("unapproved_text_layout", "title lines alter approved campaign text")
    title_start = int(editorial["title_start_frame"])
    title_end = int(editorial["title_end_frame"])
    title = {
        "start_frame": title_start,
        "end_frame": title_end,
        "position": str(editorial["title_position"]),
        "lines": title_lines,
    }
    if not 0 <= title_start < title_end <= target:
        raise MontageRejection("invalid_title_window", "title window outside delivery")
    if title["position"] != "upper_right":
        raise MontageRejection("invalid_title_position", "title must clear the source logo")

    source_info = {
        "sha256": digest,
        "filename": filename,
        "frames": frame_count,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "review_url": profile.source_review_url if certified else None,
        "certified": certified,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": profile.name,
        "profile_sha256": profile_sha256(profile),
        "source": source_info,
        "evidence": {
            "before_frame": before,
            "after_frame": after,
            "state_difference": round(visual_difference, 6),
            "changed_pixel_fraction": round(changed_fraction, 6),
            "roi": roi,
            "verified_transformation": True,
        },
        "montage": {
            "type": "full_frame_toggle",
            "comparison_mode": comparison_mode,
            "hook": hook,
            "full_source_frames": frame_count,
            "comparison_frames": comparison_frames,
            "ending_shots": shots,
            "ending_frames": ending_frames,
            "title": title,
            "output_frames": target,
            "output_seconds": float(Fraction(target, 1) / fps),
            "approved_on_screen_text": selected_text,
        },
        "audio": {"policy": "source_audio_only", "source_track": "0:a:0", "added_music": False},
        "status": "PLANNED" if certified else "PREVIEW_ONLY",
    }


def validate_plan(plan: dict[str, Any], source: Path, profile: CampaignProfile) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("profile") != profile.name:
        raise MontageRejection("plan_version", "plan does not match campaign schema")
    if plan.get("profile_sha256") != profile_sha256(profile):
        raise MontageRejection("profile_changed", "regenerate the plan under the current profile")
    if plan.get("source", {}).get("sha256") != sha256(source):
        raise MontageRejection("source_changed", "source digest differs from the approved plan")
    if plan.get("source", {}).get("fps") != (
        f"{rate(profile).numerator}/{rate(profile).denominator}"
    ):
        raise MontageRejection("source_fps_mismatch", "source rate is not the planned rate")
    montage = plan["montage"]
    editorial = profile.config["editorial"]
    states = editorial["verified_visual_states"].get(plan["source"]["filename"])
    if states is None:
        raise MontageRejection("uncalibrated_source", "plan source has no verified states")
    expected_hook = {
        "start_frame": int(states["hook_start_frame"]),
        "frames": int(states["hook_frames"]),
    }
    expected_shots = [dict(shot) for shot in states["ending_shots"]]
    if montage["hook"] != expected_hook or montage["ending_shots"] != expected_shots:
        raise MontageRejection("source_window_changed", "source-native edit windows were modified")
    if montage.get("type") != "full_frame_toggle":
        raise MontageRejection("montage_type", "plan must use the legal full-frame edit")
    if montage.get("comparison_mode") not in editorial["comparison_modes"]:
        raise MontageRejection("invalid_comparison_mode", "unknown comparison layout")
    if int(montage["full_source_frames"]) != int(states["source_frames"]):
        raise MontageRejection("source_frames_changed", "full source is not retained")
    if int(plan["source"]["frames"]) != int(states["source_frames"]):
        raise MontageRejection("source_frames_changed", "source frame count differs")
    before = int(plan["evidence"]["before_frame"])
    after = int(plan["evidence"]["after_frame"])
    if (before, after) != (int(states["before_frame"]), int(states["after_frame"])):
        raise MontageRejection("evidence_changed", "visual-state anchors were modified")
    if int(montage["comparison_frames"]) != _frames(editorial["comparison_seconds"], rate(profile)):
        raise MontageRejection("frame_grid_mismatch", "comparison differs from profile")
    expected_ending = sum(int(s["frames"]) for s in expected_shots)
    expected_frames = (
        expected_hook["frames"]
        + int(states["source_frames"])
        + int(montage["comparison_frames"])
        + expected_ending
    )
    target = _frames(editorial["preferred_output_seconds"], rate(profile))
    if (
        int(montage["ending_frames"]) != expected_ending
        or int(montage["output_frames"]) != expected_frames
        or target != expected_frames
    ):
        raise MontageRejection("frame_grid_mismatch", "frame-exact plan is inconsistent")
    text = montage["approved_on_screen_text"]
    if text not in editorial["approved_text"]:
        raise MontageRejection("unapproved_text", "on-screen copy is not approved")
    title = montage["title"]
    if (
        title["lines"] != editorial["title_lines"].get(text)
        or title["start_frame"] != editorial["title_start_frame"]
        or title["end_frame"] != editorial["title_end_frame"]
        or title["position"] != editorial["title_position"]
        or " ".join(title["lines"]) != text
    ):
        raise MontageRejection(
            "title_changed", "title differs from approved copy/safe-zone schedule"
        )
    if plan.get("audio") != {
        "policy": "source_audio_only",
        "source_track": "0:a:0",
        "added_music": False,
    }:
        raise MontageRejection("audio_policy", "only included source audio may be used")
