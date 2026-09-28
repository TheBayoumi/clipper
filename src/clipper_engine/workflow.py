"""Profile-driven Clipper campaign stages; editorial and rendering logic live in Clipper."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from . import montage
from .profiles import CampaignProfile, load_profile
from .rendering import montage as montage_renderer
from .sources import catalog, mediasilo
from .sources import qa as source_qa


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _profile(args: argparse.Namespace) -> CampaignProfile:
    profile = load_profile(args.profile, getattr(args, "config", None))
    if profile.config.get("kind") != "announcement":
        raise ValueError("campaign CLI currently supports announcement campaigns")
    if not profile.source_review_url:
        raise ValueError("campaign has no authorized source review")
    return profile


def discover(profile: CampaignProfile, output: Path) -> dict[str, Any]:
    payload = catalog.discover(profile.source_review_url, output, None, None)
    minimum = int(profile.config["source_profile"]["minimum_count"])
    if payload["source_count"] < minimum:
        raise ValueError(
            f"source discovery found {payload['source_count']} originals; expected >= {minimum}"
        )
    if any(item.get("review_url") != profile.source_review_url for item in payload["sources"]):
        raise ValueError("MediaSilo catalog mixes authorized and unauthorized reviews")
    return payload


def acquire(profile: CampaignProfile, source_key: str, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="clipper-campaign-discovery-") as temp_dir:
        workspace = Path(temp_dir)
        catalog_path = workspace / "catalog.json"
        selected_path = workspace / "resolved.json"
        discover(profile, catalog_path)
        resolved = catalog.select(catalog_path, source_key, selected_path)
        if resolved.get("review_url") != profile.source_review_url:
            raise ValueError("selected asset is outside the configured campaign review")
        # The display title may omit its extension; file_name is the original media identity.
        original_name = str(resolved.get("file_name") or resolved["title"])
        suffix = Path(original_name).suffix.lower()
        if suffix not in catalog.VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported original source extension: {original_name}")
        source_path = output_dir / f"{source_key}{suffix}"
        qa_path = output_dir / f"{source_key}.source_qa.json"
        mediasilo.download(selected_path, source_path)
        certificate = source_qa.certify(source_path, source_key, selected_path, qa_path)
    return {"source": str(source_path), "source_qa": str(qa_path), "certificate": certificate}


def _certification(
    source: Path, source_qa_path: Path | None, profile: CampaignProfile
) -> dict[str, Any] | None:
    if source_qa_path is None:
        return None
    manifest = source_qa.verify(source, source_qa_path)
    if manifest.get("review_url") != profile.source_review_url:
        raise montage.MontageRejection(
            "source_outside_campaign",
            "certified original does not belong to the configured MediaSilo review",
        )
    return manifest


def plan(
    profile: CampaignProfile,
    source: Path,
    source_qa_path: Path | None,
    output: Path,
    comparison_mode: str = "wipe",
    approved_text_index: int | None = None,
) -> dict[str, Any]:
    certificate = _certification(source, source_qa_path, profile)
    result = montage.build_plan(
        source,
        profile,
        certificate,
        comparison_mode=comparison_mode,
        approved_text_index=approved_text_index,
    )
    _write(output, result)
    return result


def render(
    profile: CampaignProfile,
    source: Path,
    source_qa_path: Path | None,
    plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    certificate = _certification(source, source_qa_path, profile)
    approved = json.loads(plan_path.read_text(encoding="utf-8"))
    if approved.get("status") == "PLANNED" and certificate is None:
        raise montage.MontageRejection(
            "source_certificate_missing",
            "campaign-qualified plans require the original source certificate",
        )
    if approved.get("status") == "PREVIEW_ONLY" and certificate is not None:
        raise montage.MontageRejection(
            "preview_not_qualified", "regenerate the plan using the certified source"
        )
    if approved.get("source", {}).get("certified") is not (certificate is not None):
        raise montage.MontageRejection(
            "certification_changed", "plan and acquisition have different certification state"
        )
    return montage_renderer.render(source, profile, approved, output_dir)


def qualify(profile: CampaignProfile, manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = montage.profile_sha256(profile)
    if manifest.get("profile") != profile.name or manifest.get("profile_sha256") != expected_hash:
        raise montage.MontageRejection(
            "profile_mismatch", "render was not produced by this profile"
        )
    if manifest.get("status") != "PASS" or not manifest.get("source", {}).get("certified"):
        raise montage.MontageRejection(
            "uncertified_preview", "a local preview cannot pass production qualification"
        )
    if manifest["source"].get("review_url") != profile.source_review_url:
        raise montage.MontageRejection("source_outside_campaign", "incorrect MediaSilo review")
    path = Path(str(manifest["file"]))
    if not path.is_file() or montage.sha256(path) != manifest.get("sha256"):
        raise montage.MontageRejection("delivery_hash", "delivery is missing or has changed")
    if not all(manifest["qa"]["checks"].values()):
        raise montage.MontageRejection("technical_qa", "render manifest reports failed checks")
    edit = manifest["plan"]["montage"]
    selected_windows = montage.source_windows(edit)
    selected_frames = montage.source_frame_count(edit)
    is_excerpt = (
        len(selected_windows) > 1
        or int(selected_windows[0]["start_frame"]) != 0
        or selected_frames != int(edit["full_source_frames"])
    )
    if is_excerpt:
        excerpt = manifest["staging"].get("source_excerpt")
        if (
            not isinstance(excerpt, dict)
            or excerpt.get("windows") != selected_windows
            or excerpt.get("frame_count") != selected_frames
            or excerpt.get("source_to_piece_hashes_exact") is not True
            or manifest["qa"]["checks"].get("source_excerpt_hashes_exact") is not True
        ):
            raise montage.MontageRejection(
                "source_excerpt", "selected source window lacks exact source->FFV1 frame proof"
            )
    portrait: dict[str, Any] | None = manifest.get("portrait")
    if profile.config["output"].get("portrait_matte", {}).get("enabled", False):
        if not isinstance(portrait, dict):
            raise montage.MontageRejection(
                "portrait_missing", "approved full-duration, source-synced portrait is required"
            )
        portrait_path = Path(str(portrait.get("file") or ""))
        if not portrait_path.is_file() or montage.sha256(portrait_path) != portrait.get("sha256"):
            raise montage.MontageRejection(
                "portrait_hash", "portrait delivery is missing or changed"
            )
        from .rendering.portrait_matte import (
            CASCADE_REQUIRED_CHECKS,
            HERO_FOCUS_REQUIRED_CHECKS,
            CONTINUOUS_REVEAL_REQUIRED_CHECKS,
            PORTRAIT_REQUIRED_CHECKS,
            REBOUND_REQUIRED_CHECKS,
            SNAPBACK_REQUIRED_CHECKS,
            SPOTLIGHT_REQUIRED_CHECKS,
        )

        required_checks = PORTRAIT_REQUIRED_CHECKS
        if manifest["plan"]["montage"]["comparison_mode"] == "cascade":
            required_checks = required_checks | CASCADE_REQUIRED_CHECKS
            from .rendering import panel_compositor

            cmp_qa = manifest["staging"]["comparison"]
            panel_qa = portrait.get("cascade")
            if (
                not isinstance(panel_qa, dict)
                or panel_qa.get("source_still_sha256") != cmp_qa.get("source_still_sha256")
                or panel_qa.get("frame_count") != manifest["plan"]["montage"]["output_frames"]
            ):
                raise montage.MontageRejection(
                    "cascade_stills", "the portrait panels differ from certified source stills"
                )
            cfg = panel_compositor.config(
                profile, int(manifest["plan"]["montage"]["comparison_frames"])
            )
            starts = int(manifest["plan"]["montage"]["hook"]["frames"]) + int(
                manifest["plan"]["montage"]["source_window"]["frames"]
            )
            if (
                panel_qa.get("switch_frames")
                != [starts + int(n) for n in cfg["switch_start_frames"]]
                or panel_qa.get("text_synced") is not True
            ):
                raise montage.MontageRejection(
                    "cascade_schedule", "panel switches do not follow calibrated timeline"
                )

        if manifest["plan"]["montage"]["comparison_mode"] == "spotlight":
            from .rendering import panel_compositor

            required_checks = required_checks | SPOTLIGHT_REQUIRED_CHECKS
            cmp_qa = manifest["staging"]["comparison"]
            panel_qa = portrait.get("spotlight")
            if (
                not isinstance(panel_qa, dict)
                or panel_qa.get("source_still_sha256") != cmp_qa.get("source_still_sha256")
                or panel_qa.get("frame_count") != edit["output_frames"]
            ):
                raise montage.MontageRejection(
                    "spotlight_stills", "spotlight does not use the certified comparison stills"
                )
            cfg = panel_compositor.spotlight_config(profile, int(edit["comparison_frames"]))
            compare_start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
            switch_frames = [
                compare_start + i * int(cfg["focus_frames"]) + int(cfg["switch_after_frames"])
                for i in range(3)
            ]
            if (
                panel_qa.get("focus_order") != cfg["operator_order"]
                or panel_qa.get("switch_frames") != switch_frames
                or panel_qa.get("group_start_frame") != compare_start + 3 * int(cfg["focus_frames"])
                or panel_qa.get("text_synced") is not True
            ):
                raise montage.MontageRejection(
                    "spotlight_schedule", "spotlight no longer follows its calibrated story"
                )

        if manifest["plan"]["montage"]["comparison_mode"] == "spotlight":
            storyboard = portrait.get("storyboard")
            expected_story_frames = [
                0,
                *switch_frames,
                compare_start + 3 * int(cfg["focus_frames"]) + 2,
                int(edit["output_frames"]) - 1,
            ]
            if (
                not isinstance(storyboard, dict)
                or storyboard.get("frames") != expected_story_frames
                or storyboard.get("source") != "actual_encoded_delivery"
                or not Path(str(storyboard.get("file") or "")).is_file()
                or montage.sha256(Path(str(storyboard["file"]))) != storyboard.get("sha256")
            ):
                raise montage.MontageRejection(
                    "spotlight_storyboard", "actual-render storyboard is missing or changed"
                )

        if edit["comparison_mode"] == "snapback":
            from .rendering import panel_compositor

            required_checks = required_checks | SNAPBACK_REQUIRED_CHECKS
            cmp_qa = manifest["staging"]["comparison"]
            panel_qa = portrait.get("snapback")
            if (
                not isinstance(panel_qa, dict)
                or panel_qa.get("source_still_sha256") != cmp_qa.get("source_still_sha256")
                or panel_qa.get("frame_count") != edit["output_frames"]
                or panel_qa.get("source_only") is not True
            ):
                raise montage.MontageRejection(
                    "snapback_stills", "snapback panels lack exact certified stills"
                )
            cfg = panel_compositor.snapback_config(profile, int(edit["comparison_frames"]))
            start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
            rewind = start + int(cfg["after_preview_frames"])
            reveal = rewind + int(cfg["before_hold_frames"])
            switched = reveal + int(cfg["transition_frames"]) - 1
            switch_frames = [rewind, reveal, switched]
            samples = panel_qa.get("sampled_states", {})
            if (
                panel_qa.get("panel_count") != 2
                or panel_qa.get("focus_operator_index") != cfg["focus_operator_index"]
                or panel_qa.get("focus_roi")
                != profile.config["output"]["portrait_matte"]["operator_rois"][
                    cfg["focus_operator_index"]
                ]
                or panel_qa.get("comparison_start_frame") != start
                or panel_qa.get("switch_frames") != switch_frames
                or panel_qa.get("text_synced") is not True
                or not isinstance(samples, dict)
                or any(
                    samples.get(str(n)) != value
                    for n, value in (
                        (rewind - 1, 1.0),
                        (rewind, 0.0),
                        (reveal - 1, 0.0),
                        (switched, 1.0),
                    )
                )
            ):
                raise montage.MontageRejection(
                    "snapback_schedule", "rewind/reveal differs from calibrated timeline"
                )
            storyboard = portrait.get("storyboard")
            expected_indices = [
                0,
                rewind - 1,
                rewind,
                reveal - 1,
                switched,
                int(edit["output_frames"]) - 1,
            ]
            if (
                not isinstance(storyboard, dict)
                or storyboard.get("frames") != expected_indices
                or storyboard.get("source") != "actual_encoded_delivery"
                or not Path(str(storyboard.get("file") or "")).is_file()
                or montage.sha256(Path(str(storyboard["file"]))) != storyboard.get("sha256")
            ):
                raise montage.MontageRejection(
                    "snapback_storyboard", "actual encoded storyboard missing or changed"
                )

        if edit["comparison_mode"] == "hero_focus":
            from .rendering import panel_compositor

            required_checks = required_checks | HERO_FOCUS_REQUIRED_CHECKS
            cmp_qa = manifest["staging"]["comparison"]
            panel_qa = portrait.get("hero_focus")
            if (
                not isinstance(panel_qa, dict)
                or panel_qa.get("source_still_sha256") != cmp_qa.get("source_still_sha256")
                or panel_qa.get("source_only") is not True
                or panel_qa.get("frame_count") != edit["output_frames"]
            ):
                raise montage.MontageRejection(
                    "hero_focus_stills", "hero cards differ from certified comparison"
                )
            cfg = panel_compositor.hero_focus_config(profile, int(edit["comparison_frames"]))
            start = int(edit["hook"]["frames"]) + int(edit["source_window"]["frames"])
            original = start + int(cfg["after_preview_frames"])
            split = original + int(cfg["before_hold_frames"])
            group = split + int(cfg["split_frames"])
            end = group + int(cfg["group_frames"])
            expected = [start, original, split, group, end]
            samples = panel_qa.get("sampled_states", {})
            if (
                panel_qa.get("mode") != "hero_focus"
                or panel_qa.get("panel_count") != 3
                or panel_qa.get("focus_operator_index") != cfg["focus_operator_index"]
                or panel_qa.get("focus_roi") != cfg["operator_rois"][cfg["focus_operator_index"]]
                or panel_qa.get("phase_frames") != expected
                or panel_qa.get("text_synced") is not True
                or not isinstance(samples, dict)
                or any(
                    samples.get(str(frame)) != {"phase": phase, "after": progress}
                    for frame, phase, progress in (
                        (original - 1, "after", 1.0),
                        (original, "before", 0.0),
                        (split, "split", 0.5),
                        (group, "group", 1.0),
                    )
                )
            ):
                raise montage.MontageRejection(
                    "hero_focus_schedule", "hero reveals differ from calibrated timeline"
                )
            storyboard = portrait.get("storyboard")
            expected_story = [
                0,
                original - 1,
                original,
                split,
                group,
                int(edit["output_frames"]) - 1,
            ]
            if (
                not isinstance(storyboard, dict)
                or storyboard.get("frames") != expected_story
                or storyboard.get("source") != "actual_encoded_delivery"
                or not Path(str(storyboard.get("file") or "")).is_file()
                or montage.sha256(Path(str(storyboard["file"]))) != storyboard.get("sha256")
            ):
                raise montage.MontageRejection(
                    "hero_focus_storyboard", "encoded hero storyboard is missing or altered"
                )

        if edit["comparison_mode"] == "rebound":
            from .rendering import panel_compositor

            required_checks = (
                (required_checks - {"toggle_opening_off"})
                | {"toggle_opening_after"}
                | REBOUND_REQUIRED_CHECKS
            )
            cmp_qa = manifest["staging"]["comparison"]
            panel_qa = portrait.get("rebound")
            if (
                not isinstance(panel_qa, dict)
                or panel_qa.get("source_still_sha256") != cmp_qa.get("source_still_sha256")
                or panel_qa.get("source_only") is not True
                or panel_qa.get("frame_count") != edit["output_frames"]
            ):
                raise montage.MontageRejection(
                    "rebound_stills", "large detail cards are not from the certified master"
                )
            cfg = panel_compositor.rebound_config(profile, int(edit["comparison_frames"]))
            start = int(edit["hook"]["frames"]) + montage.source_frame_count(edit)
            original = start + int(cfg["before_hold_frames"])
            flash = original + int(cfg["after_flash_frames"])
            group = flash + int(cfg["split_frames"])
            end = group + int(cfg["group_frames"])
            samples = panel_qa.get("sampled_states", {})
            expected = [start, original, flash, group, end]
            if (
                panel_qa.get("mode") != "rebound"
                or panel_qa.get("panel_count") != 3
                or panel_qa.get("focus_operator_index") != cfg["focus_operator_index"]
                or panel_qa.get("focus_roi") != cfg["detail_roi"]
                or panel_qa.get("phase_frames") != expected
                or panel_qa.get("live_detail_frames") != start + int(edit["output_frames"]) - end
                or panel_qa.get("live_detail_source") != "clean_canonical_ffv1_nut"
                or panel_qa.get("text_synced") is not True
                or not isinstance(samples, dict)
                or any(
                    samples.get(str(frame)) != {"phase": phase, "after": progress}
                    for frame, phase, progress in (
                        (0, "live", 1.0),
                        (int(edit["hook"]["frames"]), "live", 0.0),
                        (start, "before", 0.0),
                        (original, "after", 1.0),
                        (flash, "split", 0.5),
                        (group, "group", 1.0),
                    )
                )
            ):
                raise montage.MontageRejection(
                    "rebound_schedule", "result-first source/native detail schedule drifted"
                )
            storyboard = portrait.get("storyboard")
            expected_story = [
                0,
                int(edit["hook"]["frames"]),
                int(edit["hook"]["frames"]) + int(edit["source_windows"][0]["frames"]) + 4,
                original,
                flash,
                int(edit["output_frames"]) - 1,
            ]
            if (
                not isinstance(storyboard, dict)
                or storyboard.get("frames") != expected_story
                or storyboard.get("source") != "actual_encoded_delivery"
                or not Path(str(storyboard.get("file") or "")).is_file()
                or montage.sha256(Path(str(storyboard["file"]))) != storyboard.get("sha256")
            ):
                raise montage.MontageRejection(
                    "rebound_storyboard", "encoded result-first storyboard missing or changed"
                )

        if edit["comparison_mode"] == "continuous_reveal":
            from .rendering import kinetic_reframe

            required_checks = required_checks | CONTINUOUS_REVEAL_REQUIRED_CHECKS
            cfg = kinetic_reframe.config(profile, int(edit["output_frames"]))
            reveal = portrait.get("continuous_reveal")
            expected_window = profile.config["editorial"]["mode_timing"]["continuous_reveal"][
                "source_window"
            ]
            if (
                edit.get("type") != "continuous_source_reveal"
                or edit.get("source_window") != expected_window
                or montage.source_frame_count(edit) != int(edit["output_frames"])
                or not isinstance(reveal, dict)
                or reveal.get("mode") != "continuous_reveal"
                or reveal.get("frame_count") != int(edit["output_frames"])
                or reveal.get("source_only") is not True
                or reveal.get("source_frame_grid_exact") is not True
                or reveal.get("ai_enhancement") is not False
                or reveal.get("keyframes") != cfg["keyframes"]
                or reveal.get("storyboard_frames") != cfg["storyboard_frames"]
                or reveal.get("bottom_matte_height") != cfg["bottom_matte_height"]
                or reveal.get("minimum_effective_source_width_px", 0)
                < int(cfg["minimum_effective_source_width"])
            ):
                raise montage.MontageRejection(
                    "continuous_reveal_schedule",
                    "continuous source reveal no longer matches its calibrated Clipper story",
                )
            storyboard = portrait.get("storyboard")
            if (
                not isinstance(storyboard, dict)
                or storyboard.get("frames") != cfg["storyboard_frames"]
                or storyboard.get("source") != "actual_encoded_delivery"
                or not Path(str(storyboard.get("file") or "")).is_file()
                or montage.sha256(Path(str(storyboard["file"]))) != storyboard.get("sha256")
            ):
                raise montage.MontageRejection(
                    "continuous_reveal_storyboard",
                    "encoded continuous-reveal storyboard missing or altered",
                )

        portrait_checks = portrait.get("qa", {}).get("checks", {})
        portrait_checks = portrait.get("qa", {}).get("checks", {})
        if (
            not isinstance(portrait_checks, dict)
            or set(portrait_checks) != required_checks
            or not all(value is True for value in portrait_checks.values())
        ):
            raise montage.MontageRejection(
                "portrait_qa", "portrait technical or semantic QA failed"
            )
        from . import media_contract as media

        portrait_video = media.video_profile(portrait_path, count_frames=True)
        portrait_size = profile.config["output"]["portrait_matte"]
        if (
            (portrait_video["width"], portrait_video["height"])
            != (int(portrait_size["width"]), int(portrait_size["height"]))
            or portrait_video["frame_count"] != int(manifest["plan"]["montage"]["output_frames"])
            or portrait["title"]["text_visible_frames"] != portrait_video["frame_count"]
            or portrait["title"]["approved_copy"]
            != manifest["plan"]["montage"]["approved_on_screen_text"]
        ):
            raise montage.MontageRejection(
                "portrait_contract", "portrait frame grid, safe-title schedule or geometry changed"
            )
    # Check the encoded delivery rather than relying on an old JSON assertion.
    from . import media_contract as media

    probe = json.loads(
        media.run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    duration = float(probe["format"]["duration"])
    editorial = profile.config["editorial"]
    if (
        not float(editorial["minimum_output_seconds"])
        <= duration
        <= float(
            editorial.get("mode_timing", {})
            .get(manifest["plan"]["montage"]["comparison_mode"], {})
            .get("maximum_output_seconds", editorial["maximum_output_seconds"])
        )
    ):
        raise montage.MontageRejection("encoded_duration", f"actual duration {duration:.6f}s")
    result = {
        "profile": profile.name,
        "status": "PASS",
        "certified_review": profile.source_review_url,
        "video_sha256": manifest["sha256"],
        "encoded_duration_seconds": duration,
        "encoded_frames": manifest["qa"]["encoded_video_frames"],
        "allowed_on_screen_copy": manifest["plan"]["montage"]["approved_on_screen_text"],
        "audio_policy": "source_audio_only",
    }
    if portrait is not None:
        result["primary_delivery"] = portrait["file"]
        result["primary_delivery_sha256"] = portrait["sha256"]
        result["portrait_sync"] = portrait["title"]["progress_samples"]
        result["portrait_text_visible_frames"] = portrait["title"]["text_visible_frames"]
    _write(output, result)
    return result


def run_campaign(args: argparse.Namespace) -> int:
    profile = _profile(args)
    command = args.campaign_command
    if command == "discover":
        result = discover(profile, args.output)
    elif command == "acquire":
        result = acquire(profile, args.source_key, args.output_dir)
    elif command == "plan":
        result = plan(
            profile,
            args.source,
            args.source_qa,
            args.output,
            comparison_mode=(
                args.comparison_mode or profile.config["editorial"]["comparison_modes"][0]
            ),
            approved_text_index=args.approved_text_index,
        )
    elif command == "render":
        result = render(profile, args.source, args.source_qa, args.plan, args.output_dir)
    elif command == "qualify":
        result = qualify(profile, args.render_manifest, args.output)
    else:
        raise ValueError(f"unknown campaign stage: {command}")
    print(json.dumps(result, indent=2))
    return 0
