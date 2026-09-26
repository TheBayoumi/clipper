"""Clipper-owned deterministic announcement compositing and encoded-media QA."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from .. import media_contract as media
from ..montage import MontageRejection, rate, sha256, validate_plan
from ..profiles import CampaignProfile
from . import ffv1


def _fontfile() -> Path:
    completed = subprocess.run(
        ["fc-match", "-f", "%{file}", "DejaVu Sans:style=Bold"],
        text=True,
        capture_output=True,
        check=False,
    )
    path = Path(completed.stdout.strip())
    if completed.returncode or not path.is_file():
        raise RuntimeError("an installed readable bold font is required for approved copy")
    return path


def _escape_filter_path(path: Path) -> str:
    # FFmpeg filtergraph parser and drawtext both interpret colons/backslashes.
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def filter_graph(
    plan: dict[str, Any],
    profile: CampaignProfile,
    title_path: Path,
    font: Path,
    show_embedded_title: bool = True,
) -> str:
    """Exact-frame preview-hook -> original -> comparison -> final A/B edit."""
    output = profile.config["output"]
    fps = rate(profile)
    edit = plan["montage"]
    title = edit["title"]
    source_frames = int(edit["full_source_frames"])
    hook = edit["hook"]
    comparison_frames = int(edit["comparison_frames"])
    ending_frames = int(edit["ending_frames"])
    font_size = max(8, round(int(output["height"]) * 0.043))
    padding = max(3, round(int(output["height"]) * 0.012))
    margin = max(12, round(int(output["width"]) * 0.034))
    y = max(9, round(int(output["height"]) * 0.054))
    title_filter = (
        "drawtext="
        f"fontfile='{_escape_filter_path(font)}':"
        f"textfile='{_escape_filter_path(title_path)}':"
        f"fontsize={font_size}:fontcolor=white:"
        f"line_spacing={max(2, font_size // 6)}:"
        f"box=1:boxcolor=black@0.76:boxborderw={padding}:"
        "expansion=none:"
        f"x=w-text_w-{margin}:y={y}:"
        f"enable='between(n,{title['start_frame']},{title['end_frame'] - 1})'"
    )
    if title["position"] != "upper_right":
        raise MontageRejection("invalid_title_position", "unapproved title safe-zone")
    hstart = float(Fraction(int(hook["start_frame"]), 1) / fps)
    hend = float(Fraction(int(hook["start_frame"] + hook["frames"]), 1) / fps)
    hlen = float(Fraction(int(hook["frames"]), 1) / fps)
    source_seconds = float(Fraction(source_frames, 1) / fps)
    comparison_seconds = float(Fraction(comparison_frames, 1) / fps)
    ending_seconds = float(Fraction(ending_frames, 1) / fps)
    fade = float(profile.config["editorial"].get("audio_declick_ms", 12)) / 1000.0
    if not 0.002 <= fade <= min(hlen, comparison_seconds, ending_seconds) / 4:
        raise MontageRejection("audio_declick_invalid", "source-only edit-edge fade out of range")
    title_chain = f",{title_filter}" if show_embedded_title else ""
    # Only audio from the one certified source track, retained at normal playback rate.
    # Visual switches are under a continuous source-native audio passage to avoid pops.
    return ";".join(
        [
            "[3:v]setpts=PTS-STARTPTS[vhook]",
            "[0:v]setpts=PTS-STARTPTS[vfull]",
            "[1:v]setpts=PTS-STARTPTS[vcompare]",
            "[2:v]setpts=PTS-STARTPTS[vfinal]",
            "[vhook][vfull][vcompare][vfinal]concat=n=4:v=1:a=0,"
            f"fps={fps.numerator}/{fps.denominator},"
            f"format={output['pixel_format']}{title_chain}[outv]",
            "[0:a]asplit=4[ahook][afull][acompare][afinal]",
            f"[ahook]atrim=start={hstart:.9f}:end={hend:.9f},"
            f"asetpts=PTS-STARTPTS,apad=pad_dur=0.1,atrim=duration={hlen:.9f},"
            f"afade=t=in:st=0:d={fade:.4f},"
            f"afade=t=out:st={hlen - fade:.9f}:d={fade:.4f}[au0]",
            f"[afull]atrim=start=0:end={source_seconds:.9f},"
            "asetpts=PTS-STARTPTS,apad=pad_dur=0.1,"
            f"atrim=duration={source_seconds:.9f},"
            f"afade=t=in:st=0:d={fade:.4f},"
            f"afade=t=out:st={source_seconds - fade:.9f}:d={fade:.4f}[au1]",
            f"[acompare]atrim=start=0.8:end={0.8 + comparison_seconds:.9f},"
            "asetpts=PTS-STARTPTS,apad=pad_dur=0.1,"
            f"atrim=duration={comparison_seconds:.9f},"
            f"afade=t=in:st=0:d={fade:.4f},"
            f"afade=t=out:st={comparison_seconds - fade:.9f}:d={fade:.4f}[au2]",
            f"[afinal]atrim=start=1.8:end={1.8 + ending_seconds:.9f},"
            "asetpts=PTS-STARTPTS,apad=pad_dur=0.1,"
            f"atrim=duration={ending_seconds:.9f},"
            f"afade=t=in:st=0:d={fade:.4f},"
            f"afade=t=out:st={ending_seconds - fade:.9f}:d={fade:.4f}[au3]",
            "[au0][au1][au2][au3]concat=n=4:v=0:a=1,"
            f"aresample={int(output['audio_sample_rate'])}:async=1:first_pts=0[outa]",
        ]
    )


def _still_frame(source: Path, frame: int, target: Path) -> None:
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-f",
            "image2",
            str(target),
        ]
    )
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"failed to extract verified source frame {frame}")


def _comparison_piece(
    staged: Path,
    workspace: Path,
    plan: dict[str, Any],
    profile: CampaignProfile,
    source_profile: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Full-frame original-size visual-state comparison: wipe or intentional A/B cuts."""
    before, after = workspace / "before.png", workspace / "after.png"
    before_frame = int(plan["evidence"]["before_frame"])
    after_frame = int(plan["evidence"]["after_frame"])
    _still_frame(staged, before_frame, before)
    _still_frame(staged, after_frame, after)
    output = profile.config["output"]
    fps = rate(profile)
    frames = int(plan["montage"]["comparison_frames"])
    mode = plan["montage"]["comparison_mode"]
    if mode == "cascade":
        from . import panel_compositor

        return panel_compositor.render_comparison(
            before, after, workspace, plan, profile, source_profile
        )
    if mode == "wipe":
        # Keep both operator groups full-sized. Start with a readable before state,
        # wipe in the after state, and let the after state settle before the final beat.
        wipe_offset = float(Fraction(frames // 6, 1) / fps)
        wipe_duration = float(Fraction(2 * frames // 3, 1) / fps)
        transition = (
            "[left][right]xfade=transition=wiperight:"
            f"duration={wipe_duration:.9f}:offset={wipe_offset:.9f}"
        )
    elif mode == "cuts":
        transition = "[left][right]blend=all_expr='if(lt(T,0.45)+between(T,1.05,1.35),A,B)'"
    else:
        raise MontageRejection("invalid_comparison_mode", str(mode))
    graph = ";".join(
        [
            f"[0:v]format=yuv420p,setsar=1,setpts=PTS-STARTPTS,"
            f"fps={fps.numerator}/{fps.denominator}[left]",
            f"[1:v]format=yuv420p,setsar=1,setpts=PTS-STARTPTS,"
            f"fps={fps.numerator}/{fps.denominator}[right]",
            f"{transition},trim=end_frame={frames},"
            f"fps={fps.numerator}/{fps.denominator},"
            f"format={output['pixel_format']}[outv]",
        ]
    )
    path = workspace / "comparison.nut"
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-filter_complex_threads",
            "1",
            "-framerate",
            str(output["fps"]),
            "-loop",
            "1",
            "-i",
            str(before),
            "-framerate",
            str(output["fps"]),
            "-loop",
            "1",
            "-i",
            str(after),
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads:v",
            "1",
            *media.profile_output_args(source_profile),
            *media.color_metadata_tag_args(source_profile),
            "-f",
            "nut",
            str(path),
        ]
    )
    measured = media.video_profile(path, count_frames=True)
    dimensions = measured["width"] == int(output["width"]) and measured["height"] == int(
        output["height"]
    )
    if measured["frame_count"] != frames or measured["codec_name"] != "ffv1" or not dimensions:
        raise RuntimeError(f"full-frame comparison contract failed: {measured}")
    return path, {
        "before_frame": before_frame,
        "after_frame": after_frame,
        "comparison_mode": mode,
        "comparison_frames": frames,
        "frame_count_exact": measured["frame_count"] == frames,
        "full_frame": dimensions,
        "no_black_bar_layout": True,
        "source_only": True,
    }


def _source_windows_piece(
    staged: Path,
    workspace: Path,
    filename: str,
    windows: list[dict[str, Any]],
    source_profile: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Render calibrated native windows to one FFV1/NUT piece; verify every decoded frame."""
    if not windows:
        raise RuntimeError("source windows are empty")
    total = sum(int(shot["frames"]) for shot in windows)
    filter_parts = [
        f"[0:v]split={len(windows)}" + "".join(f"[raw{i}]" for i in range(len(windows)))
    ]
    for i, shot in enumerate(windows):
        start = int(shot["start_frame"])
        finish = start + int(shot["frames"])
        filter_parts.append(
            f"[raw{i}]trim=start_frame={start}:end_frame={finish},setpts=PTS-STARTPTS[cut{i}]"
        )
    filter_parts.append(
        "".join(f"[cut{i}]" for i in range(len(windows))) + f"concat=n={len(windows)}:v=1:a=0[outv]"
    )
    path = workspace / filename
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-filter_complex_threads",
            "1",
            "-i",
            str(staged),
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outv]",
            "-vsync",
            "0",
            "-frames:v",
            str(total),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads:v",
            "1",
            "-pix_fmt",
            str(source_profile["pix_fmt"]),
            *media.color_metadata_tag_args(source_profile),
            "-an",
            "-f",
            "nut",
            str(path),
        ]
    )
    original = media.frame_hashes(staged, pix_fmt=str(source_profile["pix_fmt"]))
    measured = media.frame_hashes(path, pix_fmt=str(source_profile["pix_fmt"]))
    expected = [
        hash_value
        for shot in windows
        for hash_value in original[
            int(shot["start_frame"]) : int(shot["start_frame"]) + int(shot["frames"])
        ]
    ]
    verified = bool(measured) and expected == measured
    if not verified:
        raise RuntimeError(f"source->{filename} FFV1/NUT decoded frame hashes diverged")
    video = media.video_profile(path, count_frames=True)
    if video["frame_count"] != total or video["codec_name"] != "ffv1":
        raise RuntimeError(f"{filename} has wrong frame count or codec")
    return path, {
        "windows": windows,
        "frame_count": len(measured),
        "source_to_piece_hashes_exact": verified,
    }


def _metric(canonical: Path, delivery: Path, filter_name: str, pattern: str) -> float:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-threads:v",
            "1",
            "-i",
            str(canonical),
            "-threads:v",
            "1",
            "-i",
            str(delivery),
            "-lavfi",
            filter_name,
            "-an",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"canonical/delivery {filter_name} failed: {completed.stderr[-1200:]}")
    matches = re.findall(pattern, completed.stderr)
    if not matches:
        raise RuntimeError(f"canonical/delivery {filter_name} metric not found")
    return float(matches[-1])


def _qa(
    canonical: Path,
    target: Path,
    stage: dict[str, Any],
    plan: dict[str, Any],
    profile: CampaignProfile,
) -> dict[str, Any]:
    output = profile.config["output"]
    editorial = profile.config["editorial"]
    canonical_video = media.video_profile(canonical, count_frames=True)
    delivery_video = media.video_profile(target, count_frames=True)
    audio = media.audio_profile(target)
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
                str(target),
            ]
        ).stdout
    )
    duration = float(probe["format"]["duration"])
    expected_frames = int(plan["montage"]["output_frames"])
    fps = f"{rate(profile).numerator}/{rate(profile).denominator}"
    source_colors = stage["source_color_metadata"]
    canonical_colors = media.stream_color_tags(canonical)
    delivery_colors = media.source_color_metadata(delivery_video)
    checks = {
        "source_stage_lossless": all(stage["checks"].values()),
        "comparison_from_verified_source_frames": stage["comparison"]["source_only"]
        and stage["comparison"]["frame_count_exact"],
        "hook_source_hashes_exact": stage["hook"]["source_to_piece_hashes_exact"],
        "reveal_source_hashes_exact": stage["reveal"]["source_to_piece_hashes_exact"],
        "full_frame_comparison": stage["comparison"]["full_frame"]
        and stage["comparison"]["no_black_bar_layout"],
        "canonical_ffv1_nut": canonical_video["codec_name"] == "ffv1" and ffv1._is_nut(canonical),
        "canonical_color_metadata_tags_exact": all(
            canonical_colors.get(field) == value for field, value in source_colors.items()
        ),
        "delivery_color_metadata_exact": all(
            delivery_colors.get(field) == value for field, value in source_colors.items()
        ),
        "encoded_h264": delivery_video["codec_name"] == "h264",
        "canonical_frames_exact": canonical_video["frame_count"] == expected_frames,
        "encoded_frames_exact": delivery_video["frame_count"] == expected_frames,
        "encoded_fps_exact": delivery_video["r_frame_rate"] == fps
        and delivery_video["avg_frame_rate"] == fps,
        "source_native_dimensions": (delivery_video["width"], delivery_video["height"])
        == (int(output["width"]), int(output["height"])),
        "source_native_pixel_format": delivery_video["pix_fmt"] == output["pixel_format"],
        "source_audio_codec": audio["codec_name"] == output["audio_codec"],
        "source_audio_rate": audio["sample_rate"] == int(output["audio_sample_rate"]),
        "source_audio_channels": audio["channels"] == int(output["audio_channels"]),
        "encoded_duration": float(editorial["minimum_output_seconds"])
        <= duration
        <= float(editorial["maximum_output_seconds"]),
        "exact_nominal_frame_duration": math.isclose(
            duration, expected_frames / float(rate(profile)), abs_tol=0.055
        ),
        "approved_copy": plan["montage"]["approved_on_screen_text"] in editorial["approved_text"],
        "no_added_music": plan["audio"]["added_music"] is False,
    }
    # Canonical is the edited visual reference; source/stage checks retain original frame hashes.
    ssim = _metric(canonical, target, "ssim", r"All:([0-9.]+)")
    psnr = _metric(canonical, target, "psnr", r"average:([0-9.]+)")
    checks["canonical_to_delivery_ssim"] = ssim >= 0.96
    checks["canonical_to_delivery_psnr_db"] = psnr >= 35.0
    if not all(checks.values()):
        raise RuntimeError(f"announcement technical QA failed: {checks}")
    return {
        "checks": checks,
        "encoded_duration_seconds": duration,
        "encoded_video_frames": delivery_video["frame_count"],
        "ssim": ssim,
        "psnr_db": psnr,
        "video_profile": delivery_video,
        "audio_profile": audio,
        "source_color_metadata": source_colors,
        "canonical_color_metadata_tags": canonical_colors,
        "delivery_color_metadata": delivery_colors,
    }


def _render_canonical(
    staged: Path,
    comparison: Path,
    ending: Path,
    hook: Path,
    graph: str,
    canonical: Path,
    output: dict[str, Any],
    source_profile: dict[str, Any],
    plan: dict[str, Any],
    total: float,
) -> None:
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads:v",
            "1",
            "-i",
            str(staged),
            "-threads:v",
            "1",
            "-i",
            str(comparison),
            "-threads:v",
            "1",
            "-i",
            str(ending),
            "-threads:v",
            "1",
            "-i",
            str(hook),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-frames:v",
            str(plan["montage"]["output_frames"]),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads:v",
            "1",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(output["audio_sample_rate"]),
            "-ac",
            str(output["audio_channels"]),
            *media.profile_output_args(source_profile),
            *media.color_metadata_tag_args(source_profile),
            "-t",
            f"{total:.9f}",
            "-f",
            "nut",
            str(canonical),
        ]
    )
    if media.video_profile(canonical, count_frames=True)["frame_count"] != int(
        plan["montage"]["output_frames"]
    ):
        raise RuntimeError("canonical montage is not exact on its frame grid")


def render(
    source: Path,
    profile: CampaignProfile,
    plan: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validate_plan(plan, source, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{profile.name}_{plan['montage']['comparison_mode']}.mp4"
    output = profile.config["output"]
    with tempfile.TemporaryDirectory(prefix="clipper-montage-", dir=output_dir) as directory:
        workspace = Path(directory)
        staged = workspace / "source_lossless.nut"
        canonical = workspace / "canonical_montage.nut"
        title = workspace / "approved_title.txt"
        title.write_text("\n".join(plan["montage"]["title"]["lines"]), encoding="utf-8")
        staging = ffv1.stage_native_source(source, staged)
        source_profile = staging["source_profile"]
        comparison, comparison_qa = _comparison_piece(
            staged, workspace, plan, profile, source_profile
        )
        hook, hook_qa = _source_windows_piece(
            staged, workspace, "hook.nut", [plan["montage"]["hook"]], source_profile
        )
        ending, ending_qa = _source_windows_piece(
            staged, workspace, "ending.nut", plan["montage"]["ending_shots"], source_profile
        )
        staging["comparison"] = comparison_qa
        staging["hook"] = hook_qa
        staging["reveal"] = ending_qa
        graph = filter_graph(plan, profile, title, _fontfile())
        total = float(plan["montage"]["output_seconds"])
        _render_canonical(
            staged,
            comparison,
            ending,
            hook,
            graph,
            canonical,
            output,
            source_profile,
            plan,
            total,
        )
        media.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads:v",
                "1",
                "-i",
                str(canonical),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                str(output["video_codec"]),
                "-preset",
                str(output["video_preset"]),
                "-crf",
                str(output["video_crf"]),
                "-threads:v",
                "2",
                "-r",
                str(output["fps"]),
                "-c:a",
                str(output["audio_codec"]),
                "-b:a",
                "320k",
                "-ar",
                str(output["audio_sample_rate"]),
                "-ac",
                str(output["audio_channels"]),
                *media.profile_output_args(source_profile),
                "-video_track_timescale",
                str(media.timing_from_profile(source_profile).track_timescale),
                "-movflags",
                "+faststart",
                "-t",
                f"{total:.9f}",
                str(target),
            ]
        )
        qa = _qa(canonical, target, staging, plan, profile)
        portrait: dict[str, Any] | None = None
        if output.get("portrait_matte", {}).get("enabled", False):
            from . import portrait_matte

            # Source-fidelity landscape QA retains its embedded approved copy.
            # The portrait derivative starts from an independently encoded CLEAN
            # canonical edit to prevent duplicate typography over the Operators.
            clean = workspace / "clean_canonical.nut"
            clean_graph = filter_graph(plan, profile, title, _fontfile(), show_embedded_title=False)
            _render_canonical(
                staged,
                comparison,
                ending,
                hook,
                clean_graph,
                clean,
                output,
                source_profile,
                plan,
                total,
            )
            portrait = portrait_matte.render_portrait(
                clean,
                output_dir,
                workspace,
                plan,
                profile,
                source_profile,
                _metric,
                comparison_stills=(workspace / "before.png", workspace / "after.png"),
                expected_still_hashes=staging["comparison"].get("source_still_sha256"),
            )

    manifest = {
        "schema_version": 1,
        "profile": profile.name,
        "profile_sha256": plan["profile_sha256"],
        "source": plan["source"],
        "plan": plan,
        "file": str(target.resolve()),
        "sha256": sha256(target),
        "staging": staging,
        "qa": qa,
        "portrait": portrait,
        "status": "PASS" if plan["source"]["certified"] else "PREVIEW_ONLY",
    }
    path = output_dir / "render_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
