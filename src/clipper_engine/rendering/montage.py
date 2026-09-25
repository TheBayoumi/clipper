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
    plan: dict[str, Any], profile: CampaignProfile, title_path: Path, font: Path
) -> str:
    """Single deterministic composition graph over independently certified video pieces."""
    output = profile.config["output"]
    fps = rate(profile)
    title = str(plan["montage"]["approved_on_screen_text"])
    width, height = int(output["width"]), int(output["height"])
    font_size = min(max(8, round(height * 0.037)), max(8, int(width * 1.35 / len(title))))
    border_size = max(3, round(height * 0.01))
    title_filter = (
        "drawtext="
        f"fontfile='{_escape_filter_path(font)}':"
        f"textfile='{_escape_filter_path(title_path)}':"
        f"fontsize={font_size}:fontcolor=white:"
        f"box=1:boxcolor=black@0.84:boxborderw={border_size}:"
        "x=(w-text_w)/2:y=36"
    )
    source_frames = int(plan["source"]["frames"])
    comparison_frames = int(plan["montage"]["comparison_frames"])
    ending_start = int(plan["montage"]["ending_start_frame"])
    source_seconds = float(Fraction(source_frames, 1) / fps)
    comparison_seconds = float(Fraction(comparison_frames, 1) / fps)
    ending_start_seconds = float(Fraction(ending_start, 1) / fps)
    return ";".join(
        [
            "[0:v]setpts=PTS-STARTPTS[vfull]",
            "[1:v]setpts=PTS-STARTPTS[vcompare]",
            "[2:v]setpts=PTS-STARTPTS[vfinal]",
            "[vfull][vcompare][vfinal]concat=n=3:v=1:a=0,"
            f"fps={fps.numerator}/{fps.denominator},"
            f"format={output['pixel_format']},{title_filter}[outv]",
            "[0:a]asplit=3[afull][acompare][afinal]",
            f"[afull]atrim=start=0:end={source_seconds:.9f},asetpts=PTS-STARTPTS[au0]",
            f"[acompare]atrim=start=0:end={comparison_seconds:.9f},asetpts=PTS-STARTPTS[au1]",
            f"[afinal]atrim=start={ending_start_seconds:.9f}:end={source_seconds:.9f},"
            "asetpts=PTS-STARTPTS[au2]",
            "[au0][au1][au2]concat=n=3:v=0:a=1,"
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
    """Bounded still extraction avoids buffering the whole 1080p reel four times."""
    before, after = workspace / "before.png", workspace / "after.png"
    _still_frame(staged, int(plan["evidence"]["before_frame"]), before)
    _still_frame(staged, int(plan["evidence"]["after_frame"]), after)
    output = profile.config["output"]
    fps = rate(profile)
    width, height = int(output["width"]), int(output["height"])
    if width % 2 or height % 2:
        raise MontageRejection("geometry_invalid", "comparison dimensions must be even")
    half_width, half_height = width // 2, height // 2
    pad_y = (height - half_height) // 2
    frames = int(plan["montage"]["comparison_frames"])
    graph = ";".join(
        [
            f"[0:v]scale={half_width}:{half_height}:flags=lanczos,"
            f"pad={half_width}:{height}:0:{pad_y}:black,setsar=1[left]",
            f"[1:v]scale={half_width}:{half_height}:flags=lanczos,"
            f"pad={half_width}:{height}:0:{pad_y}:black,setsar=1[right]",
            f"[left][right]hstack=inputs=2,"
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
            "-framerate",
            str(output["fps"]),
            "-loop",
            "1",
            "-i",
            str(before),
            "-threads:v",
            "1",
            "-framerate",
            str(output["fps"]),
            "-loop",
            "1",
            "-i",
            str(after),
            "-filter_complex_threads",
            "1",
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
            "-f",
            "nut",
            str(path),
        ]
    )
    measured = media.video_profile(path, count_frames=True)
    if measured["frame_count"] != frames or measured["codec_name"] != "ffv1":
        raise RuntimeError(f"comparison has wrong frame count or codec: {measured}")
    return path, {
        "before_frame": int(plan["evidence"]["before_frame"]),
        "after_frame": int(plan["evidence"]["after_frame"]),
        "comparison_frames": frames,
        "frame_count_exact": measured["frame_count"] == frames,
        "video_codec": measured["codec_name"],
        "source_only": True,
    }


def _ending_piece(
    staged: Path,
    workspace: Path,
    plan: dict[str, Any],
    source_profile: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    start = int(plan["montage"]["ending_start_frame"])
    frames = int(plan["montage"]["ending_frames"])
    path = workspace / "ending.nut"
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
            "-vf",
            f"trim=start_frame={start}:end_frame={start + frames},setpts=PTS-STARTPTS",
            "-vsync",
            "0",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads:v",
            "1",
            # The FFV1/NUT stream carries source color tags separately. Forcing
            # -color_range here can alter decoded pixels before a lossless copy.
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
    piece = media.frame_hashes(path, pix_fmt=str(source_profile["pix_fmt"]))
    verified = bool(piece) and original[start : start + frames] == piece
    if not verified:
        raise RuntimeError("source->reveal FFV1/NUT decoded frame hashes diverged")
    return path, {
        "source_window_start_frame": start,
        "frame_count": len(piece),
        "source_to_reveal_hashes_exact": verified,
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
        "reveal_source_hashes_exact": stage["reveal"]["source_to_reveal_hashes_exact"],
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


def render(
    source: Path,
    profile: CampaignProfile,
    plan: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validate_plan(plan, source, profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "warzone_operator_toggle.mp4"
    output = profile.config["output"]
    with tempfile.TemporaryDirectory(prefix="clipper-montage-", dir=output_dir) as directory:
        workspace = Path(directory)
        staged = workspace / "source_lossless.nut"
        canonical = workspace / "canonical_montage.nut"
        title = workspace / "approved_title.txt"
        title.write_text(plan["montage"]["approved_on_screen_text"], encoding="utf-8")
        staging = ffv1.stage_native_source(source, staged)
        source_profile = staging["source_profile"]
        comparison, comparison_qa = _comparison_piece(
            staged, workspace, plan, profile, source_profile
        )
        ending, ending_qa = _ending_piece(staged, workspace, plan, source_profile)
        staging["comparison"] = comparison_qa
        staging["reveal"] = ending_qa
        graph = filter_graph(plan, profile, title, _fontfile())
        total = float(plan["montage"]["output_seconds"])
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
            raise RuntimeError("canonical visual montage is not exact on frame grid")
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
        "status": "PASS" if plan["source"]["certified"] else "PREVIEW_ONLY",
    }
    path = output_dir / "render_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
