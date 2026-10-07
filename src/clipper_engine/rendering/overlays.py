from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import media_contract as media
from ..gameplay import analysis as semantic
from . import ffv1
from . import source_fidelity as source


def enabled(config: dict[str, Any], plan: semantic.SemanticPlan) -> bool:
    overlay = dict(config.get("text_overlay") or {})
    return bool(overlay.get("enabled") and plan.headline.strip())


def render_text_overlay_master(
    canonical_source_master: Path,
    plan: semantic.SemanticPlan,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    if not enabled(config, plan):
        raise RuntimeError("text overlay master requested for a plan with no enabled headline")
    if target.suffix.lower() != ".nut":
        raise RuntimeError("editorial canonical master must use NUT transport")

    overlay = dict(config.get("text_overlay") or {})
    width = int(source_profile["width"])
    height = int(source_profile["height"])
    font_path = Path(
        str(overlay.get("font_path") or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    )
    if not font_path.is_file():
        raise RuntimeError(f"configured text overlay font is unavailable: {font_path}")

    font_size = max(24, round(height * float(overlay.get("font_size_fraction", 0.05))))
    margin_x = max(24, round(width * float(overlay.get("horizontal_margin_fraction", 0.04))))
    box_height = max(
        font_size + 28,
        round(height * float(overlay.get("box_height_fraction", 0.14))),
    )
    text_y = max(8, round((box_height - font_size) / 2.0))
    opacity = float(overlay.get("box_opacity", 0.72))
    if not 0.0 <= opacity <= 1.0:
        raise RuntimeError("text_overlay.box_opacity must be in 0..1")

    headline_path = target.with_name(target.stem + "_headline.txt")
    headline_path.write_text(plan.headline.strip(), encoding="utf-8")
    filter_graph = (
        f"drawbox=x=0:y=0:w=iw:h={box_height}:color=black@{opacity:.4f}:t=fill,"
        f"drawtext=fontfile={font_path}:textfile={headline_path}:fontcolor=white:"
        f"fontsize={font_size}:x={margin_x}:y={text_y}:"
        "shadowcolor=black@0.8:shadowx=2:shadowy=2:fix_bounds=1"
    )
    forbidden = ("scale=", "crop=", "zscale=", "zoompan=", "perspective=", "rotate=")
    if any(item in filter_graph.lower() for item in forbidden):
        raise RuntimeError("editorial overlay graph contains a forbidden spatial transform")

    timing = source._timing(source_profile)
    fps = media.fraction_text(timing.nominal_rate)
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(canonical_source_master),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            filter_graph,
            *ffv1._ffv1_video_args(
                source_profile,
                threads=int(config.get("runtime", {}).get("ffv1_threads", 4)),
            ),
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(source._audio_rate(source_profile)),
            "-ac",
            str(source._audio_channels(source_profile)),
            "-r",
            fps,
            "-vsync",
            "cfr",
            "-f",
            "nut",
            str(target),
        ]
    )

    original_profile = media.video_profile(canonical_source_master, count_frames=True)
    overlay_profile = media.video_profile(target, count_frames=True)
    if original_profile["frame_count"] != overlay_profile["frame_count"]:
        raise RuntimeError(
            "text overlay changed canonical frame cardinality: "
            f"source={original_profile['frame_count']} overlay={overlay_profile['frame_count']}"
        )
    architecture = ffv1._transport_architecture_qa(
        target,
        source_profile,
        prefix="editorial_overlay",
        strict_cfr=True,
    )
    if not all(bool(value) for value in architecture["checks"].values()):
        raise RuntimeError(f"editorial FFV1/NUT overlay transport QA failed: {architecture}")

    return {
        "applied": True,
        "headline": plan.headline,
        "font_path": str(font_path),
        "font_size": font_size,
        "box_height": box_height,
        "box_opacity": opacity,
        "spatial_transform_used": False,
        "input_frame_count": original_profile["frame_count"],
        "output_frame_count": overlay_profile["frame_count"],
        "canonical_container": "nut",
        "canonical_codec": "ffv1 level=3 lossless",
        "canonical_audio_codec": "pcm_s16le",
        "actual_container": architecture["format_name"],
        "actual_video_codec": architecture["video_profile"]["codec_name"],
        "actual_audio_codec": architecture["audio_profile"]["codec_name"],
        "source_delivery_track_timescale": timing.track_timescale,
        "source_delivery_time_base": media.fraction_text(timing.time_base),
        "transport_time_base": architecture["timing"]["actual_time_base"],
        "source_frame_period": architecture["timing"]["source_frame_period"],
        "transport_architecture_qa": architecture,
        "source_contract_metadata_authority": "original_input_probe",
        "nut_color_metadata_authoritative": False,
        "nut_non_authoritative_metadata_fields": list(ffv1.NUT_NON_AUTHORITATIVE_VIDEO_METADATA),
        "source_profile": source_profile,
        "video_operations": "source-native edit + declared text overlay only",
    }
