from __future__ import annotations

import json
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from .. import media_contract as media
from ..gameplay import analysis as semantic
from . import ffv1, helpers, portrait_layout
from . import source_fidelity as source


def delivery_mode(config: dict[str, Any]) -> str:
    overlay = dict(config.get("text_overlay") or {})
    if not overlay.get("enabled", False):
        return "off"
    mode = str(overlay.get("delivery_mode") or "burned")
    if mode not in {"burned", "sidecar"}:
        raise RuntimeError("text_overlay.delivery_mode must be 'burned' or 'sidecar'")
    return mode


def enabled(config: dict[str, Any], plan: semantic.SemanticPlan) -> bool:
    return delivery_mode(config) == "burned" and bool(plan.headline.strip())


def sidecar_enabled(config: dict[str, Any], plan: semantic.SemanticPlan) -> bool:
    return delivery_mode(config) == "sidecar" and bool(plan.headline.strip())


def _srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def export_text_sidecars(
    video: Path,
    headline: str,
    config: dict[str, Any],
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    text = headline.strip()
    if not text:
        raise RuntimeError("editable hook text is required for text sidecar delivery")
    if not Decimal(str(duration_seconds)).is_finite() or duration_seconds <= 0:
        raise RuntimeError("encoded video duration is invalid for text sidecars")
    end_ms = int(
        (Decimal(str(duration_seconds)) * Decimal(1000)).to_integral_value(rounding=ROUND_FLOOR)
    )
    if end_ms <= 0:
        raise RuntimeError("encoded video is too short for text sidecars")

    hook_txt = video.with_name(f"{video.stem}.hook.txt")
    hook_srt = video.with_name(f"{video.stem}.hook.srt")
    hook_json = video.with_name(f"{video.stem}.hook.json")
    hook_txt.write_text(text + "\n", encoding="utf-8")
    hook_srt.write_text(
        f"1\n00:00:00,000 --> {_srt_timestamp(end_ms)}\n{text}\n",
        encoding="utf-8",
    )
    hook_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": video.name,
                "text": text,
                "start_ms": 0,
                "end_ms": end_ms,
                "burned_in_video": False,
                "editable": True,
                "recommended_style": {
                    "portrait_layout": config.get("output", {}).get("portrait_layout", {}),
                    "text_overlay": config.get("text_overlay", {}),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "generated": True,
        "burned_in_video": False,
        "source_video": video.name,
        "headline": text,
        "start_ms": 0,
        "end_ms": end_ms,
        "files": {
            suffix: {"name": path.name, "sha256": helpers.sha256(path)}
            for suffix, path in (
                ("txt", hook_txt),
                ("srt", hook_srt),
                ("json", hook_json),
            )
        },
    }


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _portrait_filter(
    headline: str,
    target: Path,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    font_path: Path,
) -> tuple[str, dict[str, Any]]:
    width = int(source_profile["width"])
    height = int(source_profile["height"])
    layout = portrait_layout.resolve(config, width, height)
    if layout is None:
        raise RuntimeError("portrait overlay requested without a portrait layout")
    rows = portrait_layout.title_lines(headline, font_path, layout)
    text_color = str(layout.get("text_hex") or "#FAFAFA").lstrip("#")
    accent = str(layout.get("accent_hex") or "#FFD848").lstrip("#")
    bar_width = round(width * float(layout.get("title_bar_width_fraction", 0.18)))
    bar_height = int(layout.get("title_bar_height", 5))
    bar_y = int(layout.get("title_bar_y", int(layout["title_top_height"]) - 90))
    filters: list[str] = []
    overlay_config = dict(config.get("text_overlay") or {})
    title_background_mode = str(overlay_config.get("title_background_mode") or "none")
    text_background_filter = ""
    if title_background_mode in {"solid", "text_box"}:
        background_hex = str(overlay_config.get("title_background_hex") or "#000000").lstrip("#")
        if len(background_hex) != 6 or any(
            char not in "0123456789abcdefABCDEF" for char in background_hex
        ):
            raise RuntimeError("text_overlay.title_background_hex must be a six-digit hex color")
        opacity = float(overlay_config.get("title_background_opacity", 1.0))
        if not 0.0 <= opacity <= 1.0:
            raise RuntimeError("text_overlay.title_background_opacity must be in 0..1")
        if title_background_mode == "solid":
            filters.append(
                "drawbox=x=0:y=0:w=iw:"
                f"h={int(layout['title_top_height'])}:"
                f"color=0x{background_hex}@{opacity:.4f}:t=fill"
            )
        else:
            padding = int(overlay_config.get("title_background_padding_px", 16))
            if padding < 0 or padding > 64:
                raise RuntimeError("text_overlay.title_background_padding_px must be in 0..64")
            text_background_filter = (
                f"box=1:boxcolor=0x{background_hex}@{opacity:.4f}:boxborderw={padding}:"
            )
    elif title_background_mode != "none":
        raise RuntimeError(
            "text_overlay.title_background_mode must be 'none', 'solid' or 'text_box'"
        )
    if layout.get("title_bar_enabled", True):
        filters.append(
            f"drawbox=x=(iw-{bar_width})/2:y={bar_y}:w={bar_width}:h={bar_height}:"
            f"color=0x{accent}:t=fill"
        )
    line_files: list[str] = []
    for index, row in enumerate(rows):
        line_path = target.with_name(f"{target.stem}_headline_{index}.txt")
        line_path.write_text(str(row["text"]), encoding="utf-8")
        line_files.append(str(line_path))
        filters.append(
            "drawtext="
            f"fontfile='{_escape_filter_path(font_path)}':"
            f"textfile='{_escape_filter_path(line_path)}':"
            f"fontcolor=0x{text_color}:fontsize={int(row['font_size'])}:"
            f"{text_background_filter}"
            f"x=(w-text_w)/2:y={int(row['y'])}:"
            "shadowcolor=black@0.35:shadowx=1:shadowy=2:fix_bounds=1"
        )
    return ",".join(filters), {
        "layout_mode": "portrait_title_safe",
        "title_lines": rows,
        "title_text_files": line_files,
        "title_top_height": int(layout["title_top_height"]),
        "title_background_mode": title_background_mode,
        "visual_top": int(layout["visual_top"]),
        "visual_height": int(layout["visual_height"]),
    }


def _legacy_filter(
    headline: str,
    target: Path,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    font_path: Path,
) -> tuple[str, dict[str, Any]]:
    overlay = dict(config.get("text_overlay") or {})
    width = int(source_profile["width"])
    height = int(source_profile["height"])
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
    headline_path.write_text(headline, encoding="utf-8")
    graph = (
        f"drawbox=x=0:y=0:w=iw:h={box_height}:color=black@{opacity:.4f}:t=fill,"
        "drawtext="
        f"fontfile='{_escape_filter_path(font_path)}':"
        f"textfile='{_escape_filter_path(headline_path)}':fontcolor=white:"
        f"fontsize={font_size}:x={margin_x}:y={text_y}:"
        "shadowcolor=black@0.8:shadowx=2:shadowy=2:fix_bounds=1"
    )
    return graph, {
        "layout_mode": "legacy_top_banner",
        "font_size": font_size,
        "box_height": box_height,
        "box_opacity": opacity,
    }


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
    font_path = Path(
        str(overlay.get("font_path") or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    )
    if not font_path.is_file():
        raise RuntimeError(f"configured text overlay font is unavailable: {font_path}")

    if (
        portrait_layout.resolve(
            config,
            int(source_profile["width"]),
            int(source_profile["height"]),
        )
        is not None
    ):
        filter_graph, layout_info = _portrait_filter(
            plan.headline.strip(),
            target,
            config,
            source_profile,
            font_path,
        )
    else:
        filter_graph, layout_info = _legacy_filter(
            plan.headline.strip(),
            target,
            config,
            source_profile,
            font_path,
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
        **layout_info,
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
        "video_operations": "declared title-safe text overlay only",
    }
