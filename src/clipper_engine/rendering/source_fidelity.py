from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .. import media_contract as media
from ..gameplay import analysis as semantic
from . import helpers as renderer


def _profile_with_contract(contract: media.SourceMediaContract) -> dict[str, Any]:
    profile = contract.video.profile_dict()
    profile.update(
        {
            "_audio_sample_rate": contract.audio.sample_rate,
            "_audio_channels": contract.audio.channels,
            "_audio_channel_layout": contract.audio.channel_layout,
            "_audio_sample_fmt": contract.audio.sample_fmt,
        }
    )
    return profile


def _timing(profile: dict[str, Any]) -> media.VideoTimingContract:
    return media.timing_from_profile(profile)


def _audio_rate(profile: dict[str, Any]) -> int:
    value = int(profile.get("_audio_sample_rate") or 0)
    if value <= 0:
        raise RuntimeError("source audio sample rate is unavailable from the source contract")
    return value


def _audio_channels(profile: dict[str, Any]) -> int:
    value = int(profile.get("_audio_channels") or 0)
    if value <= 0:
        raise RuntimeError("source audio channel count is unavailable from the source contract")
    return value


def _map_source_time(
    value: float,
    mappings: list[tuple[semantic.EditSegment, semantic.EditSegment]],
) -> float | None:
    for original, local in mappings:
        if original.start - 1e-6 <= value <= original.end + 1e-6:
            return local.start + (value - original.start)
    return None


def _append_source_native_segment(
    parts: list[str],
    index: int,
    segment: semantic.EditSegment,
) -> tuple[str, str]:
    video = f"snv{index}"
    audio = f"sna{index}"
    video_pts = (
        "PTS-STARTPTS" if abs(segment.speed - 1.0) < 1e-6 else f"(PTS-STARTPTS)/{segment.speed:.6f}"
    )
    parts.append(
        f"[0:v]trim=start={segment.start:.6f}:end={segment.end:.6f},setpts={video_pts}[{video}]"
    )
    audio_chain = f"[0:a]atrim=start={segment.start:.6f}:end={segment.end:.6f},asetpts=PTS-STARTPTS"
    if abs(segment.speed - 1.0) >= 1e-6:
        audio_chain += f",atempo={segment.speed:.6f}"
    parts.append(audio_chain + f"[{audio}]")
    return video, audio


def _build_source_native_filter(
    plan: semantic.SemanticPlan,
    source_profile: dict[str, Any],
) -> tuple[str, float]:
    parts: list[str] = []
    pairs = [
        _append_source_native_segment(parts, index, segment)
        for index, segment in enumerate(plan.segments)
    ]
    if not pairs:
        raise RuntimeError("approved plan has no edit segments")

    sar = str(source_profile.get("sample_aspect_ratio") or "")
    sar_filter = "null" if not media.is_known(sar) else f"setsar={sar.replace(':', '/')}"
    audio_rate = _audio_rate(source_profile)
    if len(pairs) == 1:
        parts.append(f"[{pairs[0][0]}]{sar_filter}[outv]")
        parts.append(f"[{pairs[0][1]}]aresample={audio_rate}[aout]")
    else:
        concat_inputs = "".join(f"[{video}][{audio}]" for video, audio in pairs)
        parts.append(f"{concat_inputs}concat=n={len(pairs)}:v=1:a=1[vcat][acat]")
        parts.append(f"[vcat]{sar_filter}[outv]")
        parts.append(f"[acat]aresample={audio_rate}[aout]")

    graph = ";".join(parts)
    forbidden = ("crop=", "scale=", "zscale=", "zoompan=", "perspective=", "rotate=")
    found = [token for token in forbidden if token in graph.lower()]
    if found:
        raise RuntimeError(
            f"source-native canonical graph contains forbidden spatial transform(s): {found}"
        )
    return graph, plan.output_duration


def _canonical_filter_hashes(
    staged_source: Path,
    graph: str,
    pix_fmt: str,
    *,
    fps: str,
    duration: float,
) -> list[str]:
    completed = media.run_capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-filter_complex_threads",
            "2",
            "-i",
            str(staged_source),
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            fps,
            "-vsync",
            "cfr",
            "-t",
            f"{duration:.6f}",
            "-pix_fmt",
            pix_fmt,
            "-f",
            "framemd5",
            "-",
            "-map",
            "[aout]",
            "-vn",
            "-f",
            "null",
            os.devnull,
        ]
    )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _encode_from_canonical_master(
    canonical_master: Path,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
    *,
    mode: str,
) -> None:
    timing = _timing(source_profile)
    encoder_args = renderer._encode_args(config, mode)
    if str(config.get("output", {}).get("codec", "")).lower() == "libx264" or mode == "shadow":
        encoder_args = media.merge_x264_params(encoder_args, media.x264_vui_params(source_profile))
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(canonical_master),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            *encoder_args,
            *media.profile_output_args(source_profile),
            "-video_track_timescale",
            str(timing.track_timescale),
            "-threads:v",
            "4",
            "-vsync",
            "0",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def _metric(command: list[str], pattern: str, label: str) -> float:
    completed = media.run_capture(command)
    matches = re.findall(pattern, completed.stderr)
    if not matches:
        raise RuntimeError(f"unable to parse {label} from FFmpeg fidelity QA")
    return float(matches[-1])


def _synthetic_source(
    target: Path,
    *,
    width: int,
    height: int,
    fps: str,
    timescale: int,
    audio_rate: int,
    channels: int,
    color: str,
) -> None:
    x264 = f"fullrange=off:colorprim={color}:transfer={color}:colormatrix={color}:chromaloc=0"
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration=2.0",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=997:sample_rate={audio_rate}:duration=2.0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            color,
            "-color_trc",
            color,
            "-color_primaries",
            color,
            "-chroma_sample_location",
            "left",
            "-x264-params",
            x264,
            "-video_track_timescale",
            str(timescale),
            "-c:a",
            "aac",
            "-ar",
            str(audio_rate),
            "-ac",
            str(channels),
            "-f",
            "mov",
            str(target),
        ]
    )


def _test_plan() -> semantic.SemanticPlan:
    segments = (
        semantic.EditSegment(0.10, 0.72, 1.0, "self_test_a"),
        semantic.EditSegment(0.95, 1.75, 1.35, "self_test_b"),
    )
    output_duration = sum((item.end - item.start) / item.speed for item in segments)
    return semantic.SemanticPlan(
        start=segments[0].start,
        end=segments[-1].end,
        raw_duration=sum(item.end - item.start for item in segments),
        output_duration=output_duration,
        score=1.0,
        retention_quality=1.0,
        payoff_quality=1.0,
        opening_quality=1.0,
        ending_quality=1.0,
        story_coherence=1.0,
        weakest_quarter_interest=1.0,
        low_interest_fraction=0.0,
        max_unexplained_low_interest_run_seconds=0.0,
        story_type="self_test",
        effect_profile="source_native_full_frame",
        segments=segments,
        effect_events=(),
        engagements=(),
        finishing_move=None,
        editorial_reasons=("renderer media-contract self-test",),
    )
