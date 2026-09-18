from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import mw4_v3_1_media_contract as media
from . import mw4_v3_1_render_one_mov as base

_ACTIVE_SOURCE_PROFILE: dict[str, Any] | None = None


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


def _verify_lossless_piece(
    source: Path,
    piece: Path,
    *,
    extract_start: float,
    extract_duration: float,
    source_profile: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    source_hashes = media.frame_hashes(
        source,
        pix_fmt=str(source_profile["pix_fmt"]),
        start=extract_start,
        duration=extract_duration,
    )
    piece_hashes = media.frame_hashes(piece, pix_fmt=str(source_profile["pix_fmt"]))
    piece_profile = media.video_profile(piece)
    checks = {
        "exact_decoded_frame_count_match": len(source_hashes) == len(piece_hashes),
        "exact_decoded_frame_hash_match": source_hashes == piece_hashes,
        "source_interval_has_frames": bool(source_hashes),
        **media.metadata_match_checks(source_profile, piece_profile, prefix="stage"),
    }
    timing_qa = media.verify_cfr_timeline(
        piece,
        _timing(source_profile),
        label=f"lossless segment {index + 1}",
    )
    checks.update({f"timing_{key}": value for key, value in timing_qa["checks"].items()})
    if not all(checks.values()):
        mismatch = next(
            (
                item
                for item, pair in enumerate(zip(source_hashes, piece_hashes, strict=False))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise RuntimeError(
            f"lossless source->stage fidelity failed for segment {index + 1}: "
            f"checks={checks}; source_frames={len(source_hashes)} stage_frames={len(piece_hashes)} "
            f"first_hash_mismatch={mismatch}; source_profile={source_profile}; "
            f"stage_profile={piece_profile}"
        )
    return {
        "segment": index + 1,
        "extract_start": extract_start,
        "extract_duration": extract_duration,
        "frame_count": len(source_hashes),
        "checks": checks,
        "stage_profile": piece_profile,
        "timing": timing_qa,
    }


def _verify_stitched_lossless(
    pieces: list[Path],
    stitched: Path,
    source_profile: dict[str, Any],
) -> dict[str, Any]:
    expected_hashes: list[str] = []
    for piece in pieces:
        expected_hashes.extend(media.frame_hashes(piece, pix_fmt=str(source_profile["pix_fmt"])))
    stitched_hashes = media.frame_hashes(stitched, pix_fmt=str(source_profile["pix_fmt"]))
    stitched_profile = media.video_profile(stitched)
    checks = {
        "exact_decoded_frame_count_match": len(expected_hashes) == len(stitched_hashes),
        "exact_decoded_frame_hash_match": expected_hashes == stitched_hashes,
        **media.metadata_match_checks(source_profile, stitched_profile, prefix="stitched"),
    }
    timing_qa = media.verify_monotonic_timeline(
        stitched,
        _timing(source_profile),
        label="stitched lossless transport",
    )
    checks.update({f"timing_{key}": value for key, value in timing_qa["checks"].items()})
    if not all(checks.values()):
        raise RuntimeError(
            "lossless piece->stitched fidelity failed: "
            f"checks={checks}; expected_frames={len(expected_hashes)} "
            f"stitched_frames={len(stitched_hashes)}; "
            f"source_profile={source_profile}; stitched_profile={stitched_profile}"
        )
    return {
        "segment": "stitched",
        "frame_count": len(stitched_hashes),
        "checks": checks,
        "stage_profile": stitched_profile,
        "timing": timing_qa,
    }


def _stage_plan_source(
    source: Path,
    plan: base.semantic.SemanticPlanV31,
    workspace: Path,
    config: dict[str, Any],
) -> tuple[Path, base.semantic.SemanticPlanV31, dict[str, Any], list[dict[str, Any]]]:
    global _ACTIVE_SOURCE_PROFILE

    workspace.mkdir(parents=True, exist_ok=True)
    contract = media.inspect_source(source, verify_timeline=True)
    media.validate_delivery_compatibility(contract, config)
    source_profile = _profile_with_contract(contract)
    _ACTIVE_SOURCE_PROFILE = dict(source_profile)

    source_info = base.legacy.renderer.probe(source)
    source_duration = float(source_info["format"]["duration"])
    timing = contract.video.timing

    pieces: list[Path] = []
    mappings: list[tuple[base.semantic.EditSegment, base.semantic.EditSegment]] = []
    staging_fidelity: list[dict[str, Any]] = []
    cursor = 0.0
    segments = list(plan.segments)
    if not segments:
        raise RuntimeError("approved plan has no source segments")

    for index, segment in enumerate(segments):
        previous = segments[index - 1] if index else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        noncontiguous_before = bool(
            previous is not None
            and (segment.start > previous.end + 0.02 or segment.start < previous.start - 0.02)
        )
        noncontiguous_after = bool(
            following is None
            or following.start > segment.end + 0.02
            or following.start < segment.start - 0.02
        )
        lead = min(0.05, segment.start) if noncontiguous_before else 0.0
        tail = min(0.05, max(0.0, source_duration - segment.end)) if noncontiguous_after else 0.0
        extract_start = max(0.0, segment.start - lead)
        extract_end = min(source_duration, segment.end + tail)
        extract_duration = extract_end - extract_start
        if extract_duration <= 0.0:
            raise RuntimeError(f"invalid staged segment {index + 1}: {segment.start}-{segment.end}")

        piece = workspace / f"segment_{index:02d}.mov"
        media.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{extract_start:.6f}",
                "-i",
                str(source),
                "-t",
                f"{extract_duration:.6f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                *media.lossless_video_args(source_profile),
                "-vsync",
                "0",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(contract.audio.sample_rate),
                "-ac",
                str(contract.audio.channels),
                "-f",
                "mov",
                str(piece),
            ]
        )
        staging_fidelity.append(
            _verify_lossless_piece(
                source,
                piece,
                extract_start=extract_start,
                extract_duration=extract_duration,
                source_profile=source_profile,
                index=index,
            )
        )

        piece_duration = float(base.legacy.renderer.probe(piece)["format"]["duration"])
        local_start = cursor + (segment.start - extract_start)
        local_end = local_start + (segment.end - segment.start)
        if local_end > cursor + piece_duration + 0.025:
            raise RuntimeError(
                f"staged segment {index + 1} is shorter than approved source interval: "
                f"need {local_end - cursor:.3f}s, have {piece_duration:.3f}s"
            )
        local_segment = base.semantic.EditSegment(
            start=round(local_start, 6),
            end=round(local_end, 6),
            speed=segment.speed,
            reason=segment.reason,
        )
        mappings.append((segment, local_segment))
        pieces.append(piece)
        cursor += piece_duration

    concat_list = workspace / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{piece.name}'\n" for piece in pieces),
        encoding="utf-8",
    )
    stitched = workspace / "approved_segments_lossless.mov"
    media.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            "-video_track_timescale",
            str(timing.track_timescale),
            "-f",
            "mov",
            str(stitched),
        ]
    )
    staging_fidelity.append(_verify_stitched_lossless(pieces, stitched, source_profile))

    local_events = []
    for event in plan.effect_events:
        mapped = base.legacy._map_source_time(event.time, mappings)
        if mapped is not None:
            local_events.append(replace(event, time=round(mapped, 6)))

    local_finishing = None
    if plan.finishing_move is not None:
        start = base.legacy._map_source_time(plan.finishing_move.start, mappings)
        payoff = base.legacy._map_source_time(plan.finishing_move.payoff, mappings)
        end = base.legacy._map_source_time(plan.finishing_move.end, mappings)
        if start is None or payoff is None or end is None:
            raise RuntimeError("verified Finishing Move was not fully preserved by staged source")
        local_finishing = replace(
            plan.finishing_move,
            start=round(start, 6),
            payoff=round(payoff, 6),
            end=round(end, 6),
        )

    local_plan = replace(
        plan,
        start=min(item.start for _, item in mappings),
        end=max(item.end for _, item in mappings),
        segments=tuple(item for _, item in mappings),
        effect_events=tuple(local_events),
        finishing_move=local_finishing,
    )
    return stitched, local_plan, source_profile, staging_fidelity


def _append_source_native_segment(
    parts: list[str],
    index: int,
    segment: base.semantic.EditSegment,
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
    plan: base.semantic.SemanticPlanV31,
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


def _render_canonical_lossless_master(
    staged_source: Path,
    plan: base.semantic.SemanticPlanV31,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    graph, duration = _build_source_native_filter(plan, source_profile)
    timing = _timing(source_profile)
    fps = media.fraction_text(timing.nominal_rate)

    media.run(
        [
            "ffmpeg",
            "-y",
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
            "-map",
            "[aout]",
            *media.lossless_video_args(source_profile),
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(_audio_rate(source_profile)),
            "-ac",
            str(_audio_channels(source_profile)),
            "-r",
            fps,
            "-vsync",
            "cfr",
            "-t",
            f"{duration:.6f}",
            "-f",
            "mov",
            str(target),
        ]
    )

    expected_hashes = _canonical_filter_hashes(
        staged_source,
        graph,
        str(source_profile["pix_fmt"]),
        fps=fps,
        duration=duration,
    )
    master_hashes = media.frame_hashes(target, pix_fmt=str(source_profile["pix_fmt"]))
    if expected_hashes != master_hashes:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(expected_hashes, master_hashes, strict=False))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise RuntimeError(
            "canonical CFR reference differs from lossless x264/MOV decoded pixels: "
            f"expected_frames={len(expected_hashes)} master_frames={len(master_hashes)} "
            f"first_hash_mismatch={mismatch}"
        )

    master_profile = media.video_profile(target)
    metadata_checks = media.metadata_match_checks(
        source_profile, master_profile, prefix="canonical"
    )
    if not all(metadata_checks.values()):
        raise RuntimeError(
            f"canonical master media metadata differs from source: checks={metadata_checks}; "
            f"source={source_profile}; canonical={master_profile}"
        )
    timing_qa = media.verify_cfr_timeline(target, timing, label="canonical lossless master")

    return {
        "graph_has_spatial_transform": False,
        "video_operations": "trim/setpts/concat + source-SAR metadata only",
        "canonical_fps": fps,
        "canonical_container": "mov",
        "canonical_codec": "libx264 qp=0 lossless",
        "video_track_timescale": timing.track_timescale,
        "frame_ticks": timing.ticks_per_frame,
        "time_base": media.fraction_text(timing.time_base),
        "exact_filtered_frame_hash_match": True,
        "timing": timing_qa,
        "source_profile": source_profile,
    }


def _encode_from_canonical_master(
    canonical_master: Path,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
    *,
    mode: str,
) -> None:
    timing = _timing(source_profile)
    encoder_args = base.legacy.renderer._encode_args(config, mode)
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


def _source_fidelity_qa(
    source_profile: dict[str, Any],
    staging_fidelity: list[dict[str, Any]],
    canonical_master: Path,
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    timing = _timing(source_profile)
    reference_profile = media.video_profile(canonical_master, count_frames=True)
    output_profile = media.video_profile(output, count_frames=True)
    reference_audio = media.audio_profile(canonical_master)
    output_audio = media.audio_profile(output)
    reference_timing = media.verify_cfr_timeline(
        canonical_master, timing, label="canonical fidelity reference"
    )
    output_timing = media.verify_cfr_timeline(output, timing, label="final H.264 output")

    settings = config["output"]
    expected_audio_rate = int(settings.get("audio_sample_rate") or _audio_rate(source_profile))
    expected_audio_channels = int(settings.get("audio_channels") or _audio_channels(source_profile))
    source_geometry = (int(source_profile["width"]), int(source_profile["height"]))
    checks = {
        "reference_geometry_matches_source": (
            reference_profile["width"],
            reference_profile["height"],
        )
        == source_geometry,
        "output_geometry_matches_source": (output_profile["width"], output_profile["height"])
        == source_geometry,
        "reference_time_base_matches_source": reference_timing["checks"][
            "time_base_matches_source"
        ],
        "reference_r_fps_matches_source": reference_timing["checks"]["r_frame_rate_matches_source"],
        "reference_strict_cfr_timestamps": reference_timing["checks"]["packet_pts_strictly_cfr"],
        "output_time_base_matches_source": output_timing["checks"]["time_base_matches_source"],
        "output_r_fps_matches_source": output_timing["checks"]["r_frame_rate_matches_source"],
        "output_strict_cfr_timestamps": output_timing["checks"]["packet_pts_strictly_cfr"],
        "exact_frame_count_match": reference_profile["frame_count"]
        == output_profile["frame_count"],
        "all_source_to_stage_frame_hashes_exact": all(
            item["checks"]["exact_decoded_frame_hash_match"] for item in staging_fidelity
        ),
        "canonical_audio_sample_rate_matches_source": reference_audio["sample_rate"]
        == _audio_rate(source_profile),
        "canonical_audio_channels_match_source": reference_audio["channels"]
        == _audio_channels(source_profile),
        "output_audio_sample_rate_matches_delivery": output_audio["sample_rate"]
        == expected_audio_rate,
        "output_audio_channels_match_delivery": output_audio["channels"] == expected_audio_channels,
    }
    checks.update(
        media.metadata_match_checks(source_profile, reference_profile, prefix="canonical")
    )
    checks.update(media.metadata_match_checks(source_profile, output_profile, prefix="output"))
    if not all(checks.values()):
        raise RuntimeError(
            "source-fidelity metadata/timeline mismatch before SSIM/PSNR: "
            f"checks={checks}; source={source_profile}; reference={reference_profile}; "
            f"output={output_profile}; reference_timing={reference_timing}; output_timing={output_timing}"
        )

    ssim = _metric(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(canonical_master),
            "-i",
            str(output),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[enc];[ref][enc]ssim[metric]",
            "-map",
            "[metric]",
            "-an",
            "-f",
            "null",
            "-",
        ],
        r"All:([0-9.]+)",
        "SSIM",
    )
    psnr = _metric(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(canonical_master),
            "-i",
            str(output),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[enc];[ref][enc]psnr[metric]",
            "-map",
            "[metric]",
            "-an",
            "-f",
            "null",
            "-",
        ],
        r"average:([0-9.]+)",
        "PSNR",
    )
    fidelity_cfg = config.get("source_fidelity", {})
    minimum_ssim = float(fidelity_cfg.get("minimum_ssim", 0.99))
    minimum_psnr = float(fidelity_cfg.get("minimum_psnr_db", 40.0))
    checks.update(
        {
            "no_spatial_crop_or_upscale": True,
            "single_canonical_visual_timeline": True,
            "ssim_encoder_fidelity": ssim >= minimum_ssim,
            "psnr_encoder_fidelity": psnr >= minimum_psnr,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(
            f"source-fidelity QA failed: SSIM={ssim:.6f} minimum={minimum_ssim:.6f}; "
            f"PSNR={psnr:.3f}dB minimum={minimum_psnr:.3f}dB; "
            f"reference_frames={reference_profile['frame_count']} output_frames={output_profile['frame_count']}"
        )
    return {
        "checks": checks,
        "ssim": ssim,
        "minimum_ssim": minimum_ssim,
        "psnr_db": psnr,
        "minimum_psnr_db": minimum_psnr,
        "source_profile": source_profile,
        "staging_fidelity": staging_fidelity,
        "reference_profile": reference_profile,
        "output_profile": output_profile,
        "reference_timing": reference_timing,
        "output_timing": output_timing,
        "reference": (
            "source-derived media/timing contract; exact source->stage decoded-frame hashes; "
            "exact source time base and CFR packet cadence on canonical/final timelines; "
            "final H.264 generation loss measured only against the canonical lossless master"
        ),
    }


def _validate_output(
    path: Path, config: dict[str, Any], *, mode: str = "production"
) -> dict[str, Any]:
    if _ACTIVE_SOURCE_PROFILE is None:
        raise RuntimeError("final output validation has no active source media contract")
    source_profile = _ACTIVE_SOURCE_PROFILE
    timing = _timing(source_profile)
    data = base.legacy.renderer.probe(path)
    video = next(item for item in data["streams"] if item["codec_type"] == "video")
    audio = next(item for item in data["streams"] if item["codec_type"] == "audio")
    fmt = data["format"]
    profile = media.video_profile(path)
    timeline = media.verify_cfr_timeline(path, timing, label="delivery output")
    settings = config["output"]
    duration = float(fmt["duration"])
    bitrate = int(fmt.get("bit_rate") or video.get("bit_rate") or 0)
    expected_codec = (
        "h264" if str(settings["codec"]).lower() == "libx264" else str(settings["codec"])
    )
    checks = {
        "duration_10_to_20s": 10.0 <= duration <= 20.0,
        "full_frame_geometry_matches_source": (int(video["width"]), int(video["height"]))
        == (int(source_profile["width"]), int(source_profile["height"])),
        "codec_matches_delivery": str(video["codec_name"]) == expected_codec,
        "profile_matches_delivery": str(video.get("profile", "")).lower()
        == str(settings.get("profile", "")).lower(),
        "time_base_matches_source": timeline["checks"]["time_base_matches_source"],
        "r_frame_rate_matches_source": timeline["checks"]["r_frame_rate_matches_source"],
        "strict_cfr_timestamps": timeline["checks"]["packet_pts_strictly_cfr"],
        "audio_sample_rate_matches_delivery": int(audio["sample_rate"])
        == int(settings["audio_sample_rate"]),
        "audio_channels_match_delivery": int(audio["channels"]) == int(settings["audio_channels"]),
        **media.metadata_match_checks(source_profile, profile, prefix="delivery"),
    }
    if mode == "production":
        expected = int(settings["video_bitrate_kbps"]) * 1000
        checks["high_bitrate_near_target"] = bitrate >= int(expected * 0.94)
    media.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if not all(checks.values()):
        raise RuntimeError(
            f"QA failed for {path.name}: {checks}; source_profile={source_profile}; "
            f"output_profile={profile}; timing={timeline}; bitrate={bitrate}"
        )
    return {"checks": checks, "probe": data, "timing": timeline}


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


def _test_plan() -> base.semantic.SemanticPlanV31:
    segments = (
        base.semantic.EditSegment(0.10, 0.72, 1.0, "self_test_a"),
        base.semantic.EditSegment(0.95, 1.75, 1.35, "self_test_b"),
    )
    output_duration = sum((item.end - item.start) / item.speed for item in segments)
    return base.semantic.SemanticPlanV31(
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


def _self_test_case(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    source = root / f"{name}_source.mov"
    _synthetic_source(source, **spec)
    contract = media.inspect_source(source, verify_timeline=True)
    if contract.video.width != spec["width"] or contract.video.height != spec["height"]:
        raise AssertionError(f"{name}: source geometry was not derived correctly")
    if contract.video.timing.track_timescale != spec["timescale"]:
        raise AssertionError(f"{name}: source timescale was not derived correctly")
    if (
        contract.audio.sample_rate != spec["audio_rate"]
        or contract.audio.channels != spec["channels"]
    ):
        raise AssertionError(f"{name}: source audio contract was not derived correctly")

    config = {
        "output": {
            "width": spec["width"],
            "height": spec["height"],
            "full_source_frame": True,
            "fps": spec["fps"],
            "codec": "libx264",
            "profile": "high",
            "preset": "ultrafast",
            "video_bitrate_kbps": 12000,
            "audio_codec": "aac",
            "audio_bitrate_kbps": 192,
            "audio_sample_rate": spec["audio_rate"],
            "audio_channels": spec["channels"],
        },
        "source_fidelity": {"minimum_ssim": 0.99, "minimum_psnr_db": 40.0},
    }
    plan = _test_plan()
    workspace = root / f"{name}_work"
    staged, local_plan, source_profile, staging = _stage_plan_source(
        source, plan, workspace, config
    )
    canonical = root / f"{name}_canonical.mov"
    _render_canonical_lossless_master(staged, local_plan, config, source_profile, canonical)
    output = root / f"{name}_final.mp4"
    _encode_from_canonical_master(canonical, config, source_profile, output, mode="production")
    qa = _source_fidelity_qa(source_profile, staging, canonical, output, config)
    return {
        "source": contract.to_json(),
        "ssim": qa["ssim"],
        "psnr_db": qa["psnr_db"],
        "output_timing": qa["output_timing"],
    }


def preflight() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mw4_media_contract_preflight_") as tmp:
        root = Path(tmp)
        cases = {
            "ntsc_5994_bt709": {
                "width": 128,
                "height": 72,
                "fps": "60000/1001",
                "timescale": 60000,
                "audio_rate": 48000,
                "channels": 2,
                "color": "bt709",
            },
            "ntsc_2997_smpte170m": {
                "width": 160,
                "height": 90,
                "fps": "30000/1001",
                "timescale": 30000,
                "audio_rate": 44100,
                "channels": 1,
                "color": "smpte170m",
            },
        }
        results = {name: _self_test_case(root, name, spec) for name, spec in cases.items()}
        if (
            results["ntsc_5994_bt709"]["source"]["video"]["track_timescale"]
            == results["ntsc_2997_smpte170m"]["source"]["video"]["track_timescale"]
        ):
            raise AssertionError(
                "preflight did not exercise distinct source-derived timing contracts"
            )
        print(
            json.dumps(
                {"source_derived_renderer_preflight": "PASS", "cases": results}, sort_keys=True
            )
        )
        return results


def _install() -> None:
    base.lossless.preflight = preflight
    base._stage_plan_source = _stage_plan_source
    base._verify_stitched_lossless = _verify_stitched_lossless
    base._render_canonical_lossless_master = _render_canonical_lossless_master
    base._source_fidelity_qa = _source_fidelity_qa
    base.legacy._build_source_native_filter = _build_source_native_filter
    base.legacy._encode_from_canonical_master = _encode_from_canonical_master
    base.legacy.renderer.validate_output = _validate_output


def main() -> None:
    if "--self-test" in sys.argv:
        preflight()
        return
    _install()
    base.main()


if __name__ == "__main__":
    main()
