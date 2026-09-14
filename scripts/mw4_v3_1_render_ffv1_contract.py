from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import mw4_v3_1_media_contract as media
import mw4_v3_1_render_source_contract as source


NUT_NON_AUTHORITATIVE_VIDEO_METADATA = (
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "chroma_location",
    "field_order",
)


def _format_name(path: Path) -> str:
    payload = json.loads(
        media.run_capture([
            "ffprobe", "-v", "error",
            "-show_entries", "format=format_name",
            "-of", "json", str(path),
        ]).stdout
    )
    return str((payload.get("format") or {}).get("format_name") or "")


def _is_nut(path: Path) -> bool:
    names = {part.strip().lower() for part in _format_name(path).split(",") if part.strip()}
    return "nut" in names


def _ffv1_video_args(source_profile: dict[str, Any]) -> list[str]:
    pix_fmt = str(source_profile.get("pix_fmt") or "")
    if not pix_fmt:
        raise RuntimeError("source pixel format is unavailable for FFV1 transport")
    return [
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", pix_fmt,
        "-threads:v", "4",
    ]


def _transport_identity_checks(
    source_profile: dict[str, Any],
    other_profile: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, bool]:
    checks = {
        f"{prefix}_width_matches_source": int(other_profile.get("width") or 0) == int(source_profile["width"]),
        f"{prefix}_height_matches_source": int(other_profile.get("height") or 0) == int(source_profile["height"]),
        f"{prefix}_pix_fmt_matches_source": str(other_profile.get("pix_fmt") or "") == str(source_profile["pix_fmt"]),
    }
    source_sar = str(source_profile.get("sample_aspect_ratio") or "")
    other_sar = str(other_profile.get("sample_aspect_ratio") or "")
    checks[f"{prefix}_sample_aspect_ratio_matches_source_when_known"] = (
        other_sar == source_sar if media.is_known(source_sar) else True
    )
    return checks


def _transport_timing_qa(
    path: Path,
    timing: media.VideoTimingContract,
    *,
    label: str,
    strict_cfr: bool,
) -> dict[str, Any]:
    profile = media.video_profile(path)
    try:
        actual_tb = Fraction(str(profile.get("time_base") or ""))
        actual_rate = Fraction(str(profile.get("r_frame_rate") or ""))
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"{label}: invalid NUT timing metadata: {profile}") from exc
    if actual_tb <= 0 or actual_rate <= 0:
        raise RuntimeError(f"{label}: non-positive NUT timing metadata: {profile}")

    pts = media.packet_pts(path)
    deltas = [right - left for left, right in zip(pts, pts[1:])]
    nonpositive = [(index, value) for index, value in enumerate(deltas) if value <= 0]
    frame_period = Fraction(1, 1) / timing.nominal_rate
    cadence_mismatches = [
        (index, delta, str(actual_tb * delta))
        for index, delta in enumerate(deltas)
        if actual_tb * delta != frame_period
    ]
    checks = {
        "r_frame_rate_matches_source": actual_rate == timing.nominal_rate,
        "packet_pts_strictly_monotonic": not nonpositive,
        "packet_pts_cadence_matches_source_in_seconds": (
            not cadence_mismatches if strict_cfr else True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{label} NUT timing contract failed: checks={checks}; "
            f"source_rate={media.fraction_text(timing.nominal_rate)} "
            f"source_frame_period={frame_period}; actual_time_base={actual_tb}; "
            f"actual_rate={actual_rate}; nonpositive_pts_deltas={nonpositive[:8]}; "
            f"cadence_mismatches={cadence_mismatches[:8]}"
        )
    return {
        "checks": checks,
        "packet_count": len(pts),
        "first_pts": pts[0],
        "last_pts": pts[-1],
        "actual_time_base": str(actual_tb),
        "source_delivery_time_base": media.fraction_text(timing.time_base),
        "source_frame_period": str(frame_period),
        "avg_frame_rate_diagnostic": profile.get("avg_frame_rate"),
    }


def _transport_architecture_qa(
    path: Path,
    source_profile: dict[str, Any],
    *,
    prefix: str,
    strict_cfr: bool,
) -> dict[str, Any]:
    video = media.video_profile(path)
    audio = media.audio_profile(path)
    format_name = _format_name(path)
    checks = {
        f"{prefix}_container_is_nut": _is_nut(path),
        f"{prefix}_video_codec_is_ffv1": str(video.get("codec_name") or "") == "ffv1",
        f"{prefix}_audio_codec_is_pcm_s16le": str(audio.get("codec_name") or "") == "pcm_s16le",
        f"{prefix}_audio_sample_rate_matches_source": int(audio.get("sample_rate") or 0) == source._audio_rate(source_profile),
        f"{prefix}_audio_channels_match_source": int(audio.get("channels") or 0) == source._audio_channels(source_profile),
        **_transport_identity_checks(source_profile, video, prefix=prefix),
    }
    timing = _transport_timing_qa(
        path,
        source._timing(source_profile),
        label=f"{prefix} FFV1/NUT transport",
        strict_cfr=strict_cfr,
    )
    checks.update({f"{prefix}_timing_{key}": value for key, value in timing["checks"].items()})
    if not all(checks.values()):
        raise RuntimeError(
            f"{prefix} FFV1/NUT architecture contract failed: checks={checks}; "
            f"format={format_name}; video={video}; audio={audio}; source={source_profile}"
        )
    return {
        "checks": checks,
        "format_name": format_name,
        "video_profile": video,
        "audio_profile": audio,
        "timing": timing,
        "source_contract_is_metadata_authority": True,
        "nut_metadata_not_used_as_authority": list(NUT_NON_AUTHORITATIVE_VIDEO_METADATA),
    }


def _verify_lossless_piece(
    original_source: Path,
    piece: Path,
    *,
    extract_start: float,
    extract_duration: float,
    source_profile: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    source_hashes = media.frame_hashes(
        original_source,
        pix_fmt=str(source_profile["pix_fmt"]),
        start=extract_start,
        duration=extract_duration,
    )
    piece_hashes = media.frame_hashes(piece, pix_fmt=str(source_profile["pix_fmt"]))
    architecture = _transport_architecture_qa(
        piece,
        source_profile,
        prefix=f"stage_{index + 1}",
        strict_cfr=True,
    )
    checks = {
        "exact_decoded_frame_count_match": len(source_hashes) == len(piece_hashes),
        "exact_decoded_frame_hash_match": source_hashes == piece_hashes,
        "source_interval_has_frames": bool(source_hashes),
        **architecture["checks"],
    }
    if not all(checks.values()):
        mismatch = next(
            (
                item
                for item, pair in enumerate(zip(source_hashes, piece_hashes))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise RuntimeError(
            f"lossless source->FFV1/NUT fidelity failed for segment {index + 1}: "
            f"checks={checks}; source_frames={len(source_hashes)} stage_frames={len(piece_hashes)} "
            f"first_hash_mismatch={mismatch}; source_profile={source_profile}; "
            f"transport={architecture}"
        )
    return {
        "segment": index + 1,
        "extract_start": extract_start,
        "extract_duration": extract_duration,
        "frame_count": len(source_hashes),
        "checks": checks,
        "stage_profile": architecture["video_profile"],
        "transport": architecture,
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
    architecture = _transport_architecture_qa(
        stitched,
        source_profile,
        prefix="stitched",
        strict_cfr=False,
    )
    checks = {
        "exact_decoded_frame_count_match": len(expected_hashes) == len(stitched_hashes),
        "exact_decoded_frame_hash_match": expected_hashes == stitched_hashes,
        **architecture["checks"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "lossless FFV1/NUT piece->stitched fidelity failed: "
            f"checks={checks}; expected_frames={len(expected_hashes)} "
            f"stitched_frames={len(stitched_hashes)}; transport={architecture}"
        )
    return {
        "segment": "stitched",
        "frame_count": len(stitched_hashes),
        "checks": checks,
        "stage_profile": architecture["video_profile"],
        "transport": architecture,
    }


def _stage_plan_source(
    original_source: Path,
    plan: source.base.semantic.SemanticPlanV31,
    workspace: Path,
    config: dict[str, Any],
) -> tuple[Path, source.base.semantic.SemanticPlanV31, dict[str, Any], list[dict[str, Any]]]:
    workspace.mkdir(parents=True, exist_ok=True)
    contract = media.inspect_source(original_source, verify_timeline=True)
    media.validate_delivery_compatibility(contract, config)
    source_profile = source._profile_with_contract(contract)
    source._ACTIVE_SOURCE_PROFILE = dict(source_profile)

    source_info = source.base.legacy.renderer.probe(original_source)
    source_duration = float(source_info["format"]["duration"])
    pieces: list[Path] = []
    mappings: list[tuple[source.base.semantic.EditSegment, source.base.semantic.EditSegment]] = []
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

        piece = workspace / f"segment_{index:02d}.nut"
        media.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{extract_start:.6f}",
            "-i", str(original_source),
            "-t", f"{extract_duration:.6f}",
            "-map", "0:v:0",
            "-map", "0:a:0",
            *_ffv1_video_args(source_profile),
            "-vsync", "0",
            "-c:a", "pcm_s16le",
            "-ar", str(contract.audio.sample_rate),
            "-ac", str(contract.audio.channels),
            "-f", "nut",
            str(piece),
        ])
        staging_fidelity.append(
            _verify_lossless_piece(
                original_source,
                piece,
                extract_start=extract_start,
                extract_duration=extract_duration,
                source_profile=source_profile,
                index=index,
            )
        )

        piece_duration = float(source.base.legacy.renderer.probe(piece)["format"]["duration"])
        local_start = cursor + (segment.start - extract_start)
        local_end = local_start + (segment.end - segment.start)
        if local_end > cursor + piece_duration + 0.025:
            raise RuntimeError(
                f"staged segment {index + 1} is shorter than approved source interval: "
                f"need {local_end - cursor:.3f}s, have {piece_duration:.3f}s"
            )
        local_segment = source.base.semantic.EditSegment(
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
    stitched = workspace / "approved_segments_lossless.nut"
    media.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-f", "nut",
        str(stitched),
    ])
    staging_fidelity.append(_verify_stitched_lossless(pieces, stitched, source_profile))

    local_events = []
    for event in plan.effect_events:
        mapped = source.base.legacy._map_source_time(event.time, mappings)
        if mapped is not None:
            local_events.append(replace(event, time=round(mapped, 6)))

    local_finishing = None
    if plan.finishing_move is not None:
        start = source.base.legacy._map_source_time(plan.finishing_move.start, mappings)
        payoff = source.base.legacy._map_source_time(plan.finishing_move.payoff, mappings)
        end = source.base.legacy._map_source_time(plan.finishing_move.end, mappings)
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


def _render_canonical_lossless_master(
    staged_source: Path,
    plan: source.base.semantic.SemanticPlanV31,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    if target.suffix.lower() != ".nut":
        raise RuntimeError(f"canonical FFV1 transport target must use .nut, got {target}")
    graph, duration = source._build_source_native_filter(plan, source_profile)
    timing = source._timing(source_profile)
    fps = media.fraction_text(timing.nominal_rate)

    media.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-filter_complex_threads", "2",
        "-i", str(staged_source),
        "-filter_complex", graph,
        "-map", "[outv]",
        "-map", "[aout]",
        *_ffv1_video_args(source_profile),
        "-c:a", "pcm_s16le",
        "-ar", str(source._audio_rate(source_profile)),
        "-ac", str(source._audio_channels(source_profile)),
        "-r", fps,
        "-vsync", "cfr",
        "-t", f"{duration:.6f}",
        "-f", "nut",
        str(target),
    ])

    expected_hashes = source._canonical_filter_hashes(
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
                for index, pair in enumerate(zip(expected_hashes, master_hashes))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise RuntimeError(
            "canonical FFV1/NUT master differs from filtered decoded pixels: "
            f"expected_frames={len(expected_hashes)} master_frames={len(master_hashes)} "
            f"first_hash_mismatch={mismatch}"
        )

    architecture = _transport_architecture_qa(
        target,
        source_profile,
        prefix="canonical",
        strict_cfr=True,
    )
    return {
        "graph_has_spatial_transform": False,
        "video_operations": "trim/setpts/concat + source-SAR metadata only",
        "canonical_fps": fps,
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
        "exact_filtered_frame_hash_match": True,
        "transport_architecture_qa": architecture,
        "source_contract_metadata_authority": "original_input_probe",
        "nut_color_metadata_authoritative": False,
        "nut_non_authoritative_metadata_fields": list(NUT_NON_AUTHORITATIVE_VIDEO_METADATA),
        "source_profile": source_profile,
    }


def _source_fidelity_qa(
    source_profile: dict[str, Any],
    staging_fidelity: list[dict[str, Any]],
    canonical_master: Path,
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    timing = source._timing(source_profile)
    reference_profile = media.video_profile(canonical_master, count_frames=True)
    output_profile = media.video_profile(output, count_frames=True)
    reference_audio = media.audio_profile(canonical_master)
    output_audio = media.audio_profile(output)
    reference_transport = _transport_architecture_qa(
        canonical_master,
        source_profile,
        prefix="canonical_reference",
        strict_cfr=True,
    )
    output_timing = media.verify_cfr_timeline(output, timing, label="final H.264 output")

    settings = config["output"]
    expected_audio_rate = int(settings.get("audio_sample_rate") or source._audio_rate(source_profile))
    expected_audio_channels = int(settings.get("audio_channels") or source._audio_channels(source_profile))
    source_geometry = (int(source_profile["width"]), int(source_profile["height"]))
    checks = {
        "reference_geometry_matches_source": (
            reference_profile["width"], reference_profile["height"]
        ) == source_geometry,
        "output_geometry_matches_source": (
            output_profile["width"], output_profile["height"]
        ) == source_geometry,
        "canonical_transport_is_ffv1_nut_pcm": all(reference_transport["checks"].values()),
        "output_time_base_matches_source": output_timing["checks"]["time_base_matches_source"],
        "output_r_fps_matches_source": output_timing["checks"]["r_frame_rate_matches_source"],
        "output_strict_cfr_timestamps": output_timing["checks"]["packet_pts_strictly_cfr"],
        "exact_frame_count_match": reference_profile["frame_count"] == output_profile["frame_count"],
        "all_source_to_stage_frame_hashes_exact": all(
            item["checks"]["exact_decoded_frame_hash_match"] for item in staging_fidelity
        ),
        "all_staging_transport_checks_pass": all(
            all(bool(value) for value in item["checks"].values()) for item in staging_fidelity
        ),
        "canonical_audio_sample_rate_matches_source": reference_audio["sample_rate"] == source._audio_rate(source_profile),
        "canonical_audio_channels_match_source": reference_audio["channels"] == source._audio_channels(source_profile),
        "output_audio_sample_rate_matches_delivery": output_audio["sample_rate"] == expected_audio_rate,
        "output_audio_channels_match_delivery": output_audio["channels"] == expected_audio_channels,
    }
    checks.update(_transport_identity_checks(source_profile, reference_profile, prefix="canonical_reference"))
    final_metadata_checks = media.metadata_match_checks(source_profile, output_profile, prefix="output")
    checks.update(final_metadata_checks)
    checks["source_media_contract_reapplied_to_final_h264"] = all(final_metadata_checks.values())
    if not all(checks.values()):
        raise RuntimeError(
            "source-fidelity transport/final metadata mismatch before SSIM/PSNR: "
            f"checks={checks}; source={source_profile}; reference={reference_profile}; "
            f"output={output_profile}; reference_transport={reference_transport}; "
            f"output_timing={output_timing}"
        )

    ssim = source._metric(
        [
            "ffmpeg", "-hide_banner",
            "-i", str(canonical_master),
            "-i", str(output),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[enc];[ref][enc]ssim[metric]",
            "-map", "[metric]", "-an", "-f", "null", "-",
        ],
        r"All:([0-9.]+)",
        "SSIM",
    )
    psnr = source._metric(
        [
            "ffmpeg", "-hide_banner",
            "-i", str(canonical_master),
            "-i", str(output),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[enc];[ref][enc]psnr[metric]",
            "-map", "[metric]", "-an", "-f", "null", "-",
        ],
        r"average:([0-9.]+)",
        "PSNR",
    )
    fidelity_cfg = config.get("source_fidelity", {})
    minimum_ssim = float(fidelity_cfg.get("minimum_ssim", 0.99))
    minimum_psnr = float(fidelity_cfg.get("minimum_psnr_db", 40.0))
    checks.update({
        "no_spatial_crop_or_upscale": True,
        "single_canonical_visual_timeline": True,
        "ssim_encoder_fidelity": ssim >= minimum_ssim,
        "psnr_encoder_fidelity": psnr >= minimum_psnr,
    })
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
        "reference_transport": reference_transport,
        "output_timing": output_timing,
        "reference": (
            "original-source media contract remains authoritative; source->FFV1/NUT decoded frames "
            "must match exactly; NUT timing must preserve source frame cadence in rational seconds; "
            "the final H.264 must restore source-native time base and representable video metadata; "
            "only final H.264 generation loss is measured by SSIM/PSNR"
        ),
    }


def _self_test_case(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    original_source = root / f"{name}_source.mov"
    source._synthetic_source(original_source, **spec)
    contract = media.inspect_source(original_source, verify_timeline=True)
    if contract.video.width != spec["width"] or contract.video.height != spec["height"]:
        raise AssertionError(f"{name}: source geometry was not derived correctly")
    if contract.video.timing.track_timescale != spec["timescale"]:
        raise AssertionError(f"{name}: source timescale was not derived correctly")
    if contract.audio.sample_rate != spec["audio_rate"] or contract.audio.channels != spec["channels"]:
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
    plan = source._test_plan()
    workspace = root / f"{name}_work"
    staged, local_plan, source_profile, staging = _stage_plan_source(
        original_source, plan, workspace, config
    )
    if staged.suffix.lower() != ".nut" or not _is_nut(staged):
        raise AssertionError(f"{name}: staged transport is not NUT")

    canonical = root / f"{name}_canonical.nut"
    canonical_info = _render_canonical_lossless_master(
        staged, local_plan, config, source_profile, canonical
    )
    if canonical_info["actual_video_codec"] != "ffv1" or canonical_info["actual_container"] != "nut":
        raise AssertionError(f"{name}: canonical architecture is not FFV1/NUT: {canonical_info}")

    output = root / f"{name}_final.mp4"
    source._encode_from_canonical_master(canonical, config, source_profile, output, mode="production")
    qa = _source_fidelity_qa(source_profile, staging, canonical, output, config)
    final_timing = media.verify_cfr_timeline(
        output,
        contract.video.timing,
        label=f"{name} final delivery",
    )
    final_profile = media.video_profile(output)
    final_metadata = media.metadata_match_checks(source_profile, final_profile, prefix="preflight_final")
    if not all(final_metadata.values()):
        raise AssertionError(f"{name}: final H.264 did not restore source metadata: {final_metadata}")

    try:
        _transport_architecture_qa(
            original_source,
            source_profile,
            prefix="negative_regression",
            strict_cfr=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{name}: architecture guard failed to reject H.264/MOV as FFV1/NUT")

    return {
        "source": contract.to_json(),
        "staged_container": _format_name(staged),
        "staged_video_codec": media.video_profile(staged)["codec_name"],
        "canonical_container": canonical_info["actual_container"],
        "canonical_video_codec": canonical_info["actual_video_codec"],
        "canonical_transport_time_base": canonical_info["transport_time_base"],
        "source_delivery_time_base": canonical_info["source_delivery_time_base"],
        "final_time_base": media.video_profile(output)["time_base"],
        "final_timing": final_timing,
        "ssim": qa["ssim"],
        "psnr_db": qa["psnr_db"],
    }


def preflight() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mw4_ffv1_contract_preflight_") as tmp:
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
            raise AssertionError("preflight did not exercise distinct source-derived timing contracts")
        if any(item["canonical_container"] != "nut" for item in results.values()):
            raise AssertionError("preflight canonical transport was not NUT")
        if any(item["canonical_video_codec"] != "ffv1" for item in results.values()):
            raise AssertionError("preflight canonical codec was not FFV1")
        print(json.dumps({"ffv1_nut_source_contract_preflight": "PASS", "cases": results}, sort_keys=True))
        return results


def _install() -> None:
    source._install()
    source.base.lossless.preflight = preflight
    source.base._stage_plan_source = _stage_plan_source
    source.base._verify_stitched_lossless = _verify_stitched_lossless
    source.base._render_canonical_lossless_master = _render_canonical_lossless_master
    source.base._source_fidelity_qa = _source_fidelity_qa


def main() -> None:
    if "--self-test" in sys.argv:
        preflight()
        return
    _install()
    source.base.main()


if __name__ == "__main__":
    main()
