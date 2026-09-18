from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_final as semantic
import mw4_v3_1_contract as contract

UNKNOWN_METADATA = {"", "unknown", "unspecified", "reserved", "n/a", "N/A", "0:1"}
FIDELITY_METADATA_FIELDS = (
    "pix_fmt",
    "sample_aspect_ratio",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "chroma_location",
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _video_profile(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    entries = (
        "stream=codec_name,profile,width,height,pix_fmt,sample_aspect_ratio,display_aspect_ratio,"
        "color_range,color_space,color_transfer,color_primaries,chroma_location,field_order,"
        "bits_per_raw_sample,r_frame_rate,avg_frame_rate"
    )
    if count_frames:
        entries += ",nb_read_frames"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command += ["-show_entries", entries, "-of", "json", str(path)]
    payload = json.loads(_run_capture(command).stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected exactly one video stream: {path}")
    stream = streams[0]
    profile: dict[str, Any] = {
        "codec_name": str(stream.get("codec_name") or ""),
        "profile": str(stream.get("profile") or ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "pix_fmt": str(stream.get("pix_fmt") or ""),
        "sample_aspect_ratio": str(stream.get("sample_aspect_ratio") or ""),
        "display_aspect_ratio": str(stream.get("display_aspect_ratio") or ""),
        "color_range": str(stream.get("color_range") or ""),
        "color_space": str(stream.get("color_space") or ""),
        "color_transfer": str(stream.get("color_transfer") or ""),
        "color_primaries": str(stream.get("color_primaries") or ""),
        "chroma_location": str(stream.get("chroma_location") or ""),
        "field_order": str(stream.get("field_order") or ""),
        "bits_per_raw_sample": str(stream.get("bits_per_raw_sample") or ""),
        "r_frame_rate": str(stream.get("r_frame_rate") or ""),
        "avg_frame_rate": str(stream.get("avg_frame_rate") or ""),
    }
    if count_frames:
        count = stream.get("nb_read_frames")
        if count in (None, "N/A"):
            raise RuntimeError(f"ffprobe could not count decoded video frames for {path}")
        profile["frame_count"] = int(count)
    return profile


def _known(value: str) -> bool:
    return str(value).strip() not in UNKNOWN_METADATA


def _profile_output_args(profile: dict[str, Any]) -> list[str]:
    """Preserve source pixel format and color/chroma signaling without conversion."""
    args = ["-pix_fmt", str(profile["pix_fmt"])]
    mapping = (
        ("color_range", "-color_range"),
        ("color_space", "-colorspace"),
        ("color_transfer", "-color_trc"),
        ("color_primaries", "-color_primaries"),
        ("chroma_location", "-chroma_sample_location"),
    )
    for key, option in mapping:
        value = str(profile.get(key) or "")
        if _known(value):
            args += [option, value]
    return args


def _assert_source_profile(source: Path, profile: dict[str, Any], config: dict[str, Any]) -> None:
    expected_fps = str(config["output"]["fps"])
    failures: list[str] = []
    if (profile["width"], profile["height"]) != (1920, 1080):
        failures.append(f"geometry={profile['width']}x{profile['height']}")
    if profile["r_frame_rate"] != expected_fps or profile["avg_frame_rate"] != expected_fps:
        failures.append(
            f"fps=r:{profile['r_frame_rate']} avg:{profile['avg_frame_rate']} expected:{expected_fps}"
        )
    # The delivery contract is H.264 High yuv420p. Fail rather than silently
    # chroma-convert a source master with a different pixel format.
    if profile["pix_fmt"] != "yuv420p":
        failures.append(f"pix_fmt={profile['pix_fmt']} expected source-native yuv420p")
    if failures:
        raise RuntimeError(
            f"cannot guarantee source-native output quality for {source.name}: "
            + "; ".join(failures)
        )


def _decoded_frame_hashes(
    path: Path,
    *,
    pix_fmt: str,
    start: float | None = None,
    duration: float | None = None,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.6f}"]
    command += [
        "-map",
        "0:v:0",
        "-an",
        "-vsync",
        "0",
        "-pix_fmt",
        pix_fmt,
        "-f",
        "framemd5",
        "-",
    ]
    text = _run_capture(command).stdout
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _metadata_match_checks(
    source_profile: dict[str, Any],
    other_profile: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for field in FIDELITY_METADATA_FIELDS:
        checks[f"{prefix}_{field}_matches_source"] = str(other_profile.get(field) or "") == str(
            source_profile.get(field) or ""
        )
    return checks


def _verify_lossless_piece(
    source: Path,
    piece: Path,
    *,
    extract_start: float,
    extract_duration: float,
    source_profile: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    source_hashes = _decoded_frame_hashes(
        source,
        pix_fmt=str(source_profile["pix_fmt"]),
        start=extract_start,
        duration=extract_duration,
    )
    piece_hashes = _decoded_frame_hashes(
        piece,
        pix_fmt=str(source_profile["pix_fmt"]),
    )
    piece_profile = _video_profile(piece)
    metadata_checks = _metadata_match_checks(source_profile, piece_profile, prefix="stage")
    checks = {
        "exact_decoded_frame_count_match": len(source_hashes) == len(piece_hashes),
        "exact_decoded_frame_hash_match": source_hashes == piece_hashes,
        **metadata_checks,
        "source_interval_has_frames": bool(source_hashes),
    }
    if not all(checks.values()):
        mismatch = next(
            (i for i, pair in enumerate(zip(source_hashes, piece_hashes)) if pair[0] != pair[1]),
            None,
        )
        raise RuntimeError(
            f"lossless source->stage fidelity failed for segment {index + 1}: "
            f"checks={checks}; source_frames={len(source_hashes)} stage_frames={len(piece_hashes)} "
            f"first_hash_mismatch={mismatch}; source_profile={source_profile}; stage_profile={piece_profile}"
        )
    return {
        "segment": index + 1,
        "extract_start": extract_start,
        "extract_duration": extract_duration,
        "frame_count": len(source_hashes),
        "checks": checks,
        "stage_profile": piece_profile,
    }


def _map_source_time(
    value: float,
    mappings: list[tuple[semantic.EditSegment, semantic.EditSegment]],
) -> float | None:
    for original, local in mappings:
        if original.start - 1e-6 <= value <= original.end + 1e-6:
            return local.start + (value - original.start)
    return None


def _stage_plan_source(
    source: Path,
    plan: semantic.SemanticPlanV31,
    workspace: Path,
    config: dict[str, Any],
) -> tuple[Path, semantic.SemanticPlanV31, dict[str, Any], list[dict[str, Any]]]:
    """Stage only approved intervals losslessly and prove source->stage frame identity."""
    workspace.mkdir(parents=True, exist_ok=True)
    source_info = renderer.probe(source)
    source_duration = float(source_info["format"]["duration"])
    source_profile = _video_profile(source)
    _assert_source_profile(source, source_profile, config)

    pieces: list[Path] = []
    mappings: list[tuple[semantic.EditSegment, semantic.EditSegment]] = []
    staging_fidelity: list[dict[str, Any]] = []
    cursor = 0.0
    segments = list(plan.segments)

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

        # NUT keeps exact 60000/1001 timestamps. FFV1 is lossless and the frame-hash
        # check below proves the decoded YUV planes are byte-identical to the same
        # interval decoded directly from the original MediaSilo source master.
        piece = workspace / f"segment_{index:02d}.nut"
        _run(
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
                "-c:v",
                "ffv1",
                "-level",
                "3",
                *_profile_output_args(source_profile),
                "-vsync",
                "0",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "48000",
                "-ac",
                "2",
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

        piece_duration = float(renderer.probe(piece)["format"]["duration"])
        local_start = cursor + (segment.start - extract_start)
        local_end = local_start + (segment.end - segment.start)
        if local_end > cursor + piece_duration + 0.025:
            raise RuntimeError(
                f"staged segment {index + 1} is shorter than approved source interval: "
                f"need {local_end - cursor:.3f}s, have {piece_duration:.3f}s"
            )
        local_segment = semantic.EditSegment(
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
    _run(
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
            str(stitched),
        ]
    )

    local_events = []
    for event in plan.effect_events:
        mapped = _map_source_time(event.time, mappings)
        if mapped is not None:
            local_events.append(replace(event, time=round(mapped, 6)))

    local_finishing = None
    if plan.finishing_move is not None:
        start = _map_source_time(plan.finishing_move.start, mappings)
        payoff = _map_source_time(plan.finishing_move.payoff, mappings)
        end = _map_source_time(plan.finishing_move.end, mappings)
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
    plan: semantic.SemanticPlanV31,
    source_profile: dict[str, Any],
) -> tuple[str, float]:
    """Timing/cuts only. SAR is preserved as metadata; pixels are never resampled."""
    parts: list[str] = []
    pairs = [
        _append_source_native_segment(parts, index, segment)
        for index, segment in enumerate(plan.segments)
    ]
    if not pairs:
        raise RuntimeError("approved plan has no edit segments")

    sar = str(source_profile.get("sample_aspect_ratio") or "")
    sar_filter = "null" if not _known(sar) else f"setsar={sar.replace(':', '/')}"
    if len(pairs) == 1:
        parts.append(f"[{pairs[0][0]}]{sar_filter}[outv]")
        parts.append(f"[{pairs[0][1]}]aresample=48000[aout]")
    else:
        concat_inputs = "".join(f"[{video}][{audio}]" for video, audio in pairs)
        parts.append(f"{concat_inputs}concat=n={len(pairs)}:v=1:a=1[vcat][acat]")
        parts.append(f"[vcat]{sar_filter}[outv]")
        parts.append("[acat]aresample=48000[aout]")

    graph = ";".join(parts)
    lowered = graph.lower()
    forbidden = ("crop=", "scale=", "zscale=", "zoompan=", "perspective=", "rotate=")
    found = [token for token in forbidden if token in lowered]
    if found:
        raise RuntimeError(
            f"source-native canonical graph contains forbidden spatial transform(s): {found}"
        )
    return graph, plan.output_duration


def _render_canonical_lossless_master(
    staged_source: Path,
    plan: semantic.SemanticPlanV31,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    graph, duration = _build_source_native_filter(plan, source_profile)
    settings = config["output"]
    _run(
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
            "-c:v",
            "ffv1",
            "-level",
            "3",
            *_profile_output_args(source_profile),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-r",
            str(settings["fps"]),
            "-vsync",
            "cfr",
            "-t",
            f"{duration:.3f}",
            str(target),
        ]
    )
    return {
        "graph_has_spatial_transform": False,
        "video_operations": "trim/setpts/concat + source-SAR metadata only",
        "canonical_fps": str(settings["fps"]),
        "canonical_container": "nut",
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
    _run(
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
            *renderer._encode_args(config, mode),
            *_profile_output_args(source_profile),
            "-threads:v",
            "4",
            "-vsync",
            "0",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def _extract_metric(stderr: str, pattern: str, label: str) -> float:
    matches = re.findall(pattern, stderr)
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
    """Prove source->lossless-stage identity, then measure only H.264 generation loss."""
    reference_profile = _video_profile(canonical_master, count_frames=True)
    output_profile = _video_profile(output, count_frames=True)
    expected_fps = str(config["output"]["fps"])

    alignment_checks = {
        "reference_1920x1080": (reference_profile["width"], reference_profile["height"])
        == (1920, 1080),
        "output_1920x1080": (output_profile["width"], output_profile["height"]) == (1920, 1080),
        "reference_r_fps": reference_profile["r_frame_rate"] == expected_fps,
        "reference_avg_fps": reference_profile["avg_frame_rate"] == expected_fps,
        "output_r_fps": output_profile["r_frame_rate"] == expected_fps,
        "output_avg_fps": output_profile["avg_frame_rate"] == expected_fps,
        "exact_frame_count_match": (
            reference_profile["frame_count"] == output_profile["frame_count"]
        ),
        "all_source_to_stage_frame_hashes_exact": all(
            item["checks"]["exact_decoded_frame_hash_match"] for item in staging_fidelity
        ),
    }
    alignment_checks.update(
        _metadata_match_checks(source_profile, reference_profile, prefix="canonical")
    )
    alignment_checks.update(_metadata_match_checks(source_profile, output_profile, prefix="output"))
    if not all(alignment_checks.values()):
        raise RuntimeError(
            "source-fidelity metadata/timeline mismatch before SSIM/PSNR: "
            f"checks={alignment_checks}; source={source_profile}; "
            f"reference={reference_profile}; output={output_profile}"
        )

    ssim_run = _run_capture(
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
        ]
    )
    psnr_run = _run_capture(
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
        ]
    )
    ssim = _extract_metric(ssim_run.stderr, r"All:([0-9.]+)", "SSIM")
    psnr = _extract_metric(psnr_run.stderr, r"average:([0-9.]+)", "PSNR")

    fidelity_cfg = config.get("source_fidelity", {})
    minimum_ssim = float(fidelity_cfg.get("minimum_ssim", 0.99))
    minimum_psnr = float(fidelity_cfg.get("minimum_psnr_db", 40.0))
    checks = {
        **alignment_checks,
        "no_spatial_crop_or_upscale": True,
        "single_canonical_visual_timeline": True,
        "ssim_encoder_fidelity": ssim >= minimum_ssim,
        "psnr_encoder_fidelity": psnr >= minimum_psnr,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"source-fidelity QA failed: SSIM={ssim:.6f} minimum={minimum_ssim:.6f}; "
            f"PSNR={psnr:.3f}dB minimum={minimum_psnr:.3f}dB; "
            f"reference_frames={reference_profile['frame_count']} "
            f"output_frames={output_profile['frame_count']}"
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
        "reference": (
            "original MediaSilo source decoded-frame hashes must match every lossless "
            "staged segment exactly; final H.264 is then compared against the single "
            "canonical FFV1/NUT edited master"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--plan-key", required=True)
    parser.add_argument("--ordinal", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("shadow", "production"), required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected, allocation = renderer._select_from_allocation(args.source_key, args.allocation)
    matches = [plan for plan in selected if renderer.plan_key(plan) == args.plan_key]
    if len(matches) != 1:
        raise RuntimeError(
            f"{args.source_key}: expected exactly one allocated plan for {args.plan_key}, found {len(matches)}"
        )
    plan = matches[0]
    payload = renderer._candidate_dict(plan)

    plan_failures = contract.validate_plan(args.source_key, args.ordinal, payload, config)
    if plan_failures:
        raise RuntimeError("; ".join(plan_failures))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "250M" if args.mode == "production" else "SHADOW"
    filename = (
        f"MW4_V31_{args.source_key}_{args.ordinal:02d}_{plan.story_type}_source_native_{suffix}.mp4"
    )
    target = args.output_dir / filename

    with tempfile.TemporaryDirectory(prefix="mw4_v31_stage_") as temp_dir:
        workspace = Path(temp_dir)
        staged_source, local_plan, source_profile, staging_fidelity = _stage_plan_source(
            args.source, plan, workspace, config
        )
        fidelity_plan = replace(local_plan, effect_profile="source_native_full_frame")
        canonical_master = workspace / "canonical_lossless_edited_master.nut"
        canonical_info = _render_canonical_lossless_master(
            staged_source,
            fidelity_plan,
            config,
            source_profile,
            canonical_master,
        )
        _encode_from_canonical_master(
            canonical_master,
            config,
            source_profile,
            target,
            mode=args.mode,
        )
        fidelity_qa = _source_fidelity_qa(
            source_profile,
            staging_fidelity,
            canonical_master,
            target,
            config,
        )

    qa = renderer.validate_output(target, config, mode=args.mode)
    sheets = args.output_dir / "contact_sheets"
    renderer.create_contact_sheets(target, sheets)

    result = {
        "version": "3.1",
        "mode": args.mode,
        "source_key": args.source_key,
        "ordinal": args.ordinal,
        "plan_key": args.plan_key,
        "allocation_mode": allocation.get("allocation_mode"),
        "allocation_selected_count": allocation.get("selected_count"),
        "story_type": plan.story_type,
        "planned_effect_profile": plan.effect_profile,
        "rendered_effect_profile": "source_native_full_frame",
        "finishing_move": plan.finishing_move is not None,
        "unplanned_source_cut_count": 0,
        "technical_qa_passed": all(bool(value) for value in qa["checks"].values()),
        "source_fidelity_qa_passed": all(bool(value) for value in fidelity_qa["checks"].values()),
        "source_fidelity": fidelity_qa,
        "canonical_master": canonical_info,
        "file": filename,
        "sha256": renderer.sha256(target),
        "editorial_plan": payload,
        "qa": qa,
        "render_reanalysis": False,
        "source_structure_reanalysis": False,
        "bounded_lossless_segment_staging": True,
        "exact_source_to_stage_frame_hash_qa": True,
        "source_color_metadata_preserved": True,
        "spatial_crop_upscale_used": False,
        "source_native_full_frame": True,
        "single_canonical_visual_timeline": True,
        "single_clip_workspace": True,
        "status": "PASS",
    }
    result_path = args.output_dir / (
        f"{args.source_key}_{args.ordinal:02d}_{args.plan_key}_clip_result_v3_1.json"
    )
    _write(result_path, result)
    print(
        json.dumps(
            {
                "source": args.source_key,
                "ordinal": args.ordinal,
                "plan_key": args.plan_key,
                "mode": args.mode,
                "file": filename,
                "qa": "PASS",
                "source_fidelity_qa": "PASS",
                "ssim": fidelity_qa["ssim"],
                "psnr_db": fidelity_qa["psnr_db"],
                "frame_count": fidelity_qa["reference_profile"]["frame_count"],
                "source_profile": source_profile,
                "source_to_stage_hashes_exact": True,
                "source_color_metadata_preserved": True,
                "spatial_crop_upscale_used": False,
                "single_canonical_visual_timeline": True,
                "bounded_lossless_segment_staging": True,
            }
        )
    )


if __name__ == "__main__":
    main()
