from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mw4_semantic_gameplay_v3_1_final as semantic


ENGINE = "deterministic-gameplay-v3.1-final"
EDITOR = "semantic-editor-v3.1-final"
CANDIDATE_MODE = "engagement_driven_hardened_source_integrity"


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def probe(path: Path) -> dict[str, Any]:
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,bit_rate,sample_rate,channels:format=duration,size,bit_rate",
        "-of", "json", str(path),
    ], capture=True)
    return json.loads(result.stdout)


def _append_segment(
    parts: list[str],
    index: int,
    start: float,
    end: float,
    speed: float,
) -> tuple[str, str]:
    v = f"sv{index}"
    a = f"sa{index}"
    vpts = "PTS-STARTPTS" if abs(speed - 1.0) < 1e-6 else f"(PTS-STARTPTS)/{speed:.6f}"
    parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts={vpts}[{v}]")
    achain = f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS"
    if abs(speed - 1.0) >= 1e-6:
        achain += f",atempo={speed:.6f}"
    parts.append(achain + f"[{a}]")
    return v, a


def source_time_to_output(
    source_time: float,
    segments: tuple[semantic.EditSegment, ...],
) -> float | None:
    out = 0.0
    for segment in segments:
        if segment.start <= source_time <= segment.end:
            return out + (source_time - segment.start) / segment.speed
        out += (segment.end - segment.start) / segment.speed
    return None


def gate(times: list[float], before: float, after: float) -> str:
    return "+".join(
        f"between(t,{max(0.0, value-before):.3f},{value+after:.3f})"
        for value in times
    ) or "0"


def _planned_transition_times(plan: semantic.SemanticPlanV31) -> list[float]:
    joins: list[float] = []
    output_cursor = 0.0
    for index, segment in enumerate(plan.segments[:-1]):
        output_cursor += (segment.end - segment.start) / segment.speed
        next_segment = plan.segments[index + 1]
        if next_segment.start <= segment.end + 0.02:
            continue

        intentional = False
        if (
            plan.story_type == "semantic_montage"
            and segment.reason == "semantic_montage_moment"
            and next_segment.reason == "semantic_montage_moment"
        ):
            intentional = True
        elif plan.story_type == "finishing_move_open" and index == 0:
            intentional = segment.reason == "finishing_move_open_hero"
        if intentional:
            joins.append(output_cursor)
    return joins


def _plan_key_from_dict(plan: dict[str, Any]) -> str:
    finishing = plan.get("finishing_move")
    payload = {
        "story_type": str(plan.get("story_type", "")),
        "effect_profile": str(plan.get("effect_profile", "")),
        "segments": [
            [
                round(float(segment["start"]), 3),
                round(float(segment["end"]), 3),
                round(float(segment.get("speed", 1.0)), 3),
                str(segment.get("reason", "")),
            ]
            for segment in (plan.get("segments") or [])
        ],
        "finishing_move": None
        if finishing is None
        else [
            round(float(finishing["start"]), 3),
            round(float(finishing["payoff"]), 3),
            round(float(finishing["end"]), 3),
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def plan_key(plan: semantic.SemanticPlanV31) -> str:
    return _plan_key_from_dict(asdict(plan))


def build_filter(
    plan: semantic.SemanticPlanV31,
    config: dict[str, Any],
) -> tuple[str, float]:
    parts: list[str] = []
    pairs: list[tuple[str, str]] = []
    for index, segment in enumerate(plan.segments):
        pairs.append(_append_segment(parts, index, segment.start, segment.end, segment.speed))

    if len(pairs) == 1:
        parts.append(f"[{pairs[0][0]}]null[vsrc]")
        parts.append(f"[{pairs[0][1]}]anull[asrc]")
    else:
        concat_inputs = "".join(f"[{v}][{a}]" for v, a in pairs)
        parts.append(f"{concat_inputs}concat=n={len(pairs)}:v=1:a=1[vsrc][asrc]")

    times = [
        mapped
        for event in plan.effect_events
        if (mapped := source_time_to_output(event.time, plan.segments)) is not None
    ]
    profile = plan.effect_profile
    transition_cfg = config.get("semantic_editor", {}).get("semantic_montage", {})
    transition_times = _planned_transition_times(plan)
    transition_opacity = float(transition_cfg.get("transition_flash_opacity", 0.08))
    transition_before = float(transition_cfg.get("transition_flash_before_seconds", 0.025))
    transition_after = float(transition_cfg.get("transition_flash_after_seconds", 0.055))

    if profile == "finishing_move_hero" and plan.finishing_move is not None:
        payoff = source_time_to_output(plan.finishing_move.payoff, plan.segments)
        impact_percent = float(config.get("editorial", {}).get("finishing_move_impact_bump_percent", 0.012))
        impact_before = float(config.get("editorial", {}).get("finishing_move_impact_bump_before_seconds", 0.035))
        impact_after = float(config.get("editorial", {}).get("finishing_move_impact_bump_after_seconds", 0.085))
        if payoff is None or impact_percent <= 0.0:
            parts.append("[vsrc]scale=1920:1080:flags=lanczos,setsar=1[outv]")
        else:
            impact_w = max(1920, int(round(1920 * (1.0 + impact_percent) / 2.0) * 2))
            impact_h = max(1080, int(round(1080 * (1.0 + impact_percent) / 2.0) * 2))
            x = max(0, (impact_w - 1920) // 2)
            y = max(0, (impact_h - 1080) // 2)
            parts.extend([
                "[vsrc]split=2[vbase][vimpact]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                f"[vimpact]scale={impact_w}:{impact_h}:flags=lanczos,crop=1920:1080:{x}:{y},setsar=1[impact]",
                f"[base][impact]overlay=0:0:enable='{gate([payoff], impact_before, impact_after)}'[outv]",
            ])
    elif profile == "semantic_montage":
        punch_times = times[:3]
        if punch_times:
            parts.extend([
                "[vsrc]split=2[vbase][vpunch]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vpunch]crop=1882:1059:(iw-1882)/2:(ih-1059)/2,scale=1920:1080:flags=lanczos,setsar=1[punch]",
                f"[base][punch]overlay=0:0:enable='{gate(punch_times, .055, .155)}'[fx1]",
            ])
        else:
            parts.append("[vsrc]scale=1920:1080:flags=lanczos,setsar=1[fx1]")
        if transition_times:
            parts.append(
                f"[fx1]drawbox=x=0:y=0:w=iw:h=ih:color=white@{transition_opacity:.3f}:t=fill:"
                f"enable='{gate(transition_times, transition_before, transition_after)}'[outv]"
            )
        else:
            parts.append("[fx1]null[outv]")
    elif profile == "precision_punch" and times:
        first = times[0]
        second = times[1] if len(times) > 1 else first
        parts.extend([
            "[vsrc]split=3[vbase][vz1][vz2]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vz1]crop=1860:1046:(iw-1860)/2:(ih-1046)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
            "[vz2]crop=1828:1028:(iw-1828)/2:(ih-1028)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
            f"[base][z1]overlay=0:0:enable='{gate([first], .08, .22)}'[fx1]",
            f"[fx1][z2]overlay=0:0:enable='{gate([second], .10, .28)}'[outv]",
        ])
    elif profile == "impact_flash" and times:
        impact_times = times[:2]
        parts.extend([
            "[vsrc]split=2[vbase][vshake]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vshake]scale=1944:1094:flags=lanczos,crop=1920:1080:x='12+6*sin(100*t)':y='7+3*sin(83*t)',setsar=1[shake]",
            f"[base][shake]overlay=0:0:enable='{gate(impact_times, .05, .14)}'[fx1]",
            f"[fx1]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.08:t=fill:enable='{gate(impact_times, .01, .04)}'[outv]",
        ])
    elif profile == "chain_escalation" and len(times) >= 3:
        first, second, third = times[:3]
        parts.extend([
            "[vsrc]split=4[vbase][vz1][vz2][vz3]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vz1]crop=1870:1052:(iw-1870)/2:(ih-1052)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
            "[vz2]crop=1838:1034:(iw-1838)/2:(ih-1034)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
            "[vz3]crop=1806:1016:(iw-1806)/2:(ih-1016)/2,scale=1920:1080:flags=lanczos,setsar=1[z3]",
            f"[base][z1]overlay=0:0:enable='{gate([first], .07, .18)}'[fx1]",
            f"[fx1][z2]overlay=0:0:enable='{gate([second], .08, .22)}'[fx2]",
            f"[fx2][z3]overlay=0:0:enable='{gate([third], .10, .28)}'[outv]",
        ])
    elif profile == "clean_pressure" and times:
        strongest = times[0]
        parts.extend([
            "[vsrc]split=2[vbase][vz]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vz]crop=1890:1063:(iw-1890)/2:(ih-1063)/2,scale=1920:1080:flags=lanczos,setsar=1[z]",
            f"[base][z]overlay=0:0:enable='{gate([strongest], .06, .17)}'[outv]",
        ])
    else:
        parts.append("[vsrc]scale=1920:1080:flags=lanczos,setsar=1[outv]")

    parts.append("[asrc]aresample=48000[aout]")
    return ";".join(parts), plan.output_duration


def _encode_args(config: dict[str, Any], mode: str) -> list[str]:
    settings = config["output"]
    if mode == "production":
        bitrate = int(settings["video_bitrate_kbps"])
        return [
            "-c:v", str(settings["codec"]),
            "-preset", str(settings["preset"]),
            "-profile:v", str(settings["profile"]),
            "-level:v", "5.2",
            "-pix_fmt", "yuv420p",
            "-b:v", f"{bitrate}k",
            "-minrate", f"{bitrate}k",
            "-maxrate", f"{bitrate}k",
            "-bufsize", f"{bitrate * 2}k",
            "-x264-params", "nal-hrd=cbr",
            "-c:a", str(settings["audio_codec"]),
            "-b:a", f"{int(settings['audio_bitrate_kbps'])}k",
            "-ar", str(settings["audio_sample_rate"]),
            "-ac", str(settings["audio_channels"]),
        ]
    return [
        "-c:v", "libx264",
        "-preset", "superfast",
        "-profile:v", "high",
        "-level:v", "5.2",
        "-pix_fmt", "yuv420p",
        "-b:v", "8M",
        "-maxrate", "10M",
        "-bufsize", "20M",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
    ]


def render_candidate(
    source: Path,
    plan: semantic.SemanticPlanV31,
    config: dict[str, Any],
    output: Path,
    *,
    mode: str = "production",
) -> None:
    graph, duration = build_filter(plan, config)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-filter_complex", graph,
        "-map", "[outv]",
        "-map", "[aout]",
        *_encode_args(config, mode),
        "-movflags", "+faststart",
        "-t", f"{duration:.3f}",
        str(output),
    ]
    run(command)


def validate_output(
    path: Path,
    config: dict[str, Any],
    *,
    mode: str = "production",
) -> dict[str, Any]:
    data = probe(path)
    video = next(item for item in data["streams"] if item["codec_type"] == "video")
    audio = next(item for item in data["streams"] if item["codec_type"] == "audio")
    fmt = data["format"]
    settings = config["output"]
    duration = float(fmt["duration"])
    bitrate = int(fmt.get("bit_rate") or video.get("bit_rate") or 0)
    checks = {
        "duration_10_to_20s": 10.0 <= duration <= 20.0,
        "full_frame_1920x1080": (video["width"], video["height"]) == (1920, 1080),
        "codec_h264": video["codec_name"] == "h264",
        "profile_high": str(video.get("profile", "")).lower() == "high",
        "r_frame_rate": video["r_frame_rate"] == settings["fps"],
        "avg_frame_rate": video["avg_frame_rate"] == settings["fps"],
        "audio_48khz": audio["sample_rate"] == "48000",
        "audio_stereo": audio["channels"] == 2,
    }
    if mode == "production":
        expected = int(settings["video_bitrate_kbps"]) * 1000
        checks["high_bitrate_near_250mbps"] = bitrate >= int(expected * 0.94)
    run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if not all(checks.values()):
        raise RuntimeError(f"QA failed for {path.name}: {checks}")
    return {"checks": checks, "probe": data}


def create_contact_sheets(video: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", "fps=4,scale=320:-1,tile=6x5:padding=2:margin=2",
        "-frames:v", "3",
        str(output_dir / f"{video.stem}_%02d.jpg"),
    ])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_contract_failure(
    selected: list[semantic.SemanticPlanV31],
    timeline: semantic.SemanticTimelineV31,
    config: dict[str, Any],
    source_key: str,
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    for index, plan in enumerate(selected, 1):
        for item in semantic.plan_integrity_violations(plan, timeline, config, source_key):
            violations.append(f"clip {index}: {item}")
    selected_finishers = [
        plan for plan in selected
        if plan.story_type == "finishing_move_open" and plan.finishing_move is not None
    ]
    failures: list[str] = []
    if timeline.finishing_moves and not selected_finishers:
        failures.append("verified Finishing Move exists but no finishing_move_open clip was selected")
    if violations:
        failures.append("selected plans violate the source-integrity contract")
    return failures, violations


def _automatic_finishing_candidates(
    timeline: semantic.SemanticTimelineV31,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = config.get("finishing_move_detector", {})
    if not bool(cfg.get("automatic_discovery_enabled", True)):
        return []
    review_threshold = float(cfg.get("automatic_review_confidence", 0.72))
    heuristic = semantic.refined.core.detect_finishing_moves(
        timeline.base,
        timeline.shots,
        timeline.engagements,
        config,
    )
    return [
        asdict(span)
        for span in heuristic
        if float(span.confidence) >= review_threshold
    ]


def _candidate_dict(plan: semantic.SemanticPlanV31) -> dict[str, Any]:
    payload = asdict(plan)
    payload["plan_key"] = _plan_key_from_dict(payload)
    return payload


def _select_from_allocation(
    plans: list[semantic.SemanticPlanV31],
    source_key: str,
    allocation_path: Path,
) -> tuple[list[semantic.SemanticPlanV31], dict[str, Any]]:
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    source_allocation = allocation.get("source_allocations", {}).get(source_key)
    if not isinstance(source_allocation, dict):
        raise RuntimeError(f"global allocation has no entry for {source_key}")
    wanted = [str(item) for item in source_allocation.get("plan_keys", [])]
    available = {plan_key(plan): plan for plan in plans}
    missing = [key for key in wanted if key not in available]
    if missing:
        raise RuntimeError(
            f"{source_key}: allocation references candidate keys not reproduced by canonical planner: {missing}"
        )
    selected = [available[key] for key in wanted]
    return selected, allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("shadow", "production", "analysis"), default="production")
    parser.add_argument("--selection-file", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    excluded = config.get("excluded_windows", {}).get(args.source_key, [])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timeline = semantic.analyze_source(args.source, config)
    diagnostics = semantic.diagnose_source(timeline, config, excluded, args.source_key)
    plans = semantic.build_plans_for_source(timeline, config, excluded, args.source_key)
    automatic_candidates = _automatic_finishing_candidates(timeline, config)

    if args.mode == "analysis":
        failure: list[str] = []
        minimum = int(config.get("minimum_count_per_source", 2))
        if len(plans) < minimum:
            failure.append(
                f"only {len(plans)} semantic candidates passed; minimum is {minimum}; quality gates were not lowered"
            )
        analysis_path = args.output_dir / f"{args.source_key}_analysis_v3_1.json"
        analysis = {
            "version": "3.1",
            "mode": "analysis",
            "source_key": args.source_key,
            "semantic_engine": ENGINE,
            "editorial_planner": EDITOR,
            "candidate_mode": CANDIDATE_MODE,
            "diagnostics": diagnostics,
            "candidate_count_after_semantic_gates": len(plans),
            "candidate_pool": [_candidate_dict(plan) for plan in plans],
            "verified_finishing_move_count": len(timeline.finishing_moves),
            "verified_finishing_moves": [asdict(span) for span in timeline.finishing_moves],
            "automatic_finishing_move_candidates": automatic_candidates,
            "automatic_finishing_move_candidates_are_discovery_only": True,
            "failure": failure or None,
        }
        _write_json(analysis_path, analysis)
        if failure:
            raise RuntimeError("; ".join(failure))
        print(json.dumps({
            "source": args.source_key,
            "mode": "analysis",
            "candidates": len(plans),
            "verified_finishing_moves": len(timeline.finishing_moves),
            "automatic_finishing_move_candidates": len(automatic_candidates),
        }))
        return

    if args.selection_file is None:
        raise RuntimeError(
            "shadow/production rendering requires --selection-file from the global pre-render allocator"
        )

    selected, allocation = _select_from_allocation(
        plans,
        args.source_key,
        args.selection_file,
    )
    manifest_path = args.output_dir / f"{args.source_key}_manifest_v3_1.json"
    summary_path = args.output_dir / f"{args.source_key}_pipeline_summary.json"

    failure: list[str] = []
    minimum = int(config.get("minimum_count_per_source", 2))
    maximum = int(config.get("count_per_source_max", 4))
    if not (minimum <= len(selected) <= maximum):
        failure.append(
            f"global allocation selected {len(selected)} for {args.source_key}; required {minimum}..{maximum}"
        )

    source_failures, integrity_violations = _source_contract_failure(
        selected, timeline, config, args.source_key
    )
    failure.extend(source_failures)

    outputs: list[dict[str, Any]] = []
    if not failure:
        sheets = args.output_dir / "contact_sheets"
        for index, plan in enumerate(selected, 1):
            suffix = "250M" if args.mode == "production" else "SHADOW"
            name = (
                f"MW4_V31_{args.source_key}_{index:02d}_"
                f"{plan.story_type}_{plan.effect_profile}_{suffix}.mp4"
            )
            target = args.output_dir / name
            render_candidate(args.source, plan, config, target, mode=args.mode)
            qa = validate_output(target, config, mode=args.mode)
            create_contact_sheets(target, sheets)
            payload = _candidate_dict(plan)
            outputs.append({
                "file": name,
                "sha256": sha256(target),
                "plan_key": payload["plan_key"],
                "editorial_plan": payload,
                "qa": qa,
            })

    selected_finishers = [
        plan for plan in selected
        if plan.story_type == "finishing_move_open" and plan.finishing_move is not None
    ]
    selected_payloads = [_candidate_dict(item) for item in selected]
    manifest = {
        "version": "3.1",
        "mode": args.mode,
        "source_key": args.source_key,
        "semantic_engine": ENGINE,
        "editorial_planner": EDITOR,
        "candidate_mode": CANDIDATE_MODE,
        "allocation_mode": allocation.get("allocation_mode"),
        "allocation_target_count": allocation.get("target_count"),
        "diagnostics": diagnostics,
        "candidate_count_after_semantic_gates": len(plans),
        "selected": selected_payloads,
        "verified_finishing_move_count": len(timeline.finishing_moves),
        "selected_finishing_move_count": len(selected_finishers),
        "automatic_finishing_move_candidates": automatic_candidates,
        "unplanned_source_cuts": integrity_violations,
        "unplanned_source_cut_count": len(integrity_violations),
        "finishing_move_policy": (
            "verified Finishing Moves are opening-only, full-frame 1.0x protected hero events; "
            "no semantic-montage continuation and no transition flash are permitted"
        ),
        "full_source_frame": True,
        "hook_text_added": False,
        "campaign_text_added": False,
        "campaign_logo_added": False,
        "source_audio_only": True,
        "output_settings": config["output"],
        "outputs": outputs,
        "failure": failure or None,
    }
    _write_json(manifest_path, manifest)

    summary = {
        "mode": args.mode,
        "source_key": args.source_key,
        "selected_count": len(selected),
        "rendered_count": len(outputs),
        "selected_plan_keys": [item["plan_key"] for item in selected_payloads],
        "stories": [plan.story_type for plan in selected],
        "verified_finishing_move_count": len(timeline.finishing_moves),
        "selected_finishing_move_count": len(selected_finishers),
        "unplanned_source_cut_count": len(integrity_violations),
        "technical_qa_passed": (
            len(outputs) == len(selected)
            and all(all(item["qa"]["checks"].values()) for item in outputs)
        ),
        "manifest": manifest_path.name,
        "failure": failure or None,
    }
    _write_json(summary_path, summary)

    if failure:
        raise RuntimeError("; ".join(failure))

    print(json.dumps({
        "source": args.source_key,
        "mode": args.mode,
        "selected": len(selected),
        "rendered": len(outputs),
        "stories": summary["stories"],
        "verified_finishing_moves": len(timeline.finishing_moves),
        "selected_finishing_moves": len(selected_finishers),
        "unplanned_source_cuts": len(integrity_violations),
    }))


if __name__ == "__main__":
    main()
