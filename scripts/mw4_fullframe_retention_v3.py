from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import mw4_semantic_gameplay_v3 as semantic


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def probe(path: Path) -> dict[str, object]:
    result = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,"
                "r_frame_rate,avg_frame_rate,bit_rate,sample_rate,channels:"
                "format=duration,size,bit_rate"
            ),
            "-of", "json", str(path),
        ],
        capture=True,
    )
    return json.loads(result.stdout)


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def conflicts(start: float, end: float, windows: list[list[float]], margin: float = 0.0) -> bool:
    return any(
        overlap_seconds(start-margin, end+margin, float(window[0]), float(window[1])) > 0
        for window in windows
    )


def discover_candidates(
    source: Path,
    source_key: str,
    config: dict[str, object],
) -> tuple[semantic.SemanticTimeline, list[semantic.SemanticPlan]]:
    timeline = semantic.analyze_source(source, config)
    durations = [float(value) for value in config["duration_choices_seconds"]]
    step = float(config["candidate_step_seconds"])
    gap = float(config["minimum_gap_seconds"])
    desired = int(config["count_per_source"])
    minimum = int(config.get("minimum_count_per_source", 2))
    excluded = config["excluded_windows"].get(source_key, [])

    pool: list[semantic.SemanticPlan] = []
    for raw_duration in durations:
        start = 0.0
        latest = timeline.duration - raw_duration - 0.15
        while start <= latest:
            end = start + raw_duration
            if not conflicts(start, end, excluded, margin=0.15):
                plan = semantic.plan_window(timeline, start, end, config)
                if plan is not None:
                    pool.append(plan)
            start += step

    pool.sort(key=lambda item: item.score, reverse=True)
    selected: list[semantic.SemanticPlan] = []

    def compatible(candidate: semantic.SemanticPlan) -> bool:
        return not any(
            overlap_seconds(
                candidate.start-gap,
                candidate.end+gap,
                chosen.start,
                chosen.end,
            ) > 0
            for chosen in selected
        )

    # First pass deliberately diversifies editorial story types instead of simply taking
    # four nearly identical high-activity windows.
    seen_stories: set[str] = set()
    for candidate in pool:
        if candidate.story_type in seen_stories or not compatible(candidate):
            continue
        selected.append(candidate)
        seen_stories.add(candidate.story_type)
        if len(selected) == desired:
            break

    # Second pass fills remaining capacity with the strongest non-overlapping plans.
    if len(selected) < desired:
        for candidate in pool:
            if candidate in selected or not compatible(candidate):
                continue
            selected.append(candidate)
            if len(selected) == desired:
                break

    if len(selected) < minimum:
        raise RuntimeError(
            f"{source_key}: only {len(selected)} semantic candidates passed the no-dull/editorial gates; "
            f"minimum is {minimum}. Refusing to lower quality gates."
        )

    selected.sort(key=lambda item: item.start)
    return timeline, selected


def gate(times: list[float], before: float, after: float) -> str:
    return "+".join(
        f"between(t,{max(0.0, value-before):.3f},{value+after:.3f})"
        for value in times
    ) or "0"


def _append_segment(
    parts: list[str],
    index: int,
    start: float,
    end: float,
    speed: float,
) -> tuple[str, str]:
    video_label = f"sv{index}"
    audio_label = f"sa{index}"
    video_setpts = "PTS-STARTPTS" if abs(speed-1.0) < 1e-6 else f"(PTS-STARTPTS)/{speed:.6f}"
    parts.append(
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts={video_setpts}[{video_label}]"
    )
    audio_chain = (
        f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS"
    )
    if abs(speed-1.0) >= 1e-6:
        audio_chain += f",atempo={speed:.6f}"
    parts.append(audio_chain + f"[{audio_label}]")
    return video_label, audio_label


def _effect_times(plan: semantic.SemanticPlan, teaser_seconds: float) -> list[float]:
    prefix = teaser_seconds if plan.cold_open_source_start is not None else 0.0
    mapped: list[float] = []
    if plan.cold_open_source_start is not None and plan.effect_events:
        strongest = plan.effect_events[0]
        mapped.append(max(0.0, strongest.time-plan.cold_open_source_start))
    for event in plan.effect_events:
        output_time = semantic.source_time_to_output(event.time, plan.segments, prefix_seconds=prefix)
        if output_time is not None:
            mapped.append(output_time)
    deduped: list[float] = []
    for value in mapped:
        if all(abs(value-prior) >= 0.10 for prior in deduped):
            deduped.append(value)
    return deduped


def build_filter(plan: semantic.SemanticPlan, config: dict[str, object]) -> tuple[str, float]:
    parts: list[str] = []
    pairs: list[tuple[str, str]] = []
    teaser_seconds = float(config["editorial"]["cold_open_teaser_seconds"])
    segment_index = 0

    if plan.cold_open_source_start is not None:
        pairs.append(
            _append_segment(
                parts,
                segment_index,
                plan.cold_open_source_start,
                plan.cold_open_source_start+teaser_seconds,
                1.0,
            )
        )
        segment_index += 1

    for segment in plan.segments:
        pairs.append(
            _append_segment(parts, segment_index, segment.start, segment.end, segment.speed)
        )
        segment_index += 1

    if len(pairs) == 1:
        video_label, audio_label = pairs[0]
        parts.append(f"[{video_label}]null[vsrc]")
        parts.append(f"[{audio_label}]anull[asrc]")
    else:
        concat_inputs = "".join(f"[{video}][{audio}]" for video, audio in pairs)
        parts.append(f"{concat_inputs}concat=n={len(pairs)}:v=1:a=1[vsrc][asrc]")

    times = _effect_times(plan, teaser_seconds)
    profile = plan.effect_profile

    if profile == "precision_punch" and times:
        first = times[0]
        second = times[1] if len(times) > 1 else first
        parts.extend(
            [
                "[vsrc]split=3[vbase][vz1][vz2]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz1]crop=1860:1046:(iw-1860)/2:(ih-1046)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
                "[vz2]crop=1828:1028:(iw-1828)/2:(ih-1028)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
                f"[base][z1]overlay=0:0:enable='{gate([first],.08,.23)}'[fx1]",
                f"[fx1][z2]overlay=0:0:enable='{gate([second],.10,.30)}'[outv]",
            ]
        )
    elif profile == "impact_flash" and times:
        impact_times = times[:2]
        parts.extend(
            [
                "[vsrc]split=2[vbase][vshake]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vshake]scale=1944:1094:flags=lanczos,crop=1920:1080:x='12+6*sin(100*t)':y='7+3*sin(83*t)',setsar=1[shake]",
                f"[base][shake]overlay=0:0:enable='{gate(impact_times,.05,.14)}'[fx1]",
                f"[fx1]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=fill:enable='{gate(impact_times,.01,.05)}'[outv]",
            ]
        )
    elif profile == "chain_escalation" and len(times) >= 3:
        first, second, third = times[:3]
        parts.extend(
            [
                "[vsrc]split=4[vbase][vz1][vz2][vz3]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz1]crop=1870:1052:(iw-1870)/2:(ih-1052)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
                "[vz2]crop=1838:1034:(iw-1838)/2:(ih-1034)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
                "[vz3]crop=1806:1016:(iw-1806)/2:(ih-1016)/2,scale=1920:1080:flags=lanczos,setsar=1[z3]",
                f"[base][z1]overlay=0:0:enable='{gate([first],.07,.20)}'[fx1]",
                f"[fx1][z2]overlay=0:0:enable='{gate([second],.08,.24)}'[fx2]",
                f"[fx2][z3]overlay=0:0:enable='{gate([third],.10,.30)}'[outv]",
            ]
        )
    elif profile == "cold_open_teaser" and times:
        first = times[0]
        later = times[-1]
        parts.extend(
            [
                "[vsrc]split=2[vbase][vz]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz]crop=1838:1034:(iw-1838)/2:(ih-1034)/2,scale=1920:1080:flags=lanczos,setsar=1[z]",
                f"[base][z]overlay=0:0:enable='{gate([first,later],.08,.26)}'[outv]",
            ]
        )
    elif profile == "clean_pressure" and times:
        strongest = times[0]
        parts.extend(
            [
                "[vsrc]split=2[vbase][vz]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz]crop=1890:1063:(iw-1890)/2:(ih-1063)/2,scale=1920:1080:flags=lanczos,setsar=1[z]",
                f"[base][z]overlay=0:0:enable='{gate([strongest],.06,.18)}'[outv]",
            ]
        )
    else:
        parts.append("[vsrc]scale=1920:1080:flags=lanczos,setsar=1[outv]")

    parts.append("[asrc]aresample=48000[aout]")
    return ";".join(parts), plan.output_duration


def render_candidate(
    source: Path,
    plan: semantic.SemanticPlan,
    config: dict[str, object],
    output: Path,
) -> float:
    filter_graph, duration = build_filter(plan, config)
    settings = config["output"]
    bitrate = int(settings["video_bitrate_kbps"])
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-filter_complex", filter_graph,
        "-map", "[outv]", "-map", "[aout]",
        "-c:v", str(settings["codec"]), "-preset", str(settings["preset"]),
        "-profile:v", str(settings["profile"]), "-level:v", "5.2", "-pix_fmt", "yuv420p",
        "-b:v", f"{bitrate}k", "-minrate", f"{bitrate}k", "-maxrate", f"{bitrate}k",
        "-bufsize", f"{bitrate*2}k", "-x264-params", "nal-hrd=cbr",
        "-c:a", str(settings["audio_codec"]), "-b:a", f"{int(settings['audio_bitrate_kbps'])}k",
        "-ar", str(settings["audio_sample_rate"]), "-ac", str(settings["audio_channels"]),
        "-movflags", "+faststart", "-t", f"{duration:.3f}", str(output),
    ]
    run(command)
    return duration


def validate_output(path: Path, config: dict[str, object]) -> dict[str, object]:
    data = probe(path)
    streams = data["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    fmt = data["format"]
    settings = config["output"]
    duration = float(fmt["duration"])
    bitrate = int(fmt.get("bit_rate") or video.get("bit_rate") or 0)
    expected = int(settings["video_bitrate_kbps"])*1000
    checks = {
        "duration_10_to_20s": 10.0 <= duration <= 20.0,
        "full_frame_1920x1080": (video["width"], video["height"]) == (1920,1080),
        "codec_h264": video["codec_name"] == "h264",
        "profile_high": str(video.get("profile","")).lower() == "high",
        "r_frame_rate": video["r_frame_rate"] == settings["fps"],
        "avg_frame_rate": video["avg_frame_rate"] == settings["fps"],
        "high_bitrate_near_250mbps": bitrate >= int(expected*.94),
        "audio_48khz": audio["sample_rate"] == "48000",
        "audio_stereo": audio["channels"] == 2,
    }
    run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"])
    if not all(checks.values()):
        raise RuntimeError(f"QA failed for {path.name}: {checks}; probe={data}")
    return {"checks": checks, "probe": data}


def create_contact_sheets(video: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(video),
            "-vf","fps=4,scale=320:-1,tile=6x5:padding=2:margin=2",
            "-frames:v","3",str(output_dir/f"{video.stem}_sheet_%02d.jpg"),
        ]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="MW4 V3 semantic gameplay editor + deterministic 250 Mbps renderer")
    parser.add_argument("--source-key", choices=("r1","batch2","week2"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline, plans = discover_candidates(args.source, args.source_key, config)

    (args.output_dir/f"{args.source_key}_semantic_timeline.json").write_text(
        json.dumps(semantic.timeline_summary(timeline), indent=2), encoding="utf-8"
    )
    (args.output_dir/f"{args.source_key}_selected_editorial_plans.json").write_text(
        json.dumps([asdict(plan) for plan in plans], indent=2), encoding="utf-8"
    )

    outputs: list[dict[str, object]] = []
    for index, plan in enumerate(plans, start=1):
        filename = (
            f"MW4_FULLFRAME_SEMANTIC_V3_{args.source_key.upper()}_{index:02d}_"
            f"{plan.effect_profile.upper()}_250M.mp4"
        )
        output = args.output_dir/filename
        planned_duration = render_candidate(args.source, plan, config, output)
        qa = validate_output(output, config)
        create_contact_sheets(output, args.output_dir/"contact_sheets")
        outputs.append(
            {
                "file": filename,
                "editorial_plan": asdict(plan),
                "planned_duration": round(planned_duration,3),
                "sha256": sha256(output),
                "qa": qa,
            }
        )

    manifest = {
        "campaign": config["campaign"],
        "source_key": args.source_key,
        "semantic_engine": "deterministic-gameplay-v1",
        "editorial_planner": "semantic-editor-v1",
        "semantic_labels_are_hypotheses_not_ground_truth_kills": True,
        "full_source_frame": True,
        "hook_text_added": False,
        "campaign_text_added": False,
        "campaign_logo_added": False,
        "source_audio_only": True,
        "retention_target_percent": config["performance_targets"]["retention_target_percent"],
        "engagement_goal": config["performance_targets"]["engagement_goal"],
        "actual_platform_retention_or_engagement_guaranteed": False,
        "no_dull_moment_contract": config["performance_targets"]["no_dull_moment_contract"],
        "prior_work_overlap_allowed": False,
        "new_batch_overlap_allowed": False,
        "output_settings": config["output"],
        "outputs": outputs,
    }
    (args.output_dir/f"{args.source_key}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
