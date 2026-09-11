from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mw4_semantic_gameplay_v3_1 as semantic


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


def _append_segment(parts: list[str], index: int, start: float, end: float, speed: float) -> tuple[str, str]:
    v = f"sv{index}"
    a = f"sa{index}"
    vpts = "PTS-STARTPTS" if abs(speed - 1.0) < 1e-6 else f"(PTS-STARTPTS)/{speed:.6f}"
    parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts={vpts}[{v}]")
    achain = f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS"
    if abs(speed - 1.0) >= 1e-6:
        achain += f",atempo={speed:.6f}"
    parts.append(achain + f"[{a}]")
    return v, a


def source_time_to_output(source_time: float, segments: tuple[semantic.EditSegment, ...]) -> float | None:
    out = 0.0
    for segment in segments:
        if segment.start <= source_time <= segment.end:
            return out + (source_time - segment.start) / segment.speed
        out += (segment.end - segment.start) / segment.speed
    return None


def gate(times: list[float], before: float, after: float) -> str:
    return "+".join(
        f"between(t,{max(0.0, value-before):.3f},{value+after:.3f})" for value in times
    ) or "0"


def build_filter(plan: semantic.SemanticPlanV31, config: dict[str, Any]) -> tuple[str, float]:
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
        mapped for event in plan.effect_events
        if (mapped := source_time_to_output(event.time, plan.segments)) is not None
    ]
    profile = plan.effect_profile

    if profile == "finishing_move_hero" and plan.finishing_move is not None:
        trigger = source_time_to_output(plan.finishing_move.start, plan.segments)
        payoff = source_time_to_output(plan.finishing_move.payoff, plan.segments)
        finish_end = source_time_to_output(plan.finishing_move.end, plan.segments)
        if trigger is None or payoff is None:
            parts.append("[vsrc]scale=1920:1080:flags=lanczos,setsar=1[outv]")
        else:
            span_end = finish_end if finish_end is not None else payoff + 0.35
            parts.extend([
                "[vsrc]split=3[vbase][vhero][vimpact]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vhero]crop=1888:1062:(iw-1888)/2:(ih-1062)/2,scale=1920:1080:flags=lanczos,setsar=1[hero]",
                "[vimpact]scale=1944:1094:flags=lanczos,crop=1920:1080:x='12+5*sin(95*t)':y='7+2*sin(79*t)',setsar=1[impact]",
                f"[base][hero]overlay=0:0:enable='between(t,{max(0.0, trigger):.3f},{max(trigger, span_end):.3f})'[fx1]",
                f"[fx1][impact]overlay=0:0:enable='{gate([payoff],.055,.135)}'[outv]",
            ])
    elif profile == "precision_punch" and times:
        first = times[0]
        second = times[1] if len(times) > 1 else first
        parts.extend([
            "[vsrc]split=3[vbase][vz1][vz2]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vz1]crop=1860:1046:(iw-1860)/2:(ih-1046)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
            "[vz2]crop=1828:1028:(iw-1828)/2:(ih-1028)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
            f"[base][z1]overlay=0:0:enable='{gate([first],.08,.22)}'[fx1]",
            f"[fx1][z2]overlay=0:0:enable='{gate([second],.10,.28)}'[outv]",
        ])
    elif profile == "impact_flash" and times:
        impact_times = times[:2]
        parts.extend([
            "[vsrc]split=2[vbase][vshake]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vshake]scale=1944:1094:flags=lanczos,crop=1920:1080:x='12+6*sin(100*t)':y='7+3*sin(83*t)',setsar=1[shake]",
            f"[base][shake]overlay=0:0:enable='{gate(impact_times,.05,.14)}'[fx1]",
            f"[fx1]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.08:t=fill:enable='{gate(impact_times,.01,.04)}'[outv]",
        ])
    elif profile == "chain_escalation" and len(times) >= 3:
        first, second, third = times[:3]
        parts.extend([
            "[vsrc]split=4[vbase][vz1][vz2][vz3]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vz1]crop=1870:1052:(iw-1870)/2:(ih-1052)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
            "[vz2]crop=1838:1034:(iw-1838)/2:(ih-1034)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
            "[vz3]crop=1806:1016:(iw-1806)/2:(ih-1016)/2,scale=1920:1080:flags=lanczos,setsar=1[z3]",
            f"[base][z1]overlay=0:0:enable='{gate([first],.07,.18)}'[fx1]",
            f"[fx1][z2]overlay=0:0:enable='{gate([second],.08,.22)}'[fx2]",
            f"[fx2][z3]overlay=0:0:enable='{gate([third],.10,.28)}'[outv]",
        ])
    elif profile == "clean_pressure" and times:
        strongest = times[0]
        parts.extend([
            "[vsrc]split=2[vbase][vz]",
            "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
            "[vz]crop=1890:1063:(iw-1890)/2:(ih-1063)/2,scale=1920:1080:flags=lanczos,setsar=1[z]",
            f"[base][z]overlay=0:0:enable='{gate([strongest],.06,.17)}'[outv]",
        ])
    else:
        parts.append("[vsrc]scale=1920:1080:flags=lanczos,setsar=1[outv]")

    parts.append("[asrc]aresample=48000[aout]")
    return ";".join(parts), plan.output_duration


def render_candidate(source: Path, plan: semantic.SemanticPlanV31, config: dict[str, Any], output: Path) -> None:
    graph, duration = build_filter(plan, config)
    settings = config["output"]
    bitrate = int(settings["video_bitrate_kbps"])
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-filter_complex", graph, "-map", "[outv]", "-map", "[aout]",
        "-c:v", str(settings["codec"]), "-preset", str(settings["preset"]),
        "-profile:v", str(settings["profile"]), "-level:v", "5.2", "-pix_fmt", "yuv420p",
        "-b:v", f"{bitrate}k", "-minrate", f"{bitrate}k", "-maxrate", f"{bitrate}k",
        "-bufsize", f"{bitrate*2}k", "-x264-params", "nal-hrd=cbr",
        "-c:a", str(settings["audio_codec"]), "-b:a", f"{int(settings['audio_bitrate_kbps'])}k",
        "-ar", str(settings["audio_sample_rate"]), "-ac", str(settings["audio_channels"]),
        "-movflags", "+faststart", "-t", f"{duration:.3f}", str(output),
    ]
    run(command)


def validate_output(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    data = probe(path)
    video = next(item for item in data["streams"] if item["codec_type"] == "video")
    audio = next(item for item in data["streams"] if item["codec_type"] == "audio")
    fmt = data["format"]
    settings = config["output"]
    duration = float(fmt["duration"])
    bitrate = int(fmt.get("bit_rate") or video.get("bit_rate") or 0)
    expected = int(settings["video_bitrate_kbps"]) * 1000
    checks = {
        "duration_10_to_20s": 10.0 <= duration <= 20.0,
        "full_frame_1920x1080": (video["width"], video["height"]) == (1920, 1080),
        "codec_h264": video["codec_name"] == "h264",
        "profile_high": str(video.get("profile", "")).lower() == "high",
        "r_frame_rate": video["r_frame_rate"] == settings["fps"],
        "avg_frame_rate": video["avg_frame_rate"] == settings["fps"],
        "high_bitrate_near_250mbps": bitrate >= int(expected * 0.94),
        "audio_48khz": audio["sample_rate"] == "48000",
        "audio_stereo": audio["channels"] == 2,
    }
    run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if not all(checks.values()):
        raise RuntimeError(f"QA failed for {path.name}: {checks}")
    return {"checks": checks, "probe": data}


def create_contact_sheets(video: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", "fps=4,scale=320:-1,tile=6x5:padding=2:margin=2",
        "-frames:v", "2", str(output_dir / f"{video.stem}_%02d.jpg"),
    ])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    timeline = semantic.analyze_source(args.source, config)
    plans = semantic.build_plans(timeline, config, config.get("excluded_windows", {}).get(args.source_key, []))
    selected = semantic.select_plans(plans, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shadow = {
        "source_key": args.source_key,
        "semantic_engine": "deterministic-gameplay-v3.1",
        "editorial_planner": "semantic-editor-v3.1",
        "shot_count": len(timeline.shots),
        "engagement_count": len(timeline.engagements),
        "finishing_move_like_count": len(timeline.finishing_moves),
        "finishing_moves": [asdict(item) for item in timeline.finishing_moves],
        "selected": [asdict(item) for item in selected],
    }
    (args.output_dir / f"{args.source_key}_shadow.json").write_text(json.dumps(shadow, indent=2), encoding="utf-8")
    if args.analysis_only:
        print(json.dumps({"source": args.source_key, "selected": len(selected), "analysis_only": True}))
        return

    outputs: list[dict[str, Any]] = []
    sheets = args.output_dir / "contact_sheets"
    for index, plan in enumerate(selected, 1):
        name = f"MW4_V31_{args.source_key}_{index:02d}_{plan.story_type}_{plan.effect_profile}_250M.mp4"
        target = args.output_dir / name
        render_candidate(args.source, plan, config, target)
        qa = validate_output(target, config)
        create_contact_sheets(target, sheets)
        outputs.append({
            "file": name,
            "sha256": sha256(target),
            "editorial_plan": asdict(plan),
            "qa": qa,
        })

    manifest = {
        "version": "3.1",
        "source_key": args.source_key,
        "semantic_engine": "deterministic-gameplay-v3.1",
        "editorial_planner": "semantic-editor-v3.1",
        "candidate_mode": "engagement_driven",
        "finishing_move_policy": "detected finishing-move-like choreography must open the clip and is protected from cuts/speedups",
        "finishing_move_classifier_scope": "heuristic hypothesis only; not a guaranteed exact move-name classifier",
        "full_source_frame": True,
        "hook_text_added": False,
        "campaign_text_added": False,
        "campaign_logo_added": False,
        "source_audio_only": True,
        "output_settings": config["output"],
        "outputs": outputs,
    }
    (args.output_dir / f"{args.source_key}_manifest_v3_1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": args.source_key,
        "rendered": len(outputs),
        "finishing_move_like": len(timeline.finishing_moves),
        "profiles": [item["editorial_plan"]["effect_profile"] for item in outputs],
    }))


if __name__ == "__main__":
    main()
