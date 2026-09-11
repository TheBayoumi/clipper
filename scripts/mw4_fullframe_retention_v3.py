from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import mw4_gameplay_batch as base

SAMPLE_FPS = base.SAMPLE_FPS
PROFILES = (
    "precision_punch",
    "impact_flash",
    "chain_escalation",
    "cold_open_teaser",
)
SOURCE_PROFILE_OFFSET = {"r1": 0, "batch2": 1, "week2": 2}


@dataclass(frozen=True)
class Candidate:
    source_key: str
    start: float
    end: float
    score: float
    retention_proxy: float
    engagement_proxy: float
    opening_activity: float
    closing_activity: float
    weakest_quarter_activity: float
    low_activity_fraction: float
    max_low_run_seconds: float
    peak_times: tuple[float, ...]
    effect_profile: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def probe(path: Path) -> dict[str, object]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,"
                "r_frame_rate,avg_frame_rate,bit_rate,sample_rate,channels:"
                "format=duration,size,bit_rate"
            ),
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(result.stdout)


def longest_low_run_seconds(values: np.ndarray, threshold: float) -> float:
    longest = 0
    current = 0
    for value in values:
        if float(value) < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / SAMPLE_FPS


def candidate_metrics(
    activity: np.ndarray,
    motion: np.ndarray,
    start: float,
    end: float,
) -> dict[str, object] | None:
    start_index = max(0, int(math.floor(start * SAMPLE_FPS)))
    end_index = min(len(activity), int(math.ceil(end * SAMPLE_FPS)))
    segment = activity[start_index:end_index]
    segment_motion = motion[start_index:end_index]
    if segment.size < int(10.0 * SAMPLE_FPS):
        return None

    quarters = [part for part in np.array_split(segment, 4) if part.size]
    quarter_means = [float(np.mean(part)) for part in quarters]
    opening_bins = max(2, int(round(1.5 * SAMPLE_FPS)))
    closing_bins = opening_bins
    opening = float(np.mean(segment[:opening_bins]))
    closing = float(np.mean(segment[-closing_bins:]))
    mean_activity = float(np.mean(segment))
    p82 = float(np.percentile(segment, 82))
    p92 = float(np.percentile(segment, 92))
    weakest_quarter = min(quarter_means)
    low_fraction = float(np.mean(segment < 0.24))
    static_fraction = float(np.mean(segment_motion < 0.12))
    max_low_run = longest_low_run_seconds(segment, 0.22)
    peaks = base.choose_peaks(activity, start_index, end_index)
    peak_density = min(1.0, len(peaks) / 3.0)

    retention_proxy = (
        0.29 * mean_activity
        + 0.18 * p82
        + 0.20 * weakest_quarter
        + 0.15 * opening
        + 0.10 * closing
        + 0.08 * peak_density
        - 0.12 * low_fraction
        - 0.07 * static_fraction
        - 0.04 * min(1.0, max_low_run / 3.0)
    )
    engagement_proxy = (
        0.34 * p92
        + 0.24 * opening
        + 0.16 * closing
        + 0.16 * peak_density
        + 0.10 * mean_activity
        - 0.08 * low_fraction
    )
    combined = (
        0.46 * retention_proxy
        + 0.36 * engagement_proxy
        + 0.10 * weakest_quarter
        + 0.08 * opening
    )

    return {
        "score": combined,
        "retention_proxy": retention_proxy,
        "engagement_proxy": engagement_proxy,
        "opening_activity": opening,
        "closing_activity": closing,
        "weakest_quarter_activity": weakest_quarter,
        "low_activity_fraction": low_fraction,
        "max_low_run_seconds": max_low_run,
        "peaks": peaks,
    }


def discover_candidates(source: Path, source_key: str, config: dict[str, object]) -> list[Candidate]:
    activity, motion = base.analyze_activity(source)
    duration = base.source_duration(source)
    durations = [float(value) for value in config["duration_choices_seconds"]]
    step = float(config["candidate_step_seconds"])
    gap = float(config["minimum_gap_seconds"])
    count = int(config["count_per_source"])
    excluded = config["excluded_windows"].get(source_key, [])
    targets = config["performance_targets"]

    retention_min = float(targets["retention_proxy_min"])
    engagement_min = float(targets["engagement_proxy_min"])
    weak_quarter_min = float(targets["minimum_weak_quarter_activity"])
    low_fraction_max = float(targets["maximum_low_activity_fraction"])

    pool: list[dict[str, object]] = []
    for clip_duration in durations:
        start = 0.0
        latest = duration - clip_duration - 0.15
        while start <= latest:
            end = start + clip_duration
            if base.conflicts(start, end, excluded, margin=0.15):
                start += step
                continue
            metrics = candidate_metrics(activity, motion, start, end)
            if metrics is None:
                start += step
                continue
            peaks = metrics["peaks"]
            if (
                len(peaks) >= 3
                and float(metrics["retention_proxy"]) >= retention_min
                and float(metrics["engagement_proxy"]) >= engagement_min
                and float(metrics["weakest_quarter_activity"]) >= weak_quarter_min
                and float(metrics["low_activity_fraction"]) <= low_fraction_max
                and float(metrics["max_low_run_seconds"]) <= 3.0
            ):
                pool.append({"start": start, "end": end, **metrics})
            start += step

    pool.sort(key=lambda item: float(item["score"]), reverse=True)
    selected: list[dict[str, object]] = []
    for candidate in pool:
        start = float(candidate["start"])
        end = float(candidate["end"])
        if any(
            base.overlap_seconds(
                start - gap,
                end + gap,
                float(chosen["start"]),
                float(chosen["end"]),
            )
            > 0
            for chosen in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"{source_key}: only {len(selected)} candidates passed the retention/engagement quality gates; "
            f"required {count}. Refusing to lower the gate just to hit output count."
        )

    offset = SOURCE_PROFILE_OFFSET[source_key]
    results: list[Candidate] = []
    for index, item in enumerate(selected):
        profile = PROFILES[(index + offset) % len(PROFILES)]
        results.append(
            Candidate(
                source_key=source_key,
                start=round(float(item["start"]), 3),
                end=round(float(item["end"]), 3),
                score=round(float(item["score"]), 6),
                retention_proxy=round(float(item["retention_proxy"]), 6),
                engagement_proxy=round(float(item["engagement_proxy"]), 6),
                opening_activity=round(float(item["opening_activity"]), 6),
                closing_activity=round(float(item["closing_activity"]), 6),
                weakest_quarter_activity=round(float(item["weakest_quarter_activity"]), 6),
                low_activity_fraction=round(float(item["low_activity_fraction"]), 6),
                max_low_run_seconds=round(float(item["max_low_run_seconds"]), 3),
                peak_times=tuple(round(float(value), 3) for value in item["peaks"]),
                effect_profile=profile,
            )
        )
    return results


def gate(times: list[float], before: float, after: float) -> str:
    return "+".join(
        f"between(t,{max(0.0, value - before):.3f},{value + after:.3f})" for value in times
    ) or "0"


def output_events(candidate: Candidate, teaser_seconds: float) -> tuple[list[float], float | None]:
    base_events = [peak - candidate.start for peak in candidate.peak_times]
    if candidate.effect_profile != "cold_open_teaser":
        return base_events, None
    strongest = candidate.peak_times[1]
    teaser_start = min(max(candidate.start, strongest - 0.40), candidate.end - teaser_seconds)
    teaser_peak = strongest - teaser_start
    shifted_main = [teaser_seconds + value for value in base_events]
    return [teaser_peak, *shifted_main], teaser_start


def build_filter(candidate: Candidate, config: dict[str, object]) -> tuple[str, float]:
    teaser_seconds = float(config["editorial"]["cold_open_teaser_seconds"])
    events, teaser_start = output_events(candidate, teaser_seconds)
    parts: list[str] = []

    if teaser_start is None:
        parts.extend(
            [
                f"[0:v]trim=start={candidate.start:.3f}:end={candidate.end:.3f},setpts=PTS-STARTPTS[vsrc]",
                f"[0:a]atrim=start={candidate.start:.3f}:end={candidate.end:.3f},asetpts=PTS-STARTPTS[asrc]",
            ]
        )
        output_duration = candidate.duration
    else:
        teaser_end = teaser_start + teaser_seconds
        parts.extend(
            [
                f"[0:v]trim=start={teaser_start:.3f}:end={teaser_end:.3f},setpts=PTS-STARTPTS[tv]",
                f"[0:a]atrim=start={teaser_start:.3f}:end={teaser_end:.3f},asetpts=PTS-STARTPTS[ta]",
                f"[0:v]trim=start={candidate.start:.3f}:end={candidate.end:.3f},setpts=PTS-STARTPTS[mv]",
                f"[0:a]atrim=start={candidate.start:.3f}:end={candidate.end:.3f},asetpts=PTS-STARTPTS[ma]",
                "[tv][ta][mv][ma]concat=n=2:v=1:a=1[vsrc][asrc]",
            ]
        )
        output_duration = candidate.duration + teaser_seconds

    event3 = events[:3]
    first = event3[0]
    second = event3[1] if len(event3) > 1 else first
    third = event3[2] if len(event3) > 2 else second

    if candidate.effect_profile == "precision_punch":
        parts.extend(
            [
                "[vsrc]split=3[vbase][vz1][vz2]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz1]crop=1860:1046:(iw-1860)/2:(ih-1046)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
                "[vz2]crop=1828:1028:(iw-1828)/2:(ih-1028)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
                f"[base][z1]overlay=0:0:enable='{gate([first, second], 0.08, 0.24)}'[fx1]",
                f"[fx1][z2]overlay=0:0:enable='{gate([third], 0.10, 0.30)}'[outv]",
            ]
        )
    elif candidate.effect_profile == "impact_flash":
        parts.extend(
            [
                "[vsrc]split=2[vbase][vshake]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vshake]scale=1944:1094:flags=lanczos,crop=1920:1080:x='12+6*sin(100*t)':y='7+3*sin(83*t)',setsar=1[shake]",
                f"[base][shake]overlay=0:0:enable='{gate(event3, 0.05, 0.14)}'[fx1]",
                f"[fx1]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=fill:enable='{gate(event3, 0.01, 0.05)}'[outv]",
            ]
        )
    elif candidate.effect_profile == "chain_escalation":
        parts.extend(
            [
                "[vsrc]split=4[vbase][vz1][vz2][vz3]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz1]crop=1870:1052:(iw-1870)/2:(ih-1052)/2,scale=1920:1080:flags=lanczos,setsar=1[z1]",
                "[vz2]crop=1838:1034:(iw-1838)/2:(ih-1034)/2,scale=1920:1080:flags=lanczos,setsar=1[z2]",
                "[vz3]crop=1806:1016:(iw-1806)/2:(ih-1016)/2,scale=1920:1080:flags=lanczos,setsar=1[z3]",
                f"[base][z1]overlay=0:0:enable='{gate([first], 0.07, 0.20)}'[fx1]",
                f"[fx1][z2]overlay=0:0:enable='{gate([second], 0.08, 0.24)}'[fx2]",
                f"[fx2][z3]overlay=0:0:enable='{gate([third], 0.10, 0.30)}'[outv]",
            ]
        )
    else:
        strongest = events[0]
        later = events[-2] if len(events) >= 2 else strongest
        parts.extend(
            [
                "[vsrc]split=2[vbase][vz]",
                "[vbase]scale=1920:1080:flags=lanczos,setsar=1[base]",
                "[vz]crop=1838:1034:(iw-1838)/2:(ih-1034)/2,scale=1920:1080:flags=lanczos,setsar=1[z]",
                f"[base][z]overlay=0:0:enable='{gate([strongest, later], 0.08, 0.26)}'[outv]",
            ]
        )

    parts.append("[asrc]aresample=48000[aout]")
    return ";".join(parts), output_duration


def render_candidate(
    source: Path,
    candidate: Candidate,
    config: dict[str, object],
    output: Path,
) -> float:
    filter_graph, output_duration = build_filter(candidate, config)
    settings = config["output"]
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-map",
        "[aout]",
        "-c:v",
        str(settings["codec"]),
        "-preset",
        str(settings["preset"]),
        "-profile:v",
        str(settings["profile"]),
        "-level:v",
        "5.2",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        f"{int(settings['video_bitrate_kbps'])}k",
        "-minrate",
        f"{int(settings['video_bitrate_kbps'])}k",
        "-maxrate",
        f"{int(settings['video_bitrate_kbps'])}k",
        "-bufsize",
        f"{int(settings['video_bitrate_kbps']) * 2}k",
        "-x264-params",
        "nal-hrd=cbr",
        "-c:a",
        str(settings["audio_codec"]),
        "-b:a",
        f"{int(settings['audio_bitrate_kbps'])}k",
        "-ar",
        str(settings["audio_sample_rate"]),
        "-ac",
        str(settings["audio_channels"]),
        "-movflags",
        "+faststart",
        "-t",
        f"{output_duration:.3f}",
        str(output),
    ]
    run(command)
    return output_duration


def validate_output(path: Path, config: dict[str, object]) -> dict[str, object]:
    data = probe(path)
    streams = data["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    fmt = data["format"]
    settings = config["output"]
    duration = float(fmt["duration"])
    bitrate = int(fmt.get("bit_rate") or video.get("bit_rate") or 0)
    expected_bitrate = int(settings["video_bitrate_kbps"]) * 1000

    checks = {
        "duration_at_least_10s": duration >= 10.0,
        "full_frame_1920x1080": (video["width"], video["height"]) == (1920, 1080),
        "codec_h264": video["codec_name"] == "h264",
        "profile_high": str(video.get("profile", "")).lower() == "high",
        "r_frame_rate": video["r_frame_rate"] == settings["fps"],
        "avg_frame_rate": video["avg_frame_rate"] == settings["fps"],
        "high_bitrate_near_250mbps": bitrate >= int(expected_bitrate * 0.94),
        "audio_48khz": audio["sample_rate"] == "48000",
        "audio_stereo": audio["channels"] == 2,
    }
    run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if not all(checks.values()):
        raise RuntimeError(f"QA failed for {path.name}: {checks}; probe={data}")
    return {"checks": checks, "probe": data}


def create_contact_sheets(video: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=4,scale=320:-1,tile=6x5:padding=2:margin=2",
            "-frames:v",
            "2",
            str(output_dir / f"{video.stem}_sheet_%02d.jpg"),
        ]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full-frame retention-focused MW4 gameplay clips")
    parser.add_argument("--source-key", choices=("r1", "batch2", "week2"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates(args.source, args.source_key, config)

    (args.output_dir / f"{args.source_key}_selected_candidates.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2),
        encoding="utf-8",
    )

    outputs: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        filename = (
            f"MW4_FULLFRAME_V3_{args.source_key.upper()}_{index:02d}_"
            f"{candidate.effect_profile.upper()}_250M.mp4"
        )
        output = args.output_dir / filename
        planned_duration = render_candidate(args.source, candidate, config, output)
        qa = validate_output(output, config)
        create_contact_sheets(output, args.output_dir / "contact_sheets")
        outputs.append(
            {
                "file": filename,
                "source": asdict(candidate),
                "planned_duration": round(planned_duration, 3),
                "sha256": sha256(output),
                "qa": qa,
            }
        )

    manifest = {
        "campaign": config["campaign"],
        "source_key": args.source_key,
        "full_source_frame": True,
        "hook_text_added": False,
        "campaign_text_added": False,
        "campaign_logo_added": False,
        "source_audio_only": True,
        "retention_target_percent": config["performance_targets"]["retention_target_percent"],
        "engagement_goal": config["performance_targets"]["engagement_goal"],
        "actual_platform_retention_or_engagement_guaranteed": False,
        "prior_work_overlap_allowed": False,
        "new_batch_overlap_allowed": False,
        "output_settings": config["output"],
        "outputs": outputs,
    }
    (args.output_dir / f"{args.source_key}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
