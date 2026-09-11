from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SAMPLE_FPS = 2.0
SAMPLE_WIDTH = 160
SAMPLE_HEIGHT = 90
AUDIO_RATE = 8000
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
    peak_times: tuple[float, ...]
    effect_profile: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


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


def source_duration(path: Path) -> float:
    data = probe(path)
    return float(data["format"]["duration"])


def robust_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 95))
    if high <= low + 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def analyze_activity(source: Path) -> tuple[np.ndarray, np.ndarray]:
    video_bytes = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"fps={SAMPLE_FPS},scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
    )
    frame_size = SAMPLE_WIDTH * SAMPLE_HEIGHT
    frame_count = len(video_bytes) // frame_size
    if frame_count < 2:
        raise RuntimeError("Not enough sampled video frames for discovery")
    frames = np.frombuffer(video_bytes[: frame_count * frame_size], dtype=np.uint8)
    frames = frames.reshape(frame_count, SAMPLE_HEIGHT, SAMPLE_WIDTH).astype(np.float32)
    motion = np.zeros(frame_count, dtype=np.float32)
    motion[1:] = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2)) / 255.0
    motion[0] = motion[1]

    audio_bytes = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
    )
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    samples_per_bin = int(AUDIO_RATE / SAMPLE_FPS)
    bin_count = len(audio) // samples_per_bin
    if bin_count < 2:
        raise RuntimeError("Not enough sampled audio for discovery")
    audio = audio[: bin_count * samples_per_bin].reshape(bin_count, samples_per_bin)
    rms = np.sqrt(np.mean(np.square(audio), axis=1))

    length = min(len(motion), len(rms))
    motion_n = robust_normalize(motion[:length])
    audio_n = robust_normalize(rms[:length])
    activity = 0.46 * motion_n + 0.54 * audio_n
    return activity.astype(np.float32), motion_n.astype(np.float32)


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def conflicts(start: float, end: float, windows: list[list[float]], margin: float = 0.0) -> bool:
    return any(
        overlap_seconds(start - margin, end + margin, float(window[0]), float(window[1])) > 0
        for window in windows
    )


def choose_peaks(activity: np.ndarray, start_index: int, end_index: int) -> tuple[float, ...]:
    local = activity[start_index:end_index]
    if local.size == 0:
        return ()
    ordered = np.argsort(local)[::-1]
    chosen: list[int] = []
    minimum_spacing = int(round(1.15 * SAMPLE_FPS))
    for raw_index in ordered:
        index = int(raw_index)
        if all(abs(index - prior) >= minimum_spacing for prior in chosen):
            chosen.append(index)
        if len(chosen) == 3:
            break
    chosen.sort()
    return tuple((start_index + index + 0.5) / SAMPLE_FPS for index in chosen)


def score_window(activity: np.ndarray, motion: np.ndarray, start: float, end: float) -> tuple[float, tuple[float, ...]]:
    start_index = max(0, int(math.floor(start * SAMPLE_FPS)))
    end_index = min(len(activity), int(math.ceil(end * SAMPLE_FPS)))
    segment = activity[start_index:end_index]
    segment_motion = motion[start_index:end_index]
    if segment.size < int(10 * SAMPLE_FPS):
        return -1.0, ()

    quarters = np.array_split(segment, 4)
    quarter_means = [float(np.mean(quarter)) for quarter in quarters if quarter.size]
    opening = float(np.mean(segment[: max(2, int(1.5 * SAMPLE_FPS))]))
    closing = float(np.mean(segment[-max(2, int(1.5 * SAMPLE_FPS)) :]))
    low_fraction = float(np.mean(segment < 0.24))
    static_fraction = float(np.mean(segment_motion < 0.12))
    peaks = choose_peaks(activity, start_index, end_index)

    score = (
        0.38 * float(np.mean(segment))
        + 0.20 * float(np.percentile(segment, 82))
        + 0.24 * min(quarter_means)
        + 0.10 * opening
        + 0.10 * closing
        + 0.035 * len(peaks)
        - 0.20 * low_fraction
        - 0.10 * static_fraction
    )
    return score, peaks


def discover_candidates(
    source: Path,
    source_key: str,
    config: dict[str, object],
) -> list[Candidate]:
    activity, motion = analyze_activity(source)
    duration = source_duration(source)
    durations = [float(value) for value in config["duration_choices_seconds"]]
    step = float(config["candidate_step_seconds"])
    count = int(config["count_per_source"])
    gap = float(config["minimum_gap_seconds"])
    excluded = config["excluded_windows"].get(source_key, [])

    candidates: list[tuple[float, float, float, tuple[float, ...]]] = []
    for clip_duration in durations:
        start = 0.0
        latest = duration - clip_duration - 0.15
        while start <= latest:
            end = start + clip_duration
            if not conflicts(start, end, excluded, margin=0.15):
                score, peaks = score_window(activity, motion, start, end)
                if score >= 0 and len(peaks) == 3:
                    candidates.append((score, start, end, peaks))
            start += step

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, float, float, tuple[float, ...]]] = []
    for candidate in candidates:
        _, start, end, _ = candidate
        if any(
            overlap_seconds(start - gap, end + gap, chosen[1], chosen[2]) > 0
            for chosen in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"{source_key}: found only {len(selected)} mutually non-overlapping new candidates; "
            f"required {count}"
        )

    offset = SOURCE_PROFILE_OFFSET[source_key]
    results: list[Candidate] = []
    for index, (score, start, end, peaks) in enumerate(selected):
        profile = PROFILES[(index + offset) % len(PROFILES)]
        results.append(
            Candidate(
                source_key=source_key,
                start=round(start, 3),
                end=round(end, 3),
                score=round(score, 6),
                peak_times=tuple(round(peak, 3) for peak in peaks),
                effect_profile=profile,
            )
        )
    return results


def gate(times: list[float], before: float, after: float) -> str:
    intervals = [
        f"between(t,{max(0.0, time - before):.3f},{time + after:.3f})" for time in times
    ]
    return "+".join(intervals) if intervals else "0"


def output_events(candidate: Candidate, teaser_seconds: float) -> tuple[list[float], float | None]:
    base = [peak - candidate.start for peak in candidate.peak_times]
    if candidate.effect_profile != "cold_open_teaser":
        return base, None
    strongest = candidate.peak_times[1]
    teaser_start = min(max(candidate.start, strongest - 0.40), candidate.end - teaser_seconds)
    teaser_peak = strongest - teaser_start
    shifted_main = [teaser_seconds + value for value in base]
    return [teaser_peak, *shifted_main], teaser_start


def build_filter(
    candidate: Candidate,
    config: dict[str, object],
    *,
    logo_enabled: bool,
) -> tuple[str, float]:
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

    parts.extend(
        [
            "[vsrc]split=5[vbase][vz1][vz2][vz3][vshake]",
            "[vbase]crop=608:1080:(iw-608)/2:0,scale=1080:1920:flags=lanczos,setsar=1[base]",
            "[vz1]crop=584:1038:(iw-584)/2:(ih-1038)/2,scale=1080:1920:flags=lanczos,setsar=1[z1]",
            "[vz2]crop=578:1028:(iw-578)/2:(ih-1028)/2,scale=1080:1920:flags=lanczos,setsar=1[z2]",
            "[vz3]crop=572:1018:(iw-572)/2:(ih-1018)/2,scale=1080:1920:flags=lanczos,setsar=1[z3]",
            "[vshake]crop=608:1080:x='(iw-608)/2+8*sin(100*t)':y=0,scale=1080:1920:flags=lanczos,setsar=1[shake]",
        ]
    )

    event3 = events[:3] if len(events) >= 3 else events
    first = event3[0]
    second = event3[1] if len(event3) > 1 else first
    third = event3[2] if len(event3) > 2 else second

    if candidate.effect_profile == "precision_punch":
        parts.extend(
            [
                f"[base][z1]overlay=0:0:enable='{gate([first, second], 0.10, 0.28)}'[fx1]",
                f"[fx1][z2]overlay=0:0:enable='{gate([third], 0.12, 0.34)}'[fx]",
            ]
        )
    elif candidate.effect_profile == "impact_flash":
        impact_gate = gate(event3, 0.07, 0.15)
        flash_gate = gate(event3, 0.02, 0.07)
        parts.extend(
            [
                f"[base][shake]overlay=0:0:enable='{impact_gate}'[fx1]",
                f"[fx1]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.12:t=fill:enable='{flash_gate}'[fx]",
            ]
        )
    elif candidate.effect_profile == "chain_escalation":
        parts.extend(
            [
                f"[base][z1]overlay=0:0:enable='{gate([first], 0.08, 0.24)}'[fx1]",
                f"[fx1][z2]overlay=0:0:enable='{gate([second], 0.10, 0.28)}'[fx2]",
                f"[fx2][z3]overlay=0:0:enable='{gate([third], 0.12, 0.34)}'[fx3]",
                f"[fx3][shake]overlay=0:0:enable='{gate([third], 0.04, 0.14)}'[fx4]",
                f"[fx4]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.10:t=fill:enable='{gate([third], 0.01, 0.06)}'[fx]",
            ]
        )
    else:
        strongest = events[0]
        main_strongest = events[-2] if len(events) >= 2 else strongest
        parts.extend(
            [
                f"[base][z2]overlay=0:0:enable='{gate([strongest, main_strongest], 0.10, 0.32)}'[fx1]",
                f"[fx1][shake]overlay=0:0:enable='{gate([strongest], 0.04, 0.14)}'[fx]",
            ]
        )

    branding = config["branding"]
    if branding["enabled"]:
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        parts.extend(
            [
                "[fx]drawbox=x=255:y=1482:w=570:h=142:color=black@0.32:t=fill[brandbox]",
                (
                    f"[brandbox]drawtext=fontfile={font}:text='{branding['line_1']}':"
                    "fontsize=44:fontcolor=white:borderw=3:bordercolor=black:"
                    "x=(w-text_w)/2:y=1497[b1]"
                ),
                (
                    f"[b1]drawtext=fontfile={font}:text='{branding['line_2']}':"
                    "fontsize=38:fontcolor=white:borderw=3:bordercolor=black:"
                    "x=(w-text_w)/2:y=1552[b2]"
                ),
            ]
        )
        if logo_enabled:
            parts.extend(
                [
                    f"[1:v]scale={int(branding['logo_width'])}:-1:flags=lanczos[logo]",
                    "[b2][logo]overlay=x=(W-w)/2:y=1642:format=auto[outv]",
                ]
            )
        else:
            parts.append("[b2]null[outv]")
    else:
        parts.append("[fx]null[outv]")

    parts.append("[asrc]loudnorm=I=-15.5:TP=-1.5:LRA=11[aout]")
    return ";".join(parts), output_duration


def render_candidate(
    source: Path,
    candidate: Candidate,
    config: dict[str, object],
    output: Path,
    logo: Path | None,
) -> float:
    filter_graph, output_duration = build_filter(candidate, config, logo_enabled=logo is not None)
    settings = config["output"]
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if logo is not None:
        command.extend(["-loop", "1", "-i", str(logo)])
    command.extend(
        [
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
            "aac",
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
    )
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
        "duration_10_to_20": 10.0 <= duration <= 20.0,
        "resolution": (video["width"], video["height"]) == (1080, 1920),
        "codec_h264": video["codec_name"] == "h264",
        "profile_high": str(video.get("profile", "")).lower() == "high",
        "r_frame_rate": video["r_frame_rate"] == settings["fps"],
        "avg_frame_rate": video["avg_frame_rate"] == settings["fps"],
        "cbr_near_250mbps": bitrate >= int(expected_bitrate * 0.94),
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
            "fps=2,scale=180:-1,tile=6x4:padding=2:margin=2",
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
    parser = argparse.ArgumentParser(description="Discover and render a non-duplicate MW4 gameplay batch")
    parser.add_argument("--source-key", choices=("r1", "batch2", "week2"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logo", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_candidates(args.source, args.source_key, config)
    candidate_path = args.output_dir / f"{args.source_key}_selected_candidates.json"
    candidate_path.write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2),
        encoding="utf-8",
    )

    outputs: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        filename = (
            f"MW4_V2_{args.source_key.upper()}_{index:02d}_"
            f"{candidate.effect_profile.upper()}_250M.mp4"
        )
        output = args.output_dir / filename
        planned_duration = render_candidate(args.source, candidate, config, output, args.logo)
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
        "hook_text_added": False,
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
