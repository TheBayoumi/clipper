from __future__ import annotations

import itertools
import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

UNKNOWN_METADATA = {"", "unknown", "unspecified", "reserved", "n/a", "N/A", "0:1"}
VIDEO_METADATA_FIELDS = (
    "pix_fmt",
    "sample_aspect_ratio",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "chroma_location",
    "field_order",
)


def _command_text(command: Iterable[str]) -> str:
    return " ".join(str(item) for item in command)


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr[-6000:].strip()
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {_command_text(command)}"
            + (f"\nstderr:\n{stderr}" if stderr else "")
        )
    return completed


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr[-6000:].strip()
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {_command_text(command)}"
            + (f"\nstderr:\n{stderr}" if stderr else "")
        )


def _fraction(value: Any, *, label: str) -> Fraction:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        raise RuntimeError(f"{label} is unavailable: {value!r}")
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"invalid {label}: {value!r}") from exc
    if result <= 0:
        raise RuntimeError(f"{label} must be positive: {value!r}")
    return result


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class VideoTimingContract:
    time_base: Fraction
    nominal_rate: Fraction
    ticks_per_frame: int
    track_timescale: int
    source_avg_frame_rate: str

    def to_json(self) -> dict[str, Any]:
        return {
            "time_base": fraction_text(self.time_base),
            "nominal_rate": fraction_text(self.nominal_rate),
            "ticks_per_frame": self.ticks_per_frame,
            "track_timescale": self.track_timescale,
            "source_avg_frame_rate": self.source_avg_frame_rate,
        }


@dataclass(frozen=True)
class VideoMediaContract:
    width: int
    height: int
    codec_name: str
    profile: str
    pix_fmt: str
    sample_aspect_ratio: str
    display_aspect_ratio: str
    color_range: str
    color_space: str
    color_transfer: str
    color_primaries: str
    chroma_location: str
    field_order: str
    bits_per_raw_sample: str
    timing: VideoTimingContract

    def profile_dict(self) -> dict[str, Any]:
        return {
            "codec_name": self.codec_name,
            "profile": self.profile,
            "width": self.width,
            "height": self.height,
            "pix_fmt": self.pix_fmt,
            "sample_aspect_ratio": self.sample_aspect_ratio,
            "display_aspect_ratio": self.display_aspect_ratio,
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "chroma_location": self.chroma_location,
            "field_order": self.field_order,
            "bits_per_raw_sample": self.bits_per_raw_sample,
            "r_frame_rate": fraction_text(self.timing.nominal_rate),
            "avg_frame_rate": self.timing.source_avg_frame_rate,
            "time_base": fraction_text(self.timing.time_base),
            "track_timescale": self.timing.track_timescale,
            "ticks_per_frame": self.timing.ticks_per_frame,
        }


@dataclass(frozen=True)
class AudioMediaContract:
    codec_name: str
    sample_rate: int
    channels: int
    channel_layout: str
    sample_fmt: str


@dataclass(frozen=True)
class SourceMediaContract:
    video: VideoMediaContract
    audio: AudioMediaContract

    def to_json(self) -> dict[str, Any]:
        return {
            "video": {**self.video.profile_dict(), "timing": self.video.timing.to_json()},
            "audio": asdict(self.audio),
        }


def video_profile(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    entries = (
        "stream=codec_name,profile,width,height,pix_fmt,sample_aspect_ratio,display_aspect_ratio,"
        "color_range,color_space,color_transfer,color_primaries,chroma_location,field_order,"
        "bits_per_raw_sample,r_frame_rate,avg_frame_rate,time_base"
    )
    if count_frames:
        entries += ",nb_read_frames"
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command += ["-show_entries", entries, "-of", "json", str(path)]
    payload = json.loads(run_capture(command).stdout)
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
        "time_base": str(stream.get("time_base") or ""),
    }
    if count_frames:
        count = stream.get("nb_read_frames")
        if count in (None, "N/A"):
            raise RuntimeError(f"ffprobe could not count decoded video frames for {path}")
        profile["frame_count"] = int(count)
    return profile


def audio_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(
        run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_fmt,sample_rate,channels,channel_layout",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"source-native pipeline requires exactly one audio stream: {path}")
    stream = streams[0]
    return {
        "codec_name": str(stream.get("codec_name") or ""),
        "sample_fmt": str(stream.get("sample_fmt") or ""),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": str(stream.get("channel_layout") or ""),
    }


def packet_pts(path: Path) -> list[int]:
    text = run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts",
            "-of",
            "csv=p=0",
            str(path),
        ]
    ).stdout
    pts: list[int] = []
    for raw in text.splitlines():
        token = raw.strip().split(",", 1)[0].strip()
        if token in {"", "N/A"}:
            continue
        try:
            pts.append(int(token))
        except ValueError as exc:
            raise RuntimeError(f"invalid video packet PTS {token!r} in {path}") from exc
    if not pts:
        raise RuntimeError(f"no video packet PTS available: {path}")
    if len(set(pts)) != len(pts):
        raise RuntimeError(f"duplicate video presentation timestamps detected: {path}")
    return sorted(pts)


def timing_from_profile(profile: dict[str, Any]) -> VideoTimingContract:
    time_base = _fraction(profile.get("time_base"), label="video time_base")
    nominal_rate = _fraction(profile.get("r_frame_rate"), label="video r_frame_rate")
    timescale = Fraction(1, 1) / time_base
    if timescale.denominator != 1:
        raise RuntimeError(
            "source video time_base cannot be represented exactly by an integer MOV track timescale: "
            f"time_base={fraction_text(time_base)}"
        )
    ticks_per_frame = timescale / nominal_rate
    if ticks_per_frame.denominator != 1 or ticks_per_frame <= 0:
        raise RuntimeError(
            "source nominal frame cadence is not exactly representable on its own time base: "
            f"time_base={fraction_text(time_base)} rate={fraction_text(nominal_rate)} "
            f"ticks_per_frame={ticks_per_frame}"
        )
    return VideoTimingContract(
        time_base=time_base,
        nominal_rate=nominal_rate,
        ticks_per_frame=int(ticks_per_frame),
        track_timescale=int(timescale),
        source_avg_frame_rate=str(profile.get("avg_frame_rate") or ""),
    )


def verify_cfr_timeline(path: Path, timing: VideoTimingContract, *, label: str) -> dict[str, Any]:
    profile = video_profile(path)
    actual_tb = _fraction(profile["time_base"], label=f"{label} time_base")
    actual_rate = _fraction(profile["r_frame_rate"], label=f"{label} r_frame_rate")
    pts = packet_pts(path)
    deltas = [right - left for left, right in itertools.pairwise(pts)]
    bad = [(index, value) for index, value in enumerate(deltas) if value != timing.ticks_per_frame]
    checks = {
        "time_base_matches_source": actual_tb == timing.time_base,
        "r_frame_rate_matches_source": actual_rate == timing.nominal_rate,
        "packet_pts_strictly_cfr": not bad,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{label} timing contract failed: checks={checks}; "
            f"source_time_base={fraction_text(timing.time_base)} "
            f"actual_time_base={fraction_text(actual_tb)} "
            f"source_rate={fraction_text(timing.nominal_rate)} "
            f"actual_rate={fraction_text(actual_rate)} "
            f"expected_pts_step={timing.ticks_per_frame} bad_pts_deltas={bad[:8]}"
        )
    return {
        "checks": checks,
        "packet_count": len(pts),
        "first_pts": pts[0],
        "last_pts": pts[-1],
        "pts_step": timing.ticks_per_frame,
        "avg_frame_rate_diagnostic": profile.get("avg_frame_rate"),
    }


def verify_monotonic_timeline(
    path: Path, timing: VideoTimingContract, *, label: str
) -> dict[str, Any]:
    profile = video_profile(path)
    actual_tb = _fraction(profile["time_base"], label=f"{label} time_base")
    actual_rate = _fraction(profile["r_frame_rate"], label=f"{label} r_frame_rate")
    pts = packet_pts(path)
    deltas = [right - left for left, right in itertools.pairwise(pts)]
    nonpositive = [(index, value) for index, value in enumerate(deltas) if value <= 0]
    checks = {
        "time_base_matches_source": actual_tb == timing.time_base,
        "r_frame_rate_matches_source": actual_rate == timing.nominal_rate,
        "packet_pts_strictly_monotonic": not nonpositive,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{label} transport timing contract failed: checks={checks}; "
            f"source_time_base={fraction_text(timing.time_base)} "
            f"actual_time_base={fraction_text(actual_tb)} "
            f"source_rate={fraction_text(timing.nominal_rate)} "
            f"actual_rate={fraction_text(actual_rate)} "
            f"nonpositive_pts_deltas={nonpositive[:8]}"
        )
    return {
        "checks": checks,
        "packet_count": len(pts),
        "first_pts": pts[0],
        "last_pts": pts[-1],
        "avg_frame_rate_diagnostic": profile.get("avg_frame_rate"),
    }


def inspect_source(path: Path, *, verify_timeline: bool = True) -> SourceMediaContract:
    video = video_profile(path)
    audio = audio_profile(path)
    if video["width"] <= 0 or video["height"] <= 0 or not video["pix_fmt"]:
        raise RuntimeError(f"source video profile is incomplete: {video}")
    if audio["sample_rate"] <= 0 or audio["channels"] <= 0:
        raise RuntimeError(f"source audio profile is incomplete: {audio}")
    timing = timing_from_profile(video)
    contract = SourceMediaContract(
        video=VideoMediaContract(
            width=int(video["width"]),
            height=int(video["height"]),
            codec_name=str(video["codec_name"]),
            profile=str(video["profile"]),
            pix_fmt=str(video["pix_fmt"]),
            sample_aspect_ratio=str(video["sample_aspect_ratio"]),
            display_aspect_ratio=str(video["display_aspect_ratio"]),
            color_range=str(video["color_range"]),
            color_space=str(video["color_space"]),
            color_transfer=str(video["color_transfer"]),
            color_primaries=str(video["color_primaries"]),
            chroma_location=str(video["chroma_location"]),
            field_order=str(video["field_order"]),
            bits_per_raw_sample=str(video["bits_per_raw_sample"]),
            timing=timing,
        ),
        audio=AudioMediaContract(
            codec_name=str(audio["codec_name"]),
            sample_rate=int(audio["sample_rate"]),
            channels=int(audio["channels"]),
            channel_layout=str(audio["channel_layout"]),
            sample_fmt=str(audio["sample_fmt"]),
        ),
    )
    if verify_timeline:
        verify_cfr_timeline(path, timing, label="input source")
    return contract


def is_known(value: Any) -> bool:
    return str(value or "").strip() not in UNKNOWN_METADATA


def metadata_match_checks(
    source: dict[str, Any], other: dict[str, Any], *, prefix: str
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for field in VIDEO_METADATA_FIELDS:
        left = str(source.get(field) or "")
        right = str(other.get(field) or "")
        ok = right == left if is_known(left) else not is_known(right)
        checks[f"{prefix}_{field}_matches_source"] = ok
    return checks


def profile_output_args(profile: dict[str, Any]) -> list[str]:
    if not profile.get("pix_fmt"):
        raise RuntimeError("source pixel format is unavailable")
    args = ["-pix_fmt", str(profile["pix_fmt"])]
    mapping = (
        ("color_range", "-color_range"),
        ("color_space", "-colorspace"),
        ("color_transfer", "-color_trc"),
        ("color_primaries", "-color_primaries"),
        ("chroma_location", "-chroma_sample_location"),
    )
    for field, option in mapping:
        value = str(profile.get(field) or "")
        if is_known(value):
            args += [option, value]
    return args


_CHROMA_X264 = {
    "left": "0",
    "center": "1",
    "topleft": "2",
    "top-left": "2",
    "top": "3",
    "bottomleft": "4",
    "bottom-left": "4",
    "bottom": "5",
}


def x264_vui_params(profile: dict[str, Any]) -> list[str]:
    params: list[str] = []
    color_range = str(profile.get("color_range") or "")
    if color_range == "tv":
        params.append("fullrange=off")
    elif color_range == "pc":
        params.append("fullrange=on")
    for field, key in (
        ("color_primaries", "colorprim"),
        ("color_transfer", "transfer"),
        ("color_space", "colormatrix"),
    ):
        value = str(profile.get(field) or "")
        if is_known(value):
            params.append(f"{key}={value}")
    chroma = str(profile.get("chroma_location") or "").lower()
    if chroma in _CHROMA_X264:
        params.append(f"chromaloc={_CHROMA_X264[chroma]}")
    return params


def lossless_video_args(profile: dict[str, Any]) -> list[str]:
    timing = timing_from_profile(profile)
    vui = x264_vui_params(profile)
    args = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-qp",
        "0",
        *profile_output_args(profile),
    ]
    if vui:
        args += ["-x264-params", ":".join(vui)]
    args += [
        "-video_track_timescale",
        str(timing.track_timescale),
        "-threads:v",
        "4",
    ]
    return args


def merge_x264_params(args: list[str], extra_params: list[str]) -> list[str]:
    if not extra_params:
        return list(args)
    result = list(args)
    if "-x264-params" in result:
        index = result.index("-x264-params")
        if index + 1 >= len(result):
            raise RuntimeError("malformed encoder args: -x264-params has no value")
        current = str(result[index + 1]).strip(":")
        merged = [item for item in current.split(":") if item] + list(extra_params)
        unique: list[str] = []
        keys: set[str] = set()
        for item in merged:
            key = item.split("=", 1)[0]
            if key in keys:
                continue
            keys.add(key)
            unique.append(item)
        result[index + 1] = ":".join(unique)
    else:
        result += ["-x264-params", ":".join(extra_params)]
    return result


def validate_delivery_compatibility(contract: SourceMediaContract, config: dict[str, Any]) -> None:
    settings = config.get("output", {})
    failures: list[str] = []
    if settings.get("full_source_frame") is not True:
        failures.append("output.full_source_frame must be true in source-native mode")
    width = settings.get("width")
    height = settings.get("height")
    if width not in (None, "source", "auto") and int(width) != contract.video.width:
        failures.append(
            f"delivery width={width} conflicts with source width={contract.video.width}"
        )
    if height not in (None, "source", "auto") and int(height) != contract.video.height:
        failures.append(
            f"delivery height={height} conflicts with source height={contract.video.height}"
        )
    configured_fps = settings.get("fps")
    if configured_fps not in (None, "source", "auto"):
        try:
            wanted = Fraction(str(configured_fps))
        except (ValueError, ZeroDivisionError) as exc:
            raise RuntimeError(
                f"invalid output.fps delivery constraint: {configured_fps!r}"
            ) from exc
        if wanted != contract.video.timing.nominal_rate:
            failures.append(
                "delivery fps conflicts with source-native timing: "
                f"delivery={configured_fps} source={fraction_text(contract.video.timing.nominal_rate)}"
            )
    if failures:
        raise RuntimeError(
            "source-native delivery contract is incompatible: " + "; ".join(failures)
        )


def frame_hashes(
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
    text = run_capture(command).stdout
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
