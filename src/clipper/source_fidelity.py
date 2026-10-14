"""Original-source-driven visual fidelity for vertical TJR editorial masters.

A source video's advertised bitrate cannot alone determine its perceived
quality, especially across AV1/VP9/H.264. Use it to choose only the FIRST
encoding attempt, then compare the decoded master against the exact same
uncompressed editorial composition (including Style B captions) with SSIM.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from array import array
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any


class FidelityError(RuntimeError):
    """Source or output quality could not be verified."""


@dataclass(frozen=True, slots=True)
class SourceProfile:
    width: int
    height: int
    fps: str
    codec: str
    pix_fmt: str
    video_bitrate: int | None
    audio_bitrate: int | None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def bits_per_pixel_frame(self) -> float | None:
        if not self.video_bitrate:
            return None
        return self.video_bitrate / (self.width * self.height * float(Fraction(self.fps)))


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (ValueError, TypeError):
        return None
    return parsed if parsed > 0 else None


def parse_source_profile(info: dict[str, Any]) -> SourceProfile:
    """Fail closed rather than silently assuming an HD format or 30/60 fps."""
    try:
        streams: list[dict[str, Any]] = info["streams"]
        video = next(item for item in streams if item["codec_type"] == "video")
        audio = next(item for item in streams if item["codec_type"] == "audio")
        width, height = int(video["width"]), int(video["height"])
        rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "")
        fps = float(Fraction(rate))
        transfer = str(video.get("color_transfer") or "").lower()
        if width < 720 or height < 720 or not 23 <= fps <= 61:
            raise ValueError("source below HD or has an unsupported native cadence")
        if transfer in {"smpte2084", "arib-std-b67"}:
            raise ValueError("HDR source requires explicit tone mapping before SDR delivery")
        codec = str(video["codec_name"])
        pix_fmt = str(video["pix_fmt"])
        if not codec or not pix_fmt or codec == "none":
            raise ValueError("source codec and pixel format must be identifiable")
        video_bitrate = _positive_int(video.get("bit_rate"))
        audio_bitrate = _positive_int(audio.get("bit_rate"))
        # Container bitrate can include audio and metadata. An explicitly
        # estimated bitrate is useful for the initial try, never acceptance.
        if video_bitrate is None:
            overall = _positive_int(info.get("format", {}).get("bit_rate"))
            if overall and audio_bitrate and overall > audio_bitrate:
                video_bitrate = overall - audio_bitrate
        return SourceProfile(
            width=width,
            height=height,
            fps=rate,
            codec=codec,
            pix_fmt=pix_fmt,
            video_bitrate=video_bitrate,
            audio_bitrate=audio_bitrate,
            color_space=str(video.get("color_space") or "") or None,
            color_transfer=transfer or None,
            color_primaries=str(video.get("color_primaries") or "") or None,
        )
    except (KeyError, TypeError, ValueError, StopIteration, ZeroDivisionError) as exc:
        raise FidelityError(f"unverifiable original source profile: {exc}") from exc


def probe_source_profile(path: Path) -> SourceProfile:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=50,
        )
        parsed: dict[str, Any] = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise FidelityError("cannot independently probe the actual original video") from exc
    return parse_source_profile(parsed)


def crf_attempts(profile: SourceProfile) -> tuple[int, ...]:
    """Adapt starting point to actual source complexity; SSIM decides success.

    CRF/bitrate equivalence across codecs is NOT assumed. The measured
    comparison is to the same transformed original, before H.264 encoding.
    """
    bpp = profile.bits_per_pixel_frame
    first = 16 if bpp is not None and bpp >= 0.10 else 18
    return tuple(range(first, 11, -2))


def compare_encoded_to_composition(
    command: list[str],
    *,
    source: Path,
    output: Path,
    clip_start: float,
    duration: float,
    fps: str,
    stats_path: Path,
) -> tuple[float, int]:
    """Reconstruct the EXACT pre-encode composition and compare real frames.

    Using the same FFmpeg filter graph, ASS captions and crop means that
    intentional TikTok styling is not incorrectly counted as quality loss.
    """
    if "-filter_complex" not in command:
        raise FidelityError("cannot compare unrecorded editorial composition")
    graph = command[command.index("-filter_complex") + 1]
    if not graph.endswith("format=yuv420p,setsar=1[v]"):
        raise FidelityError("unexpected filter graph: fidelity reference would differ")
    stats_path.unlink(missing_ok=True)
    # Preserve every composition input (including its watermark) and append
    # the encoded output afterward. Replacing input 1 with the output makes
    # a watermarked graph accidentally scale the whole output as its logo.
    inputs = command[: command.index("-filter_complex")]
    input_count = inputs.count("-i")
    if input_count:
        if inputs[inputs.index("-i") + 1] != str(source):
            raise FidelityError("composition source does not match the verified original")
    else:
        inputs = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{clip_start:.3f}",
            "-i",
            str(source),
        ]
        input_count = 1
    # Compare corresponding native-cadence frames on a canonical clock.
    # Container timestamp quantization must not select a neighboring frame.
    if float(Fraction(fps)) <= 0:
        raise FidelityError("comparison requires a positive native frame rate")
    clock = f"trim=start=0,settb=AVTB,setpts=N/({fps})/TB"
    reference = (
        f"{graph};[v]{clock}[reference];[{input_count}:v]{clock}[encoded];"
        f"[reference][encoded]ssim=stats_file='{stats_path.as_posix()}'[verified]"
    )
    try:
        subprocess.run(
            [
                *inputs,
                "-i",
                str(output),
                "-filter_complex",
                reference,
                "-map",
                "[verified]",
                "-t",
                f"{duration:.3f}",
                "-an",
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=480,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = str(getattr(exc, "stderr", "") or str(exc))[-700:]
        raise FidelityError(f"source-to-delivery SSIM comparison failed: {detail}") from exc
    if not stats_path.is_file():
        raise FidelityError("SSIM did not produce frame-level evidence")
    scores: list[float] = []
    for line in stats_path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"(?<!\w)All:([0-9.]+)", line)
        if match:
            scores.append(float(match.group(1)))
    expected_frames = duration * float(Fraction(fps))
    if len(scores) < max(1, math.floor(expected_frames * 0.94)):
        raise FidelityError(f"insufficient comparable frames: {len(scores)}/{expected_frames:.0f}")
    if not scores or any(not math.isfinite(value) for value in scores):
        raise FidelityError("invalid frame-level SSIM statistics")
    return round(sum(scores) / len(scores), 6), len(scores)


def measure_audio_gain(source: Path, *, start: float, duration: float) -> float:
    """Measure once, then use constant gain with a measured true-peak ceiling."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-vn",
            "-af",
            f"atrim=duration={duration},asetpts=PTS-STARTPTS,"
            "loudnorm=I=-14:LRA=11:TP=-1.5:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.DOTALL)
    if match is None:
        raise FidelityError("audio loudness measurement is missing")
    values = json.loads(match.group())
    integrated, peak = float(values["input_i"]), float(values["input_tp"])
    if not math.isfinite(integrated) or not math.isfinite(peak):
        raise FidelityError("source clip has no measurable audio")
    return round(min(-14 - integrated, -1.5 - peak), 6)


def _audio_pcm(path: Path, *, start: float, duration: float) -> array[float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    samples: array[float] = array("f")
    samples.frombytes(result.stdout)
    if not samples or any(not math.isfinite(value) for value in samples):
        raise FidelityError("source/output audio decode is empty or invalid")
    return samples


def compare_audio_to_source(
    source: Path, output: Path, *, start: float, duration: float, enforce: bool = True
) -> dict[str, Any]:
    """Check signal continuity against the source; natural pauses must survive."""
    reference = _audio_pcm(source, start=start, duration=duration)
    delivered = _audio_pcm(output, start=0, duration=duration)
    count = min(len(reference), len(delivered))

    def correlation(lag: int) -> float:
        lo, hi = max(0, -lag), min(count, len(delivered) - lag)
        a = b = cross = 0.0
        for i in range(lo, hi, 32):
            x, y = reference[i], delivered[i + lag]
            a += x * x
            b += y * y
            cross += x * y
        return cross / math.sqrt(a * b) if a > 0 and b > 0 else 0.0

    lag = max(range(-480, 481, 160), key=correlation)
    lag = max(range(lag - 160, lag + 161, 8), key=correlation)
    lag = max(range(lag - 8, lag + 9), key=correlation)
    similarity = correlation(lag)
    # Ten-millisecond windows catch short cutoffs without treating ordinary
    # breathing, codec edge ringing or original source silence as missing speech.
    window = 160
    energies: list[tuple[float, float, float]] = []
    for i in range(max(0, -lag), min(count, len(delivered) - lag) - window + 1, window):
        source_rms = math.sqrt(sum(value * value for value in reference[i : i + window]) / window)
        output_rms = math.sqrt(
            sum(value * value for value in delivered[i + lag : i + lag + window]) / window
        )
        energies.append((i / 16000, source_rms, output_rms))
    active_floor = max(0.0001, max((item[1] for item in energies), default=0) * 0.01)
    ratios = [out / ref for _, ref, out in energies if ref > active_floor and out > 0]
    gain = median(ratios) if ratios else 0
    missing = [
        timestamp
        for timestamp, ref, out in energies
        if ref > active_floor and out < ref * gain / 16
    ]
    spans: list[dict[str, float]] = []
    for timestamp in missing:
        if spans and timestamp - spans[-1]["end"] < 0.011:
            spans[-1]["end"] = round(timestamp + 0.01, 3)
        else:
            spans.append({"start": round(timestamp, 3), "end": round(timestamp + 0.01, 3)})
    spans = [span for span in spans if span["end"] - span["start"] >= 0.019]
    passed = (
        similarity >= 0.95
        and not spans
        and abs(len(delivered) / 16000 - duration) <= 0.05
        and abs(len(reference) / 16000 - duration) <= 0.05
    )
    report = {
        "status": "SOURCE_AUDIO_MATCHED" if passed else "SOURCE_AUDIO_MISMATCH",
        "method": "decoded_source_waveform_and_10ms_energy_windows",
        "waveform_correlation": round(similarity, 6),
        "alignment_offset_ms": lag / 16,
        "source_duration_seconds": len(reference) / 16000,
        "output_duration_seconds": len(delivered) / 16000,
        "introduced_dropout_spans": spans,
        "source_silent_windows": sum(ref < active_floor for _, ref, _ in energies),
        "comparison_windows": len(energies),
    }
    if enforce and not passed:
        raise FidelityError("rendered audio failed source continuity: " + json.dumps(report))
    return report
