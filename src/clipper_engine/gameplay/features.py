from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemanticEvent:
    time: float
    kind: str
    confidence: float
    evidence: dict[str, float]


@dataclass
class SemanticTimeline:
    fps: float
    duration: float
    times: np.ndarray
    signals: dict[str, np.ndarray]
    events: list[SemanticEvent]


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 95))
    if high <= low + 1e-7:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _rolling_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size == 0:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / float(2 * radius + 1)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _rolling_max(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size == 0:
        return values.astype(np.float32, copy=True)
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    width = 2 * radius + 1
    for index in range(values.size):
        out[index] = float(np.max(padded[index : index + width]))
    return out


def _future_mean(values: np.ndarray, bins: int) -> np.ndarray:
    bins = max(1, int(bins))
    out = np.empty_like(values, dtype=np.float32)
    for index in range(values.size):
        end = min(values.size, index + bins + 1)
        out[index] = (
            float(np.mean(values[index + 1 : end])) if end > index + 1 else float(values[index])
        )
    return out


def _load_video_features(
    source: Path, fps: float, width: int, height: int
) -> dict[str, np.ndarray]:
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"fps={fps},scale={width}:{height}:flags=fast_bilinear,format=rgb24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_size = width * height * 3
    count = len(raw) // frame_size
    if count < int(fps * 3):
        raise RuntimeError("Not enough sampled frames for semantic gameplay analysis")
    frames = np.frombuffer(raw[: count * frame_size], dtype=np.uint8).reshape(
        count, height, width, 3
    )
    gray = np.mean(frames, axis=3, dtype=np.float32)
    diff = np.zeros_like(gray, dtype=np.float32)
    diff[1:] = np.abs(gray[1:] - gray[:-1]) / 255.0
    diff[0] = diff[1]

    center = diff[:, int(height * 0.20) : int(height * 0.80), int(width * 0.22) : int(width * 0.78)]
    upper_right = diff[
        :, int(height * 0.03) : int(height * 0.36), int(width * 0.68) : int(width * 0.99)
    ]
    lower_right = diff[
        :, int(height * 0.62) : int(height * 0.98), int(width * 0.66) : int(width * 0.99)
    ]
    center_hud = diff[
        :, int(height * 0.27) : int(height * 0.73), int(width * 0.30) : int(width * 0.70)
    ]

    luma = np.mean(gray, axis=(1, 2), dtype=np.float32) / 255.0
    luma_delta = np.zeros_like(luma)
    luma_delta[1:] = np.maximum(0.0, luma[1:] - luma[:-1])

    rgb16 = frames.astype(np.int16)
    red_excess = np.maximum(rgb16[..., 0] - ((rgb16[..., 1] + rgb16[..., 2]) // 2), 0)
    red_signal = (
        np.mean(
            red_excess[
                :, int(height * 0.08) : int(height * 0.92), int(width * 0.08) : int(width * 0.92)
            ],
            axis=(1, 2),
            dtype=np.float32,
        )
        / 255.0
    )

    return {
        "global_motion": np.mean(diff, axis=(1, 2), dtype=np.float32),
        "center_motion": np.mean(center, axis=(1, 2), dtype=np.float32),
        "hud_upper_change": np.mean(upper_right, axis=(1, 2), dtype=np.float32),
        "hud_lower_change": np.mean(lower_right, axis=(1, 2), dtype=np.float32),
        "center_change": np.mean(center_hud, axis=(1, 2), dtype=np.float32),
        "luma_delta": luma_delta,
        "red_signal": red_signal,
    }


def _load_audio_features(source: Path, fps: float, sample_rate: int) -> dict[str, np.ndarray]:
    raw = subprocess.check_output(
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
            str(sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
    )
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    per_bin = max(1, round(sample_rate / fps))
    count = len(audio) // per_bin
    if count < int(fps * 3):
        raise RuntimeError("Not enough sampled audio for semantic gameplay analysis")
    audio = audio[: count * per_bin].reshape(count, per_bin)
    rms = np.sqrt(np.mean(np.square(audio), axis=1, dtype=np.float32))
    high_proxy = np.sqrt(np.mean(np.square(np.diff(audio, axis=1)), axis=1, dtype=np.float32))
    transient = np.zeros_like(rms)
    transient[1:] = np.maximum(0.0, rms[1:] - rms[:-1])
    return {"audio_rms": rms, "audio_high_proxy": high_proxy, "audio_transient": transient}


def _local_maxima(signal: np.ndarray, threshold: float, distance: int) -> list[int]:
    order = np.argsort(signal)[::-1]
    chosen: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        if float(signal[index]) < threshold:
            break
        left, right = max(0, index - 1), min(signal.size, index + 2)
        if float(signal[index]) < float(np.max(signal[left:right])):
            continue
        if all(abs(index - prior) >= distance for prior in chosen):
            chosen.append(index)
    chosen.sort()
    return chosen


def _event(
    index: int,
    fps: float,
    kind: str,
    signal: str,
    signals: dict[str, np.ndarray],
    evidence: tuple[str, ...],
) -> SemanticEvent:
    return SemanticEvent(
        time=round((index + 0.5) / fps, 3),
        kind=kind,
        confidence=round(float(signals[signal][index]), 4),
        evidence={name: round(float(signals[name][index]), 4) for name in evidence},
    )


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimeline:
    cfg = config.get("semantic_analysis", {})
    fps = float(cfg.get("sample_fps", 6.0))
    video = _load_video_features(
        source, fps, int(cfg.get("sample_width", 256)), int(cfg.get("sample_height", 144))
    )
    audio = _load_audio_features(source, fps, int(cfg.get("audio_sample_rate", 12000)))
    length = min(
        [len(values) for values in video.values()] + [len(values) for values in audio.values()]
    )
    signals = {
        key: _robust_normalize(values[:length]) for key, values in {**video, **audio}.items()
    }

    gm, cm = signals["global_motion"], signals["center_motion"]
    hud = np.maximum(signals["hud_upper_change"], signals["center_change"])
    lower_hud = signals["hud_lower_change"]
    audio_rms, audio_tr, audio_hi = (
        signals["audio_rms"],
        signals["audio_transient"],
        signals["audio_high_proxy"],
    )
    flash, red = signals["luma_delta"], signals["red_signal"]

    combat = np.clip(
        0.27 * audio_rms
        + 0.23 * cm
        + 0.17 * audio_tr
        + 0.13 * audio_hi
        + 0.10 * hud
        + 0.06 * flash
        + 0.04 * red,
        0,
        1,
    )
    recent = _rolling_max(combat, max(1, round(0.55 * fps)))
    post_drop = np.clip(recent - _future_mean(combat, max(1, round(0.65 * fps))), 0, 1)
    outcome = np.clip(
        0.34 * hud + 0.27 * recent + 0.18 * post_drop + 0.11 * cm + 0.10 * audio_tr, 0, 1
    )
    outcome *= np.clip((recent - 0.22) / 0.58, 0, 1)
    impact = np.clip(
        0.34 * flash + 0.29 * audio_tr + 0.18 * audio_rms + 0.11 * cm + 0.08 * red, 0, 1
    )
    contact = np.clip(0.39 * cm + 0.27 * combat + 0.13 * hud + 0.11 * red + 0.10 * audio_rms, 0, 1)
    traversal = np.clip(
        0.66 * gm + 0.14 * cm - 0.42 * combat - 0.18 * outcome - 0.10 * impact, 0, 1
    )
    recovery = np.clip(
        0.33 * lower_hud + 0.22 * gm + 0.24 * recent - 0.28 * combat - 0.12 * outcome, 0, 1
    )
    pressure = np.maximum.reduce([combat, outcome, impact, contact])
    interest = np.clip(
        0.31 * combat
        + 0.18 * contact
        + 0.17 * outcome
        + 0.12 * impact
        + 0.08 * hud
        + 0.07 * audio_rms
        + 0.07 * recent
        - 0.14 * np.maximum(0, traversal - 0.45)
        - 0.08 * np.maximum(0, recovery - 0.55),
        0,
        1,
    )
    interest = _rolling_mean(np.maximum(interest, 0.74 * pressure), max(1, round(0.16 * fps)))
    dull = (
        (interest < float(cfg.get("dull_interest_threshold", 0.28)))
        & (combat < 0.36)
        & (outcome < 0.35)
        & (impact < 0.40)
        & (contact < 0.38)
    ).astype(np.float32)
    signals.update(
        {
            "hud_change": hud.astype(np.float32),
            "combat": combat.astype(np.float32),
            "outcome": outcome.astype(np.float32),
            "impact": impact.astype(np.float32),
            "contact": contact.astype(np.float32),
            "traversal": traversal.astype(np.float32),
            "recovery": recovery.astype(np.float32),
            "interest": interest.astype(np.float32),
            "dull": dull,
        }
    )

    thresholds = cfg.get("event_thresholds", {})
    specs = (
        (
            "combat_burst",
            "combat",
            float(thresholds.get("combat_burst", 0.62)),
            0.75,
            ("combat", "audio_rms", "center_motion", "audio_transient"),
        ),
        (
            "outcome_like",
            "outcome",
            float(thresholds.get("outcome_like", 0.58)),
            0.90,
            ("outcome", "hud_change", "combat", "center_motion"),
        ),
        (
            "impact",
            "impact",
            float(thresholds.get("impact", 0.68)),
            0.70,
            ("impact", "luma_delta", "audio_transient", "audio_rms"),
        ),
        (
            "contact",
            "contact",
            float(thresholds.get("contact", 0.60)),
            0.90,
            ("contact", "center_motion", "combat", "hud_change"),
        ),
    )
    events: list[SemanticEvent] = []
    for kind, signal, threshold, spacing, evidence in specs:
        for index in _local_maxima(signals[signal], threshold, max(1, round(spacing * fps))):
            events.append(_event(index, fps, kind, signal, signals, evidence))
    events.sort(key=lambda item: (item.time, item.kind))
    times = (np.arange(length, dtype=np.float32) + 0.5) / fps
    return SemanticTimeline(
        fps=fps, duration=length / fps, times=times, signals=signals, events=events
    )


