from __future__ import annotations

import json
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


@dataclass(frozen=True)
class EditSegment:
    start: float
    end: float
    speed: float
    reason: str


@dataclass(frozen=True)
class SemanticPlan:
    start: float
    end: float
    raw_duration: float
    output_duration: float
    score: float
    retention_proxy: float
    engagement_proxy: float
    mean_interest: float
    opening_interest: float
    closing_interest: float
    weakest_quarter_interest: float
    low_interest_fraction: float
    max_unexplained_low_interest_run_seconds: float
    story_type: str
    effect_profile: str
    cold_open_source_start: float | None
    segments: tuple[EditSegment, ...]
    semantic_events: tuple[SemanticEvent, ...]
    effect_events: tuple[SemanticEvent, ...]
    editorial_reasons: tuple[str, ...]


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
        out[index] = float(np.max(padded[index:index + width]))
    return out


def _future_mean(values: np.ndarray, bins: int) -> np.ndarray:
    bins = max(1, int(bins))
    out = np.empty_like(values, dtype=np.float32)
    for index in range(values.size):
        end = min(values.size, index + bins + 1)
        out[index] = (
            float(np.mean(values[index + 1:end]))
            if end > index + 1
            else float(values[index])
        )
    return out


def _load_video_features(source: Path, fps: float, width: int, height: int) -> dict[str, np.ndarray]:
    raw = subprocess.check_output(
        [
            "ffmpeg", "-v", "error", "-i", str(source),
            "-vf", f"fps={fps},scale={width}:{height}:flags=fast_bilinear,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
    )
    frame_size = width * height * 3
    count = len(raw) // frame_size
    if count < int(fps * 3):
        raise RuntimeError("Not enough sampled frames for semantic gameplay analysis")
    frames = np.frombuffer(raw[:count * frame_size], dtype=np.uint8).reshape(count, height, width, 3)
    gray = np.mean(frames, axis=3, dtype=np.float32)
    diff = np.zeros_like(gray, dtype=np.float32)
    diff[1:] = np.abs(gray[1:] - gray[:-1]) / 255.0
    diff[0] = diff[1]

    center = diff[:, int(height * .20):int(height * .80), int(width * .22):int(width * .78)]
    upper_right = diff[:, int(height * .03):int(height * .36), int(width * .68):int(width * .99)]
    lower_right = diff[:, int(height * .62):int(height * .98), int(width * .66):int(width * .99)]
    center_hud = diff[:, int(height * .27):int(height * .73), int(width * .30):int(width * .70)]

    luma = np.mean(gray, axis=(1, 2), dtype=np.float32) / 255.0
    luma_delta = np.zeros_like(luma)
    luma_delta[1:] = np.maximum(0.0, luma[1:] - luma[:-1])

    rgb16 = frames.astype(np.int16)
    red_excess = np.maximum(rgb16[..., 0] - ((rgb16[..., 1] + rgb16[..., 2]) // 2), 0)
    red_signal = np.mean(
        red_excess[:, int(height * .08):int(height * .92), int(width * .08):int(width * .92)],
        axis=(1, 2), dtype=np.float32,
    ) / 255.0

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
            "ffmpeg", "-v", "error", "-i", str(source), "-vn", "-ac", "1",
            "-ar", str(sample_rate), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ]
    )
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    per_bin = max(1, int(round(sample_rate / fps)))
    count = len(audio) // per_bin
    if count < int(fps * 3):
        raise RuntimeError("Not enough sampled audio for semantic gameplay analysis")
    audio = audio[:count * per_bin].reshape(count, per_bin)
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


def _event(index: int, fps: float, kind: str, signal: str, signals: dict[str, np.ndarray], evidence: tuple[str, ...]) -> SemanticEvent:
    return SemanticEvent(
        time=round((index + .5) / fps, 3),
        kind=kind,
        confidence=round(float(signals[signal][index]), 4),
        evidence={name: round(float(signals[name][index]), 4) for name in evidence},
    )


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimeline:
    cfg = config.get("semantic_analysis", {})
    fps = float(cfg.get("sample_fps", 6.0))
    video = _load_video_features(source, fps, int(cfg.get("sample_width", 256)), int(cfg.get("sample_height", 144)))
    audio = _load_audio_features(source, fps, int(cfg.get("audio_sample_rate", 12000)))
    length = min([len(values) for values in video.values()] + [len(values) for values in audio.values()])
    signals = {key: _robust_normalize(values[:length]) for key, values in {**video, **audio}.items()}

    gm, cm = signals["global_motion"], signals["center_motion"]
    hud = np.maximum(signals["hud_upper_change"], signals["center_change"])
    lower_hud = signals["hud_lower_change"]
    audio_rms, audio_tr, audio_hi = signals["audio_rms"], signals["audio_transient"], signals["audio_high_proxy"]
    flash, red = signals["luma_delta"], signals["red_signal"]

    combat = np.clip(.27*audio_rms + .23*cm + .17*audio_tr + .13*audio_hi + .10*hud + .06*flash + .04*red, 0, 1)
    recent = _rolling_max(combat, max(1, int(round(.55 * fps))))
    post_drop = np.clip(recent - _future_mean(combat, max(1, int(round(.65 * fps)))), 0, 1)
    outcome = np.clip(.34*hud + .27*recent + .18*post_drop + .11*cm + .10*audio_tr, 0, 1)
    outcome *= np.clip((recent - .22) / .58, 0, 1)
    impact = np.clip(.34*flash + .29*audio_tr + .18*audio_rms + .11*cm + .08*red, 0, 1)
    contact = np.clip(.39*cm + .27*combat + .13*hud + .11*red + .10*audio_rms, 0, 1)
    traversal = np.clip(.66*gm + .14*cm - .42*combat - .18*outcome - .10*impact, 0, 1)
    recovery = np.clip(.33*lower_hud + .22*gm + .24*recent - .28*combat - .12*outcome, 0, 1)
    pressure = np.maximum.reduce([combat, outcome, impact, contact])
    interest = np.clip(
        .31*combat + .18*contact + .17*outcome + .12*impact + .08*hud + .07*audio_rms + .07*recent
        - .14*np.maximum(0, traversal - .45) - .08*np.maximum(0, recovery - .55), 0, 1,
    )
    interest = _rolling_mean(np.maximum(interest, .74*pressure), max(1, int(round(.16 * fps))))
    dull = ((interest < float(cfg.get("dull_interest_threshold", .28))) & (combat < .36) & (outcome < .35) & (impact < .40) & (contact < .38)).astype(np.float32)
    signals.update({
        "hud_change": hud.astype(np.float32), "combat": combat.astype(np.float32),
        "outcome": outcome.astype(np.float32), "impact": impact.astype(np.float32),
        "contact": contact.astype(np.float32), "traversal": traversal.astype(np.float32),
        "recovery": recovery.astype(np.float32), "interest": interest.astype(np.float32),
        "dull": dull,
    })

    thresholds = cfg.get("event_thresholds", {})
    specs = (
        ("combat_burst", "combat", float(thresholds.get("combat_burst", .62)), .75, ("combat", "audio_rms", "center_motion", "audio_transient")),
        ("outcome_like", "outcome", float(thresholds.get("outcome_like", .58)), .90, ("outcome", "hud_change", "combat", "center_motion")),
        ("impact", "impact", float(thresholds.get("impact", .68)), .70, ("impact", "luma_delta", "audio_transient", "audio_rms")),
        ("contact", "contact", float(thresholds.get("contact", .60)), .90, ("contact", "center_motion", "combat", "hud_change")),
    )
    events: list[SemanticEvent] = []
    for kind, signal, threshold, spacing, evidence in specs:
        for index in _local_maxima(signals[signal], threshold, max(1, int(round(spacing * fps)))):
            events.append(_event(index, fps, kind, signal, signals, evidence))
    events.sort(key=lambda item: (item.time, item.kind))
    times = (np.arange(length, dtype=np.float32) + .5) / fps
    return SemanticTimeline(fps=fps, duration=length/fps, times=times, signals=signals, events=events)


def _ranges(mask: np.ndarray, start: int, end: int, fps: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    run: int | None = None
    for index in range(start, end):
        if bool(mask[index]) and run is None:
            run = index
        elif not bool(mask[index]) and run is not None:
            out.append((run/fps, index/fps)); run = None
    if run is not None:
        out.append((run/fps, end/fps))
    return out


def _window_events(timeline: SemanticTimeline, start: float, end: float) -> list[SemanticEvent]:
    return [event for event in timeline.events if start <= event.time <= end]


def _protect_range(start: float, end: float, events: list[SemanticEvent], radius: float) -> tuple[float, float] | None:
    left, right = start, end
    for event in events:
        if event.kind not in {"combat_burst", "outcome_like", "impact", "contact"} or event.confidence < .55:
            continue
        if event.time + radius <= left or event.time - radius >= right:
            continue
        if event.time <= (left + right) / 2:
            left = max(left, event.time + radius)
        else:
            right = min(right, event.time - radius)
    return (left, right) if right > left else None


def _editorial_segments(timeline: SemanticTimeline, start: float, end: float, events: list[SemanticEvent], config: dict[str, Any]) -> tuple[tuple[EditSegment, ...], tuple[str, ...], float] | None:
    cfg = config.get("semantic_editor", {})
    fps = timeline.fps
    i0, i1 = max(0, int(math.floor(start*fps))), min(len(timeline.times), int(math.ceil(end*fps)))
    min_action = float(cfg.get("min_dull_action_seconds", .65))
    max_action = float(cfg.get("max_single_dull_action_seconds", 2.2))
    protection = float(cfg.get("event_protection_seconds", .38))
    max_actions = int(cfg.get("max_editorial_actions", 2))
    speed = min(float(cfg.get("max_speed", 2.0)), float(cfg.get("traversal_speed", 1.75)))

    dull_ranges = _ranges(timeline.signals["dull"] > .5, i0, i1, fps)
    traversal_mask = (timeline.signals["traversal"] >= float(cfg.get("traversal_threshold", .58))) & (timeline.signals["combat"] < .42) & (timeline.signals["outcome"] < .38) & (timeline.signals["impact"] < .42)
    candidates: list[tuple[float, float, str, float]] = []
    for raw_start, raw_end, kind in [(a,b,"cut_dull") for a,b in dull_ranges] + [(a,b,"speed_traversal") for a,b in _ranges(traversal_mask, i0, i1, fps)]:
        protected = _protect_range(raw_start, raw_end, events, protection)
        if protected is None:
            continue
        a, b = max(start, protected[0]), min(end, protected[1])
        if b-a >= min_action:
            candidates.append((a,b,kind,b-a))
    candidates.sort(key=lambda item: item[3], reverse=True)

    chosen: list[tuple[float, float, str, float]] = []
    for item in candidates:
        a,b,_,duration = item
        if duration > max_action:
            return None
        if any(max(a,x[0]) < min(b,x[1]) for x in chosen):
            continue
        chosen.append(item)
        if len(chosen) >= max_actions:
            break
    chosen.sort(key=lambda item: item[0])

    boundaries = sorted({start, end, *[value for item in chosen for value in item[:2]]})
    segments: list[EditSegment] = []
    reasons: list[str] = []
    removed = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left+right)/2
        action = next((item for item in chosen if item[0] <= midpoint <= item[1]), None)
        if action is None:
            if right-left >= .08:
                segments.append(EditSegment(round(left,3), round(right,3), 1.0, "keep"))
        elif action[2] == "cut_dull":
            removed += right-left
            reasons.append(f"cut dull {left:.2f}-{right:.2f}s")
        else:
            removed += (right-left) - (right-left)/speed
            segments.append(EditSegment(round(left,3), round(right,3), round(speed,3), "compressed_traversal"))
            reasons.append(f"compress traversal {left:.2f}-{right:.2f}s at {speed:.2f}x")
    if not segments or removed > float(cfg.get("max_total_removed_equivalent_seconds", 2.5)):
        return None

    output_duration = sum((segment.end-segment.start)/segment.speed for segment in segments)
    if output_duration < float(cfg.get("minimum_output_seconds", 10.0)):
        return None

    residual = 0.0
    for a,b in dull_ranges:
        a,b = max(start,a), min(end,b)
        if b <= a:
            continue
        remaining = sum((min(b,s.end)-max(a,s.start))/s.speed for s in segments if max(a,s.start) < min(b,s.end))
        residual = max(residual, remaining)
    if residual > float(cfg.get("max_unexplained_low_interest_run_seconds", 1.0)):
        return None
    return tuple(segments), tuple(reasons), residual


def source_time_to_output(source_time: float, segments: tuple[EditSegment, ...], prefix_seconds: float = 0.0) -> float | None:
    out = prefix_seconds
    for segment in segments:
        if segment.start <= source_time <= segment.end:
            return out + (source_time-segment.start)/segment.speed
        out += (segment.end-segment.start)/segment.speed
    return None


def _route_story(timeline: SemanticTimeline, start: float, end: float, events: list[SemanticEvent], opening: float, config: dict[str, Any]) -> tuple[str, str, float | None, tuple[SemanticEvent, ...], tuple[str, ...]]:
    cfg = config.get("semantic_editor", {})
    outcomes = [event for event in events if event.kind == "outcome_like" and event.confidence >= .55]
    impacts = [event for event in events if event.kind == "impact" and event.confidence >= .62]
    combats = [event for event in events if event.kind == "combat_burst" and event.confidence >= .55]
    relevant = outcomes + impacts + combats
    strongest = max(relevant, key=lambda event: event.confidence, default=None)
    teaser_seconds = float(config.get("editorial", {}).get("cold_open_teaser_seconds", 1.25))

    if strongest is not None:
        index = min(len(timeline.times)-1, max(0, int(round(strongest.time*timeline.fps-.5))))
        gain = float(timeline.signals["interest"][index]) - opening
        if strongest.time-start >= float(cfg.get("cold_open_min_event_delay_seconds", 1.35)) and strongest.time <= end-.35 and gain >= float(cfg.get("cold_open_min_interest_gain", .18)) and strongest.confidence >= .70:
            teaser_start = min(max(start, strongest.time-.42), end-teaser_seconds)
            return "cold_open_payoff", "cold_open_teaser", round(teaser_start,3), (strongest,), ("later semantic payoff materially beats opening",)

    chain_source = outcomes if len(outcomes) >= 3 else combats
    for index in range(len(chain_source)):
        chain = [chain_source[index]]
        for event in chain_source[index+1:]:
            if event.time-chain[-1].time <= 2.8 and event.time-chain[0].time <= 7.5:
                chain.append(event)
            if len(chain) == 3:
                return "multi_event_chain", "chain_escalation", None, tuple(chain), ("three coherent combat/outcome events",)

    if impacts:
        strongest_impact = max(impacts, key=lambda event: event.confidence)
        if strongest_impact.confidence >= float(cfg.get("impact_flash_min_confidence", .76)):
            return "explosive_impact", "impact_flash", None, (strongest_impact,), ("high-confidence audiovisual impact",)
    if outcomes:
        strongest_outcome = max(outcomes, key=lambda event: event.confidence)
        return "precision_outcome", "precision_punch", None, (strongest_outcome,), ("isolated outcome-like event",)
    selected = tuple(sorted(combats, key=lambda event: event.confidence, reverse=True)[:2])
    return "sustained_pressure", "clean_pressure", None, selected, ("sustained pressure kept readable",)


def plan_window(timeline: SemanticTimeline, start: float, end: float, config: dict[str, Any]) -> SemanticPlan | None:
    cfg = config.get("semantic_editor", {})
    fps = timeline.fps
    i0, i1 = max(0, int(math.floor(start*fps))), min(len(timeline.times), int(math.ceil(end*fps)))
    if i1-i0 < int(round(10*fps)):
        return None
    interest = timeline.signals["interest"][i0:i1]
    combat, outcome, impact = timeline.signals["combat"][i0:i1], timeline.signals["outcome"][i0:i1], timeline.signals["impact"][i0:i1]
    events = _window_events(timeline, start, end)
    meaningful = [event for event in events if event.kind in {"combat_burst","outcome_like","impact"} and event.confidence >= .55]
    if len(meaningful) < int(cfg.get("minimum_meaningful_events", 2)):
        return None

    quarters = [float(np.mean(part)) for part in np.array_split(interest, 4) if part.size]
    opening_bins = max(1, int(round(float(cfg.get("opening_seconds", 1.0))*fps)))
    opening, closing = float(np.mean(interest[:opening_bins])), float(np.mean(interest[-opening_bins:]))
    mean_interest, weakest = float(np.mean(interest)), min(quarters)
    low_threshold = float(cfg.get("low_interest_threshold", .30))
    low_fraction = float(np.mean(interest < low_threshold))

    built = _editorial_segments(timeline, start, end, events, config)
    if built is None:
        return None
    segments, action_reasons, residual = built
    base_duration = sum((segment.end-segment.start)/segment.speed for segment in segments)
    story, profile, cold_open, effect_events, route_reasons = _route_story(timeline, start, end, events, opening, config)
    planned_duration = base_duration + (float(config.get("editorial", {}).get("cold_open_teaser_seconds", 1.25)) if cold_open is not None else 0.0)
    if planned_duration > float(cfg.get("maximum_output_seconds", 20.0)):
        return None

    event_conf = float(np.mean([event.confidence for event in meaningful]))
    retention = float(np.clip(.37*mean_interest + .22*weakest + .15*opening + .10*closing + .16*min(1,len(meaningful)/4) - .12*low_fraction - .10*min(1,residual), 0, 1))
    engagement = float(np.clip(.28*float(np.percentile(interest,90)) + .22*opening + .20*event_conf + .18*min(1,len(meaningful)/4) + .12*max(float(np.max(outcome)),float(np.max(impact)),float(np.max(combat))), 0, 1))
    if retention < float(config.get("performance_targets", {}).get("retention_proxy_min", .30)) or engagement < float(config.get("performance_targets", {}).get("engagement_proxy_min", .30)) or weakest < float(cfg.get("minimum_weak_quarter_interest", .28)) or low_fraction > float(cfg.get("maximum_low_interest_fraction", .42)):
        return None

    score = .34*retention + .28*engagement + .14*weakest + .10*opening + .08*closing + .06*event_conf
    reasons = tuple(action_reasons) + tuple(route_reasons)
    if not action_reasons:
        reasons += ("no dull/traversal surgery required",)
    return SemanticPlan(
        round(start,3), round(end,3), round(end-start,3), round(planned_duration,3), round(score,6),
        round(retention,6), round(engagement,6), round(mean_interest,6), round(opening,6), round(closing,6),
        round(weakest,6), round(low_fraction,6), round(residual,3), story, profile, cold_open,
        segments, tuple(events), effect_events, reasons,
    )


def timeline_summary(timeline: SemanticTimeline, curve_fps: float = 2.0) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in timeline.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    stride = max(1, int(round(timeline.fps/curve_fps)))
    curve = [
        {
            "time": round(float(timeline.times[index]),3),
            "interest": round(float(timeline.signals["interest"][index]),4),
            "combat": round(float(timeline.signals["combat"][index]),4),
            "outcome": round(float(timeline.signals["outcome"][index]),4),
            "impact": round(float(timeline.signals["impact"][index]),4),
            "traversal": round(float(timeline.signals["traversal"][index]),4),
        }
        for index in range(0, len(timeline.times), stride)
    ]
    return {
        "semantic_engine": "deterministic-gameplay-v1",
        "sample_fps": timeline.fps,
        "duration": round(timeline.duration,3),
        "event_counts": counts,
        "events": [asdict(event) for event in timeline.events],
        "interest_curve": curve,
    }


def run_self_test() -> None:
    fps, duration = 6.0, 14.0
    length = int(fps*duration)
    times = (np.arange(length,dtype=np.float32)+.5)/fps
    interest = np.full(length,.48,dtype=np.float32); combat = np.full(length,.35,dtype=np.float32)
    outcome = np.zeros(length,dtype=np.float32); impact = np.zeros(length,dtype=np.float32)
    contact = np.full(length,.34,dtype=np.float32); traversal = np.zeros(length,dtype=np.float32)
    recovery = np.zeros(length,dtype=np.float32); dull = np.zeros(length,dtype=np.float32)
    dull[int(5*fps):int(5.9*fps)] = 1; interest[int(5*fps):int(5.9*fps)] = .16; combat[int(5*fps):int(5.9*fps)] = .10
    events: list[SemanticEvent] = []
    for when in (1.1,3.4,8.2):
        index = int(when*fps); combat[index] = .90; outcome[index] = .82; interest[index] = .92
        events += [SemanticEvent(when,"outcome_like",.82,{"outcome":.82}), SemanticEvent(when+.05,"combat_burst",.90,{"combat":.90})]
    timeline = SemanticTimeline(fps,duration,times,{"interest":interest,"combat":combat,"outcome":outcome,"impact":impact,"contact":contact,"traversal":traversal,"recovery":recovery,"dull":dull},events)
    config = {
        "editorial":{"cold_open_teaser_seconds":1.25},
        "performance_targets":{"retention_proxy_min":.20,"engagement_proxy_min":.20},
        "semantic_editor":{"minimum_meaningful_events":2,"minimum_output_seconds":10,"maximum_output_seconds":20,"min_dull_action_seconds":.65,"max_single_dull_action_seconds":2.2,"event_protection_seconds":.3,"max_editorial_actions":2,"traversal_speed":1.75,"max_speed":2,"max_total_removed_equivalent_seconds":2.5,"max_unexplained_low_interest_run_seconds":1,"minimum_weak_quarter_interest":.12,"maximum_low_interest_fraction":.5,"low_interest_threshold":.30,"cold_open_min_event_delay_seconds":1.35,"cold_open_min_interest_gain":.18,"impact_flash_min_confidence":.76}
    }
    plan = plan_window(timeline,0,duration,config)
    if plan is None or plan.max_unexplained_low_interest_run_seconds > 1 or source_time_to_output(8.2,plan.segments) is None:
        raise AssertionError("semantic editor self-test failed")
    print(json.dumps({"self_test":"PASS","story":plan.story_type,"profile":plan.effect_profile}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    else:
        parser.error("Use --self-test when invoking this module directly")
