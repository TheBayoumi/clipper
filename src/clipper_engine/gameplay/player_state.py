from __future__ import annotations

from typing import Any

import numpy as np


def _future_mean(values: np.ndarray, bins: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    bins = max(1, int(bins))
    for index in range(len(values)):
        right = min(len(values), index + bins + 1)
        out[index] = (
            float(np.mean(values[index + 1 : right]))
            if right > index + 1
            else float(values[index])
        )
    return out


def _past_max(values: np.ndarray, bins: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float32)
    bins = max(1, int(bins))
    for index in range(len(values)):
        left = max(0, index - bins)
        out[index] = float(np.max(values[left : index + 1]))
    return out


def annotate_timeline(timeline: Any, config: dict[str, Any]) -> Any:
    """Attach conservative player-death evidence to the shared semantic timeline.

    This detector marks a state discontinuity only. It does not infer victory,
    defeat quality, or require that the player win a fight.
    """
    cfg = dict(config.get("player_state_detector") or {})
    enabled = bool(cfg.get("enabled", False))
    length = len(timeline.times)
    signal = np.zeros(length, dtype=np.float32)
    if not enabled or length == 0:
        timeline.signals["player_death"] = signal
        timeline._player_state_diagnostics = {
            "enabled": enabled,
            "confirmed_count": 0,
            "candidates": [],
        }
        return timeline

    sig = timeline.signals
    arrays = [
        np.asarray(sig.get(name, np.zeros(length)), dtype=np.float32)[:length]
        for name in (
            "red_signal",
            "hud_change",
            "luma_delta",
            "global_motion",
            "combat",
            "contact",
            "outcome",
            "recovery",
        )
    ]
    red, hud, luma, motion, combat, contact, outcome, recovery = arrays
    fps = float(timeline.fps)

    activity = np.maximum.reduce((combat, contact, outcome))
    pre_activity = _past_max(activity, max(1, round(float(cfg.get("pre_seconds", 0.65)) * fps)))
    post_activity = _future_mean(
        activity,
        max(1, round(float(cfg.get("post_seconds", 0.80)) * fps)),
    )
    activity_drop = np.clip(pre_activity - post_activity, 0.0, 1.0)
    transition = np.maximum.reduce((hud, luma, motion))
    recent_red = _past_max(red, max(1, round(float(cfg.get("damage_memory_seconds", 0.55)) * fps)))
    post_recovery = _future_mean(
        recovery,
        max(1, round(float(cfg.get("post_seconds", 0.80)) * fps)),
    )

    score = np.clip(
        0.34 * recent_red
        + 0.24 * transition
        + 0.28 * activity_drop
        + 0.14 * post_recovery,
        0.0,
        1.0,
    )
    minimum_score = float(cfg.get("minimum_score", 0.82))
    minimum_red = float(cfg.get("minimum_red_signal", 0.72))
    minimum_transition = float(cfg.get("minimum_transition", 0.62))
    minimum_drop = float(cfg.get("minimum_activity_drop", 0.48))
    exclusion_seconds = float(cfg.get("exclusion_after_seconds", 1.25))
    minimum_spacing = float(cfg.get("minimum_spacing_seconds", 2.0))

    order = np.argsort(score)[::-1]
    confirmed: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for raw_index in order:
        index = int(raw_index)
        if float(score[index]) < minimum_score:
            break
        if float(recent_red[index]) < minimum_red:
            continue
        if float(transition[index]) < minimum_transition:
            continue
        if float(activity_drop[index]) < minimum_drop:
            continue
        if any(abs(index - prior) / fps < minimum_spacing for prior in confirmed):
            continue
        confirmed.append(index)

    confirmed.sort()
    for index in confirmed:
        end = min(length, index + max(1, round(exclusion_seconds * fps)))
        signal[index:end] = 1.0
        diagnostics.append(
            {
                "time": round(float(timeline.times[index]), 3),
                "score": round(float(score[index]), 4),
                "red_signal": round(float(recent_red[index]), 4),
                "transition": round(float(transition[index]), 4),
                "activity_drop": round(float(activity_drop[index]), 4),
                "post_recovery": round(float(post_recovery[index]), 4),
                "exclusion_seconds": exclusion_seconds,
            }
        )

    timeline.signals["player_death"] = signal
    timeline._player_state_diagnostics = {
        "enabled": True,
        "confirmed_count": len(diagnostics),
        "minimum_score": minimum_score,
        "minimum_red_signal": minimum_red,
        "minimum_transition": minimum_transition,
        "minimum_activity_drop": minimum_drop,
        "candidates": diagnostics,
        "policy": (
            "conservative state-discontinuity detection from severe damage, visual/HUD "
            "transition, and sustained post-event combat collapse; no player-win rule"
        ),
    }
    return timeline
