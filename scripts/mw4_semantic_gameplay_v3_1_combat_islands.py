from __future__ import annotations

import math
from typing import Any

import numpy as np


_BOUNDARY_EPS_SECONDS = 1e-3


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            out.append((start, index))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _sustained(mask: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    for left, right in _runs(mask):
        if (right - left) / fps >= seconds - 1e-9:
            out[left:right] = True
    return out


def _half_open_frame_slice(start: float, end: float, fps: float, length: int) -> tuple[int, int]:
    """Map millisecond-rounded [start, end) bounds back to the analysis frame grid.

    Canonical combat-island bounds are serialized to milliseconds. A true grid
    boundary such as 55/12 = 4.583333... therefore becomes 4.583. Using floor()
    at that rounded start incorrectly includes frame 54 from the rejected hard
    gap. Likewise a rounded-up end can include the first rejected frame after an
    island. Ceil with a one-millisecond tolerance preserves the original
    half-open interval semantics without expanding the accepted source region.
    """
    if end <= start:
        return 0, 0
    i0 = max(0, int(math.ceil((float(start) - _BOUNDARY_EPS_SECONDS) * fps)))
    i1 = min(length, int(math.ceil((float(end) - _BOUNDARY_EPS_SECONDS) * fps)))
    return i0, max(i0, i1)


def gap_signal(timeline: Any, config: dict[str, Any]) -> np.ndarray:
    cfg = config["combat_state_verifier"]
    sig = timeline.signals
    arrays = [np.asarray(sig[name], dtype=np.float32) for name in ("combat", "outcome", "impact", "traversal", "recovery")]
    n = min(map(len, arrays))
    combat, outcome, impact, traversal, recovery = (item[:n] for item in arrays)
    movement = (
        ((traversal >= _f(cfg.get("traversal_break_threshold", 0.60), 0.60)) | (recovery >= _f(cfg.get("recovery_break_threshold", 0.60), 0.60)))
        & (combat <= _f(cfg.get("break_combat_ceiling", 0.50), 0.50))
        & (outcome <= _f(cfg.get("break_outcome_ceiling", 0.42), 0.42))
        & (impact <= _f(cfg.get("break_impact_ceiling", 0.45), 0.45))
    )
    support = np.maximum.reduce((combat, outcome, impact))
    quiet = support < _f(cfg.get("combat_island_support_floor", 0.38), 0.38)
    return (
        _sustained(movement, float(timeline.fps), _f(cfg.get("maximum_continuity_break_run_seconds", 0.55), 0.55))
        | _sustained(quiet, float(timeline.fps), _f(cfg.get("maximum_combat_island_quiet_run_seconds", 0.75), 0.75))
    ).astype(np.float32)


def fragment_bounds(start: float, end: float, hard: np.ndarray, fps: float) -> list[tuple[float, float]]:
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(hard), int(math.ceil(end * fps)))
    if i1 <= i0:
        return []
    out: list[tuple[float, float]] = []
    cursor = start
    for left, right in _runs(hard[i0:i1] > 0.5):
        gap_start = max(start, (i0 + left) / fps)
        gap_end = min(end, (i0 + right) / fps)
        if gap_start > cursor + 1e-6:
            out.append((cursor, gap_start))
        cursor = max(cursor, gap_end)
    if end > cursor + 1e-6:
        out.append((cursor, end))
    return out


def build(timeline: Any, coarse: tuple[Any, ...], config: dict[str, Any], core: Any, combat: Any, Engagement: Any) -> tuple[Any, ...]:
    """Build exact gap-split combat islands. Short fragments are dropped, never padded."""
    cfg = config["combat_state_verifier"]
    minimum = _f(cfg.get("minimum_combat_island_seconds", 1.0), 1.0)
    hard = gap_signal(timeline, config)
    anchors = [(event, combat.hostile_decision(event, config)) for event in timeline.consolidated_events]
    anchors = [(event, decision) for event, decision in anchors if decision.hostile]
    islands: list[Any] = []
    for envelope in coarse:
        members = [
            (event, decision) for event, decision in anchors
            if core._shot_index(timeline.shots, float(event.time)) == int(envelope.shot_index)
            and float(envelope.start) - 1e-3 <= float(event.time) <= float(envelope.end) + 1e-3
        ]
        if not members:
            continue
        for start, end in fragment_bounds(float(envelope.start), float(envelope.end), hard, float(timeline.fps)):
            fragment = [(event, decision) for event, decision in members if start - 1e-3 <= float(event.time) <= end + 1e-3]
            if not fragment or end - start < minimum - 1e-3:
                continue
            islands.append(Engagement(
                round(start, 3), round(end, 3), int(envelope.shot_index),
                round(float(np.mean([decision.score for _, decision in fragment])), 4),
                tuple(event for event, _ in fragment),
            ))
    unique: list[Any] = []
    for island in sorted(islands, key=lambda item: (item.start, item.end)):
        key = (island.shot_index, tuple(round(float(event.time), 3) for event in island.events))
        prior = next((i for i, old in enumerate(unique) if (old.shot_index, tuple(round(float(event.time), 3) for event in old.events)) == key), None)
        if prior is None:
            unique.append(island)
        elif island.end - island.start > unique[prior].end - unique[prior].start:
            unique[prior] = island
    return tuple(sorted(unique, key=lambda item: (item.start, item.end)))


def bridge_supported(timeline: Any, left: float, right: float, config: dict[str, Any]) -> bool:
    if right <= left + 1e-3:
        return True
    cfg = config["combat_state_verifier"]
    if right - left > _f(cfg.get("maximum_verified_inter_engagement_gap_seconds", 1.25), 1.25) + 1e-3:
        return False
    fps = float(timeline.fps)
    i0, i1 = _half_open_frame_slice(left, right, fps, len(timeline.times))
    if i1 <= i0:
        return False
    hard = np.asarray(timeline.signals.get("combat_island_gap", []), dtype=np.float32)
    if hard.size and float(np.max(hard[i0:min(i1, len(hard))])) > 0.5:
        return False
    arrays = [np.asarray(timeline.signals[name], dtype=np.float32)[i0:i1] for name in ("combat", "outcome", "impact")]
    n = min(map(len, arrays))
    if n <= 0:
        return False
    support = np.maximum.reduce(tuple(item[:n] for item in arrays))
    return float(np.mean(support >= _f(cfg.get("combat_island_support_floor", 0.38), 0.38))) >= _f(cfg.get("minimum_bridge_combat_support_fraction", 0.50), 0.50)


def segment_crosses_gap(timeline: Any, start: float, end: float) -> bool:
    hard = np.asarray(timeline.signals.get("combat_island_gap", []), dtype=np.float32)
    if hard.size == 0:
        return True
    i0, i1 = _half_open_frame_slice(float(start), float(end), float(timeline.fps), len(hard))
    return i1 <= i0 or float(np.max(hard[i0:i1])) > 0.5


def self_test() -> None:
    hard = np.zeros(100, dtype=np.float32)
    hard[40:55] = 1.0
    if fragment_bounds(1.0, 9.0, hard, 10.0) != [(1.0, 4.0), (5.5, 9.0)]:
        raise AssertionError("hard combat gap was not removed exactly")

    # Millisecond serialization must not make an exact 12-fps fragment reabsorb
    # an adjacent rejected frame. Both sides exercise recurring-decimal rounding.
    hard12 = np.zeros(120, dtype=np.float32)
    hard12[41:55] = 1.0
    left_end = round(41 / 12.0, 3)   # 3.416666... -> 3.417 (rounded up)
    right_start = round(55 / 12.0, 3)  # 4.583333... -> 4.583 (rounded down)
    timeline = type("Timeline", (), {"fps": 12.0, "signals": {"combat_island_gap": hard12}})()
    if segment_crosses_gap(timeline, 1.0, left_end):
        raise AssertionError("rounded-up exact island end reabsorbed first hard-gap frame")
    if segment_crosses_gap(timeline, right_start, 8.0):
        raise AssertionError("rounded-down exact island start reabsorbed last hard-gap frame")
    if not segment_crosses_gap(timeline, 1.0, 8.0):
        raise AssertionError("true hard combat gap was ignored")

    print("MW4 canonical combat-island self-test: PASS")
