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


def _half_open_frame_slice(
    start: float,
    end: float,
    fps: float,
    length: int,
) -> tuple[int, int]:
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
    arrays = [
        np.asarray(sig[name], dtype=np.float32)
        for name in ("combat", "outcome", "impact", "traversal", "recovery")
    ]
    n = min(map(len, arrays))
    combat, outcome, impact, traversal, recovery = (item[:n] for item in arrays)
    movement = (
        (
            (
                traversal
                >= _f(cfg.get("traversal_break_threshold", 0.60), 0.60)
            )
            | (
                recovery
                >= _f(cfg.get("recovery_break_threshold", 0.60), 0.60)
            )
        )
        & (combat <= _f(cfg.get("break_combat_ceiling", 0.50), 0.50))
        & (outcome <= _f(cfg.get("break_outcome_ceiling", 0.42), 0.42))
        & (impact <= _f(cfg.get("break_impact_ceiling", 0.45), 0.45))
    )
    support = np.maximum.reduce((combat, outcome, impact))
    quiet = support < _f(cfg.get("combat_island_support_floor", 0.38), 0.38)
    return (
        _sustained(
            movement,
            float(timeline.fps),
            _f(cfg.get("maximum_continuity_break_run_seconds", 0.55), 0.55),
        )
        | _sustained(
            quiet,
            float(timeline.fps),
            _f(cfg.get("maximum_combat_island_quiet_run_seconds", 0.75), 0.75),
        )
    ).astype(np.float32)


def fragment_bounds(
    start: float,
    end: float,
    hard: np.ndarray,
    fps: float,
) -> list[tuple[float, float]]:
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


def _event_in_fragment(event: Any, start: float, end: float) -> bool:
    value = float(event.time)
    return start - _BOUNDARY_EPS_SECONDS <= value <= end + _BOUNDARY_EPS_SECONDS


def _support_component_for_event(
    timeline: Any,
    event: Any,
    hard: np.ndarray,
    core: Any,
) -> tuple[int, float, float] | None:
    """Return the exact same-shot, hard-gap-free component containing an event."""
    shot_index = core._shot_index(timeline.shots, float(event.time))
    shot = timeline.shots[shot_index]
    for start, end in fragment_bounds(
        float(shot.start),
        float(shot.end),
        hard,
        float(timeline.fps),
    ):
        if _event_in_fragment(event, start, end):
            return int(shot_index), float(start), float(end)
    return None


def _dedupe_islands(islands_in: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for island in sorted(islands_in, key=lambda item: (item.start, item.end)):
        key = (
            island.shot_index,
            tuple(round(float(event.time), 3) for event in island.events),
        )
        prior = next(
            (
                i
                for i, old in enumerate(unique)
                if (
                    old.shot_index,
                    tuple(round(float(event.time), 3) for event in old.events),
                )
                == key
            ),
            None,
        )
        if prior is None:
            unique.append(island)
        elif island.end - island.start > unique[prior].end - unique[prior].start:
            unique[prior] = island
    return unique


def _recover_orphaned_anchors(
    timeline: Any,
    anchors: list[tuple[Any, Any]],
    hard: np.ndarray,
    core: Any,
    Engagement: Any,
    islands_in: list[Any],
    minimum: float,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Recover verified hostile evidence excluded by pre-verification coarse gating.

    Coarse envelopes are built before local direct-interaction verification and can
    legitimately omit an event whose coarse semantic confidence is low even when
    the local verifier later proves a hostile interaction. Verified identity is
    authoritative here: an orphan may seed its exact same-shot, hard-gap-free
    support component. No threshold is lowered and no hard gap is crossed.
    """
    assigned_ids = {id(event) for island in islands_in for event in island.events}
    orphan_ids = {
        id(event)
        for event, _ in anchors
        if id(event) not in assigned_ids
    }
    if not orphan_ids:
        return islands_in, []

    components: dict[
        tuple[int, float, float],
        list[tuple[Any, Any]],
    ] = {}
    event_component: dict[int, tuple[int, float, float]] = {}
    for event, decision in anchors:
        component = _support_component_for_event(timeline, event, hard, core)
        if component is None:
            continue
        components.setdefault(component, []).append((event, decision))
        event_component[id(event)] = component

    recovery_keys = {
        event_component[event_id]
        for event_id in orphan_ids
        if event_id in event_component
    }
    recovered = list(islands_in)
    diagnostics: list[dict[str, Any]] = []

    for shot_index, start, end in sorted(recovery_keys):
        members = components[(shot_index, start, end)]
        duration = end - start
        member_ids = {id(event) for event, _ in members}
        orphan_times = [
            round(float(event.time), 3)
            for event, _ in members
            if id(event) in orphan_ids
        ]

        diagnostics.append(
            {
                "shot_index": int(shot_index),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "verified_event_times": [
                    round(float(event.time), 3) for event, _ in members
                ],
                "recovered_orphan_event_times": orphan_times,
                "meets_minimum_combat_island_seconds": (
                    duration >= minimum - _BOUNDARY_EPS_SECONDS
                ),
            }
        )

        if duration < minimum - _BOUNDARY_EPS_SECONDS:
            continue

        recovered = [
            old
            for old in recovered
            if not (
                int(old.shot_index) == int(shot_index)
                and float(old.start) >= start - _BOUNDARY_EPS_SECONDS
                and float(old.end) <= end + _BOUNDARY_EPS_SECONDS
                and all(id(event) in member_ids for event in old.events)
            )
        ]
        recovered.append(
            Engagement(
                round(start, 3),
                round(end, 3),
                int(shot_index),
                round(
                    float(np.mean([decision.score for _, decision in members])),
                    4,
                ),
                tuple(event for event, _ in members),
            )
        )

    return _dedupe_islands(recovered), diagnostics


def _attach_assignment_diagnostics(
    timeline: Any,
    anchors: list[tuple[Any, Any]],
    islands_out: tuple[Any, ...],
    recovery: list[dict[str, Any]] | None = None,
) -> None:
    """Expose verified-evidence assignment and any topology recovery."""
    diagnostics = dict(
        getattr(timeline, "_local_interaction_diagnostics", {}) or {}
    )
    diagnostics["canonical_combat_islands"] = [
        {
            "start": round(float(island.start), 3),
            "end": round(float(island.end), 3),
            "duration": round(float(island.end) - float(island.start), 3),
            "shot_index": int(island.shot_index),
            "confidence": round(float(island.confidence), 4),
            "event_times": [
                round(float(event.time), 3) for event in island.events
            ],
            "event_kinds": [list(event.kinds) for event in island.events],
        }
        for island in islands_out
    ]
    assignments: list[dict[str, Any]] = []
    for event, decision in anchors:
        assigned = [
            [round(float(island.start), 3), round(float(island.end), 3)]
            for island in islands_out
            if event in island.events
        ]
        assignments.append(
            {
                "time": round(float(event.time), 3),
                "kinds": list(event.kinds),
                "hostile_score": round(float(decision.score), 4),
                "assigned_islands": assigned,
                "orphaned_from_combat_islands": not bool(assigned),
            }
        )
    diagnostics["verified_hostile_event_assignment"] = assignments
    diagnostics["orphaned_verified_hostile_event_count"] = sum(
        1 for item in assignments if item["orphaned_from_combat_islands"]
    )
    diagnostics["orphan_recovery_components"] = list(recovery or [])
    diagnostics["orphan_recovery_component_count"] = len(recovery or [])
    setattr(timeline, "_local_interaction_diagnostics", diagnostics)


def build(
    timeline: Any,
    coarse: tuple[Any, ...],
    config: dict[str, Any],
    core: Any,
    combat: Any,
    Engagement: Any,
) -> tuple[Any, ...]:
    """Build exact gap-split islands without letting coarse gating drop verified hostility."""
    cfg = config["combat_state_verifier"]
    minimum = _f(cfg.get("minimum_combat_island_seconds", 1.0), 1.0)
    hard = gap_signal(timeline, config)
    anchors = [
        (event, combat.hostile_decision(event, config))
        for event in timeline.consolidated_events
    ]
    anchors = [
        (event, decision) for event, decision in anchors if decision.hostile
    ]
    combat_islands: list[Any] = []
    for envelope in coarse:
        members = [
            (event, decision)
            for event, decision in anchors
            if core._shot_index(timeline.shots, float(event.time))
            == int(envelope.shot_index)
            and float(envelope.start) - _BOUNDARY_EPS_SECONDS
            <= float(event.time)
            <= float(envelope.end) + _BOUNDARY_EPS_SECONDS
        ]
        if not members:
            continue
        for start, end in fragment_bounds(
            float(envelope.start),
            float(envelope.end),
            hard,
            float(timeline.fps),
        ):
            fragment = [
                (event, decision)
                for event, decision in members
                if _event_in_fragment(event, start, end)
            ]
            if not fragment or end - start < minimum - _BOUNDARY_EPS_SECONDS:
                continue
            combat_islands.append(
                Engagement(
                    round(start, 3),
                    round(end, 3),
                    int(envelope.shot_index),
                    round(
                        float(
                            np.mean(
                                [decision.score for _, decision in fragment]
                            )
                        ),
                        4,
                    ),
                    tuple(event for event, _ in fragment),
                )
            )

    unique = _dedupe_islands(combat_islands)
    unique, recovery = _recover_orphaned_anchors(
        timeline,
        anchors,
        hard,
        core,
        Engagement,
        unique,
        minimum,
    )
    result = tuple(sorted(unique, key=lambda item: (item.start, item.end)))
    _attach_assignment_diagnostics(timeline, anchors, result, recovery)

    remaining = [
        event
        for event, _ in anchors
        if not any(event in island.events for island in result)
    ]
    if remaining:
        times = ", ".join(f"{float(event.time):.3f}" for event in remaining)
        raise RuntimeError(
            "verified hostile event could not be assigned to a minimum-duration "
            "same-shot hard-gap-free canonical combat island; "
            f"event_times=[{times}]"
        )
    return result


def bridge_supported(
    timeline: Any,
    left: float,
    right: float,
    config: dict[str, Any],
) -> bool:
    if right <= left + 1e-3:
        return True
    cfg = config["combat_state_verifier"]
    if (
        right - left
        > _f(
            cfg.get("maximum_verified_inter_engagement_gap_seconds", 1.25),
            1.25,
        )
        + 1e-3
    ):
        return False
    fps = float(timeline.fps)
    i0, i1 = _half_open_frame_slice(
        left,
        right,
        fps,
        len(timeline.times),
    )
    if i1 <= i0:
        return False
    hard = np.asarray(
        timeline.signals.get("combat_island_gap", []),
        dtype=np.float32,
    )
    if hard.size and float(np.max(hard[i0 : min(i1, len(hard))])) > 0.5:
        return False
    arrays = [
        np.asarray(timeline.signals[name], dtype=np.float32)[i0:i1]
        for name in ("combat", "outcome", "impact")
    ]
    n = min(map(len, arrays))
    if n <= 0:
        return False
    support = np.maximum.reduce(tuple(item[:n] for item in arrays))
    return float(
        np.mean(
            support
            >= _f(cfg.get("combat_island_support_floor", 0.38), 0.38)
        )
    ) >= _f(cfg.get("minimum_bridge_combat_support_fraction", 0.50), 0.50)


def segment_crosses_gap(timeline: Any, start: float, end: float) -> bool:
    hard = np.asarray(
        timeline.signals.get("combat_island_gap", []),
        dtype=np.float32,
    )
    if hard.size == 0:
        return True
    i0, i1 = _half_open_frame_slice(
        float(start),
        float(end),
        float(timeline.fps),
        len(hard),
    )
    return i1 <= i0 or float(np.max(hard[i0:i1])) > 0.5


def self_test() -> None:
    hard = np.zeros(100, dtype=np.float32)
    hard[40:55] = 1.0
    if fragment_bounds(1.0, 9.0, hard, 10.0) != [
        (1.0, 4.0),
        (5.5, 9.0),
    ]:
        raise AssertionError("hard combat gap was not removed exactly")

    hard12 = np.zeros(120, dtype=np.float32)
    hard12[41:55] = 1.0
    left_end = round(41 / 12.0, 3)
    right_start = round(55 / 12.0, 3)
    timeline = type(
        "Timeline",
        (),
        {"fps": 12.0, "signals": {"combat_island_gap": hard12}},
    )()
    if segment_crosses_gap(timeline, 1.0, left_end):
        raise AssertionError(
            "rounded-up exact island end reabsorbed first hard-gap frame"
        )
    if segment_crosses_gap(timeline, right_start, 8.0):
        raise AssertionError(
            "rounded-down exact island start reabsorbed last hard-gap frame"
        )
    if not segment_crosses_gap(timeline, 1.0, 8.0):
        raise AssertionError("true hard combat gap was ignored")

    event_a = type(
        "Event",
        (),
        {"time": 2.0, "kinds": ("outcome_like",)},
    )()
    event_b = type(
        "Event",
        (),
        {"time": 7.5, "kinds": ("outcome_like",)},
    )()
    shot = type("Shot", (), {"start": 0.0, "end": 10.0})()
    coarse = (
        type(
            "Coarse",
            (),
            {"start": 1.5, "end": 2.5, "shot_index": 0},
        )(),
    )

    class FakeCore:
        @staticmethod
        def _shot_index(shots: tuple[Any, ...], time: float) -> int:
            return next(
                index
                for index, item in enumerate(shots)
                if float(item.start) <= time <= float(item.end)
            )

    class FakeCombat:
        @staticmethod
        def hostile_decision(event: Any, config: dict[str, Any]) -> Any:
            score = 0.8 if event is event_a else 0.9
            return type(
                "Decision",
                (),
                {"hostile": True, "score": score},
            )()

    def fake_engagement(
        start: float,
        end: float,
        shot_index: int,
        confidence: float,
        events: tuple[Any, ...],
    ) -> Any:
        return type(
            "Engagement",
            (),
            {
                "start": start,
                "end": end,
                "shot_index": shot_index,
                "confidence": confidence,
                "events": events,
            },
        )()

    support = np.ones(100, dtype=np.float32)
    zeros = np.zeros(100, dtype=np.float32)
    recovery_timeline = type(
        "Timeline",
        (),
        {
            "fps": 10.0,
            "times": np.arange(100, dtype=np.float32) / 10.0,
            "shots": (shot,),
            "consolidated_events": (event_a, event_b),
            "signals": {
                "combat": support,
                "outcome": support,
                "impact": support,
                "traversal": zeros,
                "recovery": zeros,
            },
            "_local_interaction_diagnostics": {},
        },
    )()
    recovery_config = {
        "combat_state_verifier": {
            "minimum_combat_island_seconds": 1.0,
            "traversal_break_threshold": 0.60,
            "recovery_break_threshold": 0.60,
            "break_combat_ceiling": 0.50,
            "break_outcome_ceiling": 0.42,
            "break_impact_ceiling": 0.45,
            "maximum_continuity_break_run_seconds": 0.55,
            "combat_island_support_floor": 0.38,
            "maximum_combat_island_quiet_run_seconds": 0.75,
        }
    }
    recovered = build(
        recovery_timeline,
        coarse,
        recovery_config,
        FakeCore,
        FakeCombat,
        fake_engagement,
    )
    if not any(event_b in item.events for item in recovered):
        raise AssertionError(
            "locally verified hostile event remained orphaned by coarse-envelope gating"
        )
    diagnostics = recovery_timeline._local_interaction_diagnostics
    if diagnostics["orphaned_verified_hostile_event_count"] != 0:
        raise AssertionError("orphan recovery did not produce total assignment")
    if diagnostics["orphan_recovery_component_count"] != 1:
        raise AssertionError("coarse-gating regression did not exercise recovery")

    print("MW4 canonical combat-island self-test: PASS")
