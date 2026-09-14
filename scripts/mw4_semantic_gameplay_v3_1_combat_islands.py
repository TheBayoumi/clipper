from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("combat_state_verifier", {})


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            out.append((start, index))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _sustained_mask(mask: np.ndarray, fps: float, minimum_seconds: float) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    for left, right in _runs(np.asarray(mask, dtype=bool)):
        if (right - left) / fps > minimum_seconds:
            out[left:right] = True
    return out


def _island_gap_signal(timeline: Any, config: dict[str, Any]) -> np.ndarray:
    """Return only sustained continuity failures, never transient action valleys.

    Reload/traversal/recovery is a hard break only when combat, outcome and impact
    evidence are simultaneously low. Separately, an extended absence of all three
    combat-support signals is a search/idle break even if camera motion is modest.
    Short aim adjustments and cover transitions remain inside the combat island.
    """
    cfg = _cfg(config)
    sig = timeline.signals
    combat = np.asarray(sig["combat"], dtype=np.float32)
    outcome = np.asarray(sig["outcome"], dtype=np.float32)
    impact = np.asarray(sig["impact"], dtype=np.float32)
    traversal = np.asarray(sig["traversal"], dtype=np.float32)
    recovery = np.asarray(sig["recovery"], dtype=np.float32)
    n = min(map(len, (combat, outcome, impact, traversal, recovery)))
    combat, outcome, impact = combat[:n], outcome[:n], impact[:n]
    traversal, recovery = traversal[:n], recovery[:n]

    movement_break = (
        ((traversal >= _f(cfg.get("traversal_break_threshold", 0.60), 0.60))
         | (recovery >= _f(cfg.get("recovery_break_threshold", 0.60), 0.60)))
        & (combat <= _f(cfg.get("break_combat_ceiling", 0.50), 0.50))
        & (outcome <= _f(cfg.get("break_outcome_ceiling", 0.42), 0.42))
        & (impact <= _f(cfg.get("break_impact_ceiling", 0.45), 0.45))
    )
    support = np.maximum.reduce((combat, outcome, impact))
    quiet = support < _f(cfg.get("combat_island_support_floor", 0.38), 0.38)

    hard_movement = _sustained_mask(
        movement_break,
        float(timeline.fps),
        _f(cfg.get("maximum_continuity_break_run_seconds", 0.55), 0.55),
    )
    hard_quiet = _sustained_mask(
        quiet,
        float(timeline.fps),
        _f(cfg.get("maximum_combat_island_quiet_run_seconds", 0.75), 0.75),
    )
    return (hard_movement | hard_quiet).astype(np.float32)


def _hostile_events(timeline: Any) -> tuple[Any, ...]:
    seen: set[tuple[float, tuple[str, ...]]] = set()
    events: list[Any] = []
    for engagement in timeline.engagements:
        for event in engagement.events:
            key = (round(float(event.time), 3), tuple(getattr(event, "kinds", ()) or ()))
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
    return tuple(sorted(events, key=lambda item: float(item.time)))


def _fragment_bounds(start: float, end: float, gap_signal: np.ndarray, fps: float) -> list[tuple[float, float]]:
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(gap_signal), int(math.ceil(end * fps)))
    if i1 <= i0:
        return []
    hard_runs = _runs(gap_signal[i0:i1] > 0.5)
    fragments: list[tuple[float, float]] = []
    cursor = start
    for left, right in hard_runs:
        gap_start = max(start, (i0 + left) / fps)
        gap_end = min(end, (i0 + right) / fps)
        if gap_start > cursor + 1e-6:
            fragments.append((cursor, gap_start))
        cursor = max(cursor, gap_end)
    if end > cursor + 1e-6:
        fragments.append((cursor, end))
    return fragments


def _event_in_range(event: Any, start: float, end: float) -> bool:
    value = float(event.time)
    return start - 1e-6 <= value <= end + 1e-6


def build_combat_islands(timeline: Any, config: dict[str, Any], legacy: Any, combat: Any) -> tuple[Any, ...]:
    """Grow verified hostile anchors only inside coarse combat envelopes.

    Coarse engagements are useful temporal envelopes but are not trusted as hostile.
    An envelope (or fragment of one) becomes an island only when it contains at
    least one independently verified hostile anchor. Sustained reload/search gaps
    are removed from the envelope before the island is emitted.
    """
    cfg = _cfg(config)
    coarse = legacy.refined.core.cluster_engagements(
        timeline.base,
        timeline.shots,
        timeline.consolidated_events,
        config,
    )
    anchors = _hostile_events(timeline)
    gap_signal = _island_gap_signal(timeline, config)
    minimum = _f(cfg.get("minimum_combat_island_seconds", 1.0), 1.0)
    islands: list[Any] = []

    for envelope in coarse:
        shot_index = int(envelope.shot_index)
        in_envelope = tuple(
            event for event in anchors
            if legacy.refined.core._shot_index(timeline.shots, float(event.time)) == shot_index
            and _event_in_range(event, float(envelope.start), float(envelope.end))
        )
        if not in_envelope:
            continue
        for raw_start, raw_end in _fragment_bounds(
            float(envelope.start), float(envelope.end), gap_signal, float(timeline.fps)
        ):
            fragment_events = tuple(event for event in in_envelope if _event_in_range(event, raw_start, raw_end))
            if not fragment_events:
                continue
            if raw_end - raw_start < minimum:
                center = float(fragment_events[0].time)
                wanted_left = max(float(envelope.start), center - minimum / 2.0)
                wanted_right = min(float(envelope.end), wanted_left + minimum)
                raw_start, raw_end = wanted_left, wanted_right
                if raw_end - raw_start < minimum - 1e-6:
                    continue
            decisions = [combat._event_hostile_decision(event, config) for event in fragment_events]
            scores = [decision.score for decision in decisions if decision.hostile]
            if not scores:
                continue
            islands.append(
                legacy.Engagement(
                    start=round(raw_start, 3),
                    end=round(raw_end, 3),
                    shot_index=shot_index,
                    confidence=round(float(np.mean(scores)), 4),
                    events=fragment_events,
                )
            )

    unique: list[Any] = []
    for island in sorted(islands, key=lambda item: (item.start, -(item.end - item.start))):
        key = tuple(round(float(event.time), 3) for event in island.events)
        prior_index = next((i for i, old in enumerate(unique)
                            if tuple(round(float(event.time), 3) for event in old.events) == key
                            and old.shot_index == island.shot_index), None)
        if prior_index is None:
            unique.append(island)
        elif (island.end - island.start) > (unique[prior_index].end - unique[prior_index].start):
            unique[prior_index] = island
    return tuple(sorted(unique, key=lambda item: (item.start, item.end)))


def _bridge_supported(timeline: Any, left: float, right: float, config: dict[str, Any]) -> bool:
    if right <= left + 1e-6:
        return True
    cfg = _cfg(config)
    maximum_gap = _f(cfg.get("maximum_verified_inter_engagement_gap_seconds", 1.25), 1.25)
    if right - left > maximum_gap:
        return False
    fps = float(timeline.fps)
    i0 = max(0, int(math.floor(left * fps)))
    i1 = min(len(timeline.times), int(math.ceil(right * fps)))
    if i1 <= i0:
        return True
    hard = np.asarray(timeline.signals.get("combat_island_gap", []), dtype=np.float32)
    if hard.size and float(np.max(hard[i0:min(i1, len(hard))])) > 0.5:
        return False
    combat = np.asarray(timeline.signals["combat"], dtype=np.float32)[i0:i1]
    outcome = np.asarray(timeline.signals["outcome"], dtype=np.float32)[i0:i1]
    impact = np.asarray(timeline.signals["impact"], dtype=np.float32)[i0:i1]
    n = min(len(combat), len(outcome), len(impact))
    if n == 0:
        return False
    support = np.maximum.reduce((combat[:n], outcome[:n], impact[:n]))
    floor = _f(cfg.get("combat_island_support_floor", 0.38), 0.38)
    fraction = float(np.mean(support >= floor))
    return fraction >= _f(cfg.get("minimum_bridge_combat_support_fraction", 0.50), 0.50)


def island_candidate_chains(timeline: Any, config: dict[str, Any], legacy: Any) -> list[tuple[Any, ...]]:
    """Construct contiguous stories from verified combat islands only."""
    editor = config["semantic_editor"]
    minimum = _f(editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(editor.get("maximum_output_seconds", 20.0), 20.0)
    max_engagements = int(editor.get("maximum_engagements_per_story", 7))
    tail = _f(editor["ending"].get("preferred_payoff_tail_seconds", 0.45), 0.45)
    lead = _f(config["editorial"].get("finishing_move_opening_lead_seconds", 0.28), 0.28)
    chains: list[tuple[Any, ...]] = []

    for start_index, first in enumerate(timeline.engagements):
        shot = timeline.shots[first.shot_index]
        chain: list[Any] = []
        for candidate in timeline.engagements[start_index:]:
            if candidate.shot_index != first.shot_index:
                break
            if chain and not _bridge_supported(timeline, float(chain[-1].end), float(candidate.start), config):
                break
            chain.append(candidate)
            if len(chain) > max_engagements:
                break
            start = max(float(shot.start), float(chain[0].events[0].time) - 0.30)
            payoff = legacy.refined._last_payoff(tuple(chain))
            end = min(float(shot.end), float(payoff.time) + tail) if payoff is not None else min(float(shot.end), float(chain[-1].end))
            if minimum <= end - start <= maximum:
                chains.append(tuple(chain))

    for span in timeline.finishing_moves:
        same_shot = [e for e in timeline.engagements if e.shot_index == span.shot_index and e.end >= span.start]
        if not same_shot:
            continue
        shot = timeline.shots[span.shot_index]
        start = max(float(shot.start), float(span.start) - lead)
        chain: list[Any] = []
        for engagement in same_shot:
            if chain and not _bridge_supported(timeline, float(chain[-1].end), float(engagement.start), config):
                break
            chain.append(engagement)
            if len(chain) > max_engagements:
                break
            payoff = legacy.refined._last_payoff(tuple(chain))
            if payoff is None:
                continue
            end = min(float(shot.end), float(payoff.time) + tail)
            if minimum <= end - start <= maximum:
                chains.append(tuple(chain))

    unique: list[tuple[Any, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for chain in chains:
        key = tuple(round(float(item.start), 3) for item in chain)
        if key not in seen:
            seen.add(key)
            unique.append(chain)
    return unique


def refine_timeline(timeline: Any, config: dict[str, Any], legacy: Any, combat: Any) -> Any:
    strict_count = len(timeline.engagements)
    islands = build_combat_islands(timeline, config, legacy, combat)
    timeline.engagements = islands
    timeline.signals["combat_island_gap"] = _island_gap_signal(timeline, config)
    timeline.signals["verified_hostile"] = combat._verified_hostile_signal(timeline, islands)
    diagnostics = dict(getattr(timeline, "_combat_state_diagnostics", {}))
    diagnostics.update({
        "strict_anchor_engagement_count": strict_count,
        "combat_island_count": len(islands),
        "combat_island_policy": "coarse envelopes require verified hostile anchors and are split at sustained reload/search gaps",
        "quiet_gap_is_hard_break": True,
    })
    setattr(timeline, "_combat_state_diagnostics", diagnostics)
    return timeline


def plan_island_failures(plan: Any, timeline: Any, config: dict[str, Any]) -> list[str]:
    hard = np.asarray(timeline.signals.get("combat_island_gap", []), dtype=np.float32)
    if hard.size == 0:
        return ["combat-island gap signal missing"]
    failures: list[str] = []
    fps = float(timeline.fps)
    for index, segment in enumerate(getattr(plan, "segments", ()), 1):
        if str(getattr(segment, "reason", "")) == "finishing_move_open_hero":
            continue
        i0 = max(0, int(math.floor(float(segment.start) * fps)))
        i1 = min(len(hard), int(math.ceil(float(segment.end) * fps)))
        if i1 <= i0:
            failures.append(f"segment {index} has no combat-island samples")
        elif float(np.max(hard[i0:i1])) > 0.5:
            failures.append(f"segment {index} crosses a sustained reload/search combat-island break")
    return failures


def verified_island_montage(plan: Any, timeline: Any, config: dict[str, Any], combat: Any) -> bool:
    if str(getattr(plan, "story_type", "")) != "semantic_montage":
        return False
    segments = tuple(getattr(plan, "segments", ()))
    if len(segments) < 2 or any(str(getattr(seg, "reason", "")) != "semantic_montage_moment" for seg in segments):
        return False
    if combat.plan_combat_state_failures(plan, timeline, config):
        return False
    if plan_island_failures(plan, timeline, config):
        return False
    return not combat.continuation_failures(plan, config)


def install(legacy: Any, combat: Any) -> None:
    original_analyze = legacy.analyze_source
    original_diagnose = legacy.diagnose_source

    def analyze_source(source: Any, config: dict[str, Any]) -> Any:
        return refine_timeline(original_analyze(source, config), config, legacy, combat)

    def diagnose_source(timeline: Any, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> dict[str, Any]:
        result = original_diagnose(timeline, config, excluded, source_key)
        result["combat_state_verifier"] = dict(getattr(timeline, "_combat_state_diagnostics", {}))
        result["combat_state_verifier"]["qualified_chain_policy"] = "verified combat islands; sustained reload/search breaks are unbridgeable"
        return result

    legacy.analyze_source = analyze_source
    legacy.diagnose_source = diagnose_source
    legacy.refined._candidate_engagement_chains = lambda timeline, config: island_candidate_chains(timeline, config, legacy)


def self_test() -> None:
    fps = 10.0
    times = (np.arange(120, dtype=np.float32) + 0.5) / fps
    zeros = np.zeros(120, dtype=np.float32)
    combat_signal = np.full(120, 0.70, dtype=np.float32)
    traversal = zeros.copy()
    recovery = zeros.copy()
    combat_signal[40:52] = 0.10
    traversal[40:52] = 0.90
    signals = {
        "combat": combat_signal,
        "outcome": zeros.copy(),
        "impact": zeros.copy(),
        "traversal": traversal,
        "recovery": recovery,
    }
    event1 = SimpleNamespace(time=2.0, kinds=("combat_burst",), confidence=0.8, evidence={"combat": 0.8, "audio_transient": 0.8, "center_motion": 0.8})
    event2 = SimpleNamespace(time=7.0, kinds=("outcome_like",), confidence=0.85, evidence={"combat": 0.7, "outcome": 0.8})
    strict1 = SimpleNamespace(start=1.7, end=2.4, shot_index=0, confidence=0.8, events=(event1,))
    strict2 = SimpleNamespace(start=6.7, end=7.4, shot_index=0, confidence=0.85, events=(event2,))
    coarse = SimpleNamespace(start=1.0, end=9.0, shot_index=0, confidence=0.8, events=(event1, event2))
    timeline = SimpleNamespace(
        fps=fps,
        times=times,
        signals=signals,
        engagements=(strict1, strict2),
        consolidated_events=(event1, event2),
        shots=(SimpleNamespace(start=0.0, end=12.0),),
        base=SimpleNamespace(),
        finishing_moves=(),
    )
    dummy_core = SimpleNamespace(
        cluster_engagements=lambda *args, **kwargs: (coarse,),
        _shot_index=lambda shots, value: 0,
    )
    dummy_refined = SimpleNamespace(core=dummy_core, _last_payoff=lambda chain: event2 if any(event2 in e.events for e in chain) else None)
    dummy_legacy = SimpleNamespace(refined=dummy_refined, Engagement=lambda **kwargs: SimpleNamespace(**kwargs))
    dummy_combat = SimpleNamespace(
        _event_hostile_decision=lambda event, config: SimpleNamespace(hostile=True, score=event.confidence),
        _verified_hostile_signal=lambda tl, engs: np.zeros(len(tl.times), dtype=np.float32),
    )
    cfg = {"combat_state_verifier": {"maximum_continuity_break_run_seconds": 0.55, "maximum_combat_island_quiet_run_seconds": 0.75}}
    islands = build_combat_islands(timeline, cfg, dummy_legacy, dummy_combat)
    if len(islands) != 2:
        raise AssertionError(f"reload/search break did not split combat island: {[(x.start, x.end) for x in islands]}")
    if not (islands[0].end <= 4.0 + 1e-6 and islands[1].start >= 5.2 - 1e-6):
        raise AssertionError("combat islands retained the hard reload/search break")

    timeline.signals["combat"][:] = 0.70
    timeline.signals["traversal"][:] = 0.0
    merged = build_combat_islands(timeline, cfg, dummy_legacy, dummy_combat)
    if len(merged) != 1 or merged[0].end - merged[0].start < 7.9:
        raise AssertionError("continuous verified combat context collapsed instead of forming one island")
    print("MW4 combat-island construction self-test: PASS")
