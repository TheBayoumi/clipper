from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


PAYOFF_KINDS = {"outcome_like", "impact"}


@dataclass(frozen=True)
class HostileDecision:
    hostile: bool
    score: float
    reason: str


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("combat_state_verifier", {})


def _event_hostile_decision(event: Any, config: dict[str, Any]) -> HostileDecision:
    """Fail-closed hostile classification for a consolidated semantic event.

    Generic contact/motion is never hostile evidence. A hostile anchor requires a
    payoff-like event with combat support, or an unusually strong combat burst with
    audio/center-motion support. Ambiguous actors therefore remain unknown rather
    than silently becoming enemies.
    """
    cfg = _cfg(config)
    kinds = set(getattr(event, "kinds", ()) or ())
    evidence = dict(getattr(event, "evidence", {}) or {})
    confidence = _f(getattr(event, "confidence", 0.0))
    combat = _f(evidence.get("combat"))
    outcome = _f(evidence.get("outcome"))
    impact = _f(evidence.get("impact"))
    audio_transient = _f(evidence.get("audio_transient"))
    center_motion = _f(evidence.get("center_motion"))

    min_score = _f(cfg.get("minimum_hostile_event_score", 0.64), 0.64)
    outcome_min = _f(cfg.get("outcome_minimum", 0.52), 0.52)
    payoff_combat_min = _f(cfg.get("payoff_combat_minimum", 0.42), 0.42)
    impact_min = _f(cfg.get("impact_minimum", 0.64), 0.64)
    impact_combat_min = _f(cfg.get("impact_combat_minimum", 0.55), 0.55)
    burst_min = _f(cfg.get("strong_combat_minimum", 0.70), 0.70)
    burst_audio_min = _f(cfg.get("strong_combat_audio_transient_minimum", 0.62), 0.62)
    burst_center_min = _f(cfg.get("strong_combat_center_motion_minimum", 0.40), 0.40)

    if "outcome_like" in kinds and outcome >= outcome_min and max(combat, impact) >= payoff_combat_min:
        score = min(1.0, 0.48 * max(confidence, outcome) + 0.34 * max(combat, impact) + 0.18 * max(audio_transient, center_motion))
        return HostileDecision(score >= min_score, score, "combat_supported_outcome")

    if "impact" in kinds and impact >= impact_min and combat >= impact_combat_min:
        corroboration = max(audio_transient, center_motion)
        score = min(1.0, 0.44 * max(confidence, impact) + 0.38 * combat + 0.18 * corroboration)
        return HostileDecision(score >= min_score, score, "combat_supported_impact")

    if (
        "combat_burst" in kinds
        and combat >= burst_min
        and audio_transient >= burst_audio_min
        and center_motion >= burst_center_min
    ):
        score = min(1.0, 0.46 * max(confidence, combat) + 0.29 * audio_transient + 0.25 * center_motion)
        return HostileDecision(score >= min_score, score, "strong_combat_burst")

    return HostileDecision(False, 0.0, "unknown_or_non_hostile")


def _signal_slice(timeline: Any, name: str, start: float, end: float) -> np.ndarray:
    signal = np.asarray(timeline.signals[name], dtype=np.float32)
    i0 = max(0, int(math.floor(start * timeline.fps)))
    i1 = min(len(signal), int(math.ceil(end * timeline.fps)))
    return signal[i0:i1]


def _longest_true_run(mask: np.ndarray, fps: float) -> float:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / fps


def _bridge_has_break(timeline: Any, left: float, right: float, config: dict[str, Any]) -> bool:
    if right <= left:
        return False
    cfg = _cfg(config)
    max_break = _f(cfg.get("maximum_continuity_break_run_seconds", 0.55), 0.55)
    traversal_threshold = _f(cfg.get("traversal_break_threshold", 0.60), 0.60)
    recovery_threshold = _f(cfg.get("recovery_break_threshold", 0.60), 0.60)
    combat_ceiling = _f(cfg.get("break_combat_ceiling", 0.50), 0.50)

    traversal = _signal_slice(timeline, "traversal", left, right)
    recovery = _signal_slice(timeline, "recovery", left, right)
    combat = _signal_slice(timeline, "combat", left, right)
    if min(len(traversal), len(recovery), len(combat)) == 0:
        return False
    n = min(len(traversal), len(recovery), len(combat))
    break_mask = (
        ((traversal[:n] >= traversal_threshold) | (recovery[:n] >= recovery_threshold))
        & (combat[:n] <= combat_ceiling)
    )
    return _longest_true_run(break_mask, float(timeline.fps)) > max_break


def _verified_engagements(timeline: Any, config: dict[str, Any], legacy: Any) -> tuple[Any, ...]:
    cfg = _cfg(config)
    max_anchor_gap = _f(cfg.get("maximum_hostile_anchor_gap_seconds", 1.35), 1.35)
    pre = _f(cfg.get("engagement_pre_context_seconds", 0.35), 0.35)
    post = _f(cfg.get("engagement_post_context_seconds", 0.45), 0.45)

    anchors: list[tuple[Any, HostileDecision]] = []
    for event in timeline.consolidated_events:
        decision = _event_hostile_decision(event, config)
        if decision.hostile:
            anchors.append((event, decision))

    groups: list[list[tuple[Any, HostileDecision]]] = []
    for item in anchors:
        event, _ = item
        if not groups:
            groups.append([item])
            continue
        prior_event = groups[-1][-1][0]
        same_shot = legacy.refined.core._shot_index(timeline.shots, event.time) == legacy.refined.core._shot_index(timeline.shots, prior_event.time)
        gap = float(event.time) - float(prior_event.time)
        if same_shot and gap <= max_anchor_gap and not _bridge_has_break(timeline, prior_event.time, event.time, config):
            groups[-1].append(item)
        else:
            groups.append([item])

    engagements: list[Any] = []
    for group in groups:
        events = tuple(item[0] for item in group)
        scores = [item[1].score for item in group]
        shot_index = legacy.refined.core._shot_index(timeline.shots, events[0].time)
        shot = timeline.shots[shot_index]
        start = max(float(shot.start), float(events[0].time) - pre)
        end = min(float(shot.end), float(events[-1].time) + post)
        if end <= start:
            continue
        engagements.append(
            legacy.Engagement(
                start=round(start, 3),
                end=round(end, 3),
                shot_index=int(shot_index),
                confidence=round(float(np.mean(scores)), 4),
                events=events,
            )
        )
    return tuple(engagements)


def _verified_hostile_signal(timeline: Any, engagements: tuple[Any, ...]) -> np.ndarray:
    signal = np.zeros(len(timeline.times), dtype=np.float32)
    if signal.size == 0:
        return signal
    for engagement in engagements:
        for event in engagement.events:
            idx = max(0, min(len(signal) - 1, int(round(float(event.time) * timeline.fps - 0.5))))
            signal[idx] = max(signal[idx], float(engagement.confidence))
    return signal


def _combat_break_signal(timeline: Any, config: dict[str, Any]) -> np.ndarray:
    cfg = _cfg(config)
    traversal = np.asarray(timeline.signals["traversal"], dtype=np.float32)
    recovery = np.asarray(timeline.signals["recovery"], dtype=np.float32)
    combat = np.asarray(timeline.signals["combat"], dtype=np.float32)
    return (
        ((traversal >= _f(cfg.get("traversal_break_threshold", 0.60), 0.60))
         | (recovery >= _f(cfg.get("recovery_break_threshold", 0.60), 0.60)))
        & (combat <= _f(cfg.get("break_combat_ceiling", 0.50), 0.50))
    ).astype(np.float32)


def refine_timeline(timeline: Any, config: dict[str, Any], legacy: Any) -> Any:
    if not bool(_cfg(config).get("enabled", True)):
        raise RuntimeError("combat-state verifier may not be disabled for MW4 V3.1 qualification")
    original_count = len(timeline.engagements)
    verified = _verified_engagements(timeline, config, legacy)
    timeline.engagements = verified
    timeline.signals["verified_hostile"] = _verified_hostile_signal(timeline, verified)
    timeline.signals["combat_break"] = _combat_break_signal(timeline, config)
    setattr(
        timeline,
        "_combat_state_diagnostics",
        {
            "original_engagement_count": original_count,
            "verified_engagement_count": len(verified),
            "contact_only_can_anchor_hostile": False,
            "unknown_actor_defaults_to_hostile": False,
            "verifier_policy": "payoff-or-strong-combat-confirmation; contact-only is unknown",
        },
    )
    return timeline


def verified_candidate_chains(timeline: Any, config: dict[str, Any], legacy: Any) -> list[tuple[Any, ...]]:
    """Build only continuity-proven chains; do not bridge reload/search gaps."""
    editor = config["semantic_editor"]
    cfg = _cfg(config)
    minimum = _f(editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(editor.get("maximum_output_seconds", 20.0), 20.0)
    max_gap = _f(cfg.get("maximum_verified_inter_engagement_gap_seconds", 1.25), 1.25)
    max_engagements = int(editor.get("maximum_engagements_per_story", 7))
    tail = _f(editor["ending"].get("preferred_payoff_tail_seconds", 0.45), 0.45)
    lead = _f(config["editorial"].get("finishing_move_opening_lead_seconds", 0.28), 0.28)

    chains: list[tuple[Any, ...]] = []
    engagements = timeline.engagements
    for start_index, first in enumerate(engagements):
        shot = timeline.shots[first.shot_index]
        chain: list[Any] = []
        for candidate in engagements[start_index:]:
            if candidate.shot_index != first.shot_index:
                break
            if chain:
                gap = float(candidate.start) - float(chain[-1].end)
                if gap > max_gap or _bridge_has_break(timeline, chain[-1].end, candidate.start, config):
                    break
            chain.append(candidate)
            if len(chain) > max_engagements:
                break
            first_event = chain[0].events[0].time
            start = max(float(shot.start), float(first_event) - 0.30)
            payoff = legacy.refined._last_payoff(tuple(chain))
            if payoff is not None:
                end = min(float(shot.end), float(payoff.time) + tail)
            else:
                end = min(float(shot.end), float(chain[-1].end))
            if minimum <= end - start <= maximum:
                chains.append(tuple(chain))

    for span in timeline.finishing_moves:
        same_shot = [e for e in engagements if e.shot_index == span.shot_index and e.end >= span.start]
        if not same_shot:
            continue
        shot = timeline.shots[span.shot_index]
        start = max(float(shot.start), float(span.start) - lead)
        chain: list[Any] = []
        for engagement in same_shot:
            if chain:
                gap = float(engagement.start) - float(chain[-1].end)
                if gap > max_gap or _bridge_has_break(timeline, chain[-1].end, engagement.start, config):
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


def continuation_failures(plan: Any, config: dict[str, Any]) -> list[str]:
    """Require the body of a Finishing Move clip to qualify independently."""
    cfg = _cfg(config).get("finishing_continuation", {})
    failures: list[str] = []
    min_retention = max(
        _f(config.get("performance_targets", {}).get("retention_quality_min", 0.36), 0.36),
        _f(cfg.get("minimum_retention_quality", 0.40), 0.40),
    )
    min_payoff = max(
        _f(config.get("performance_targets", {}).get("payoff_quality_min", 0.34), 0.34),
        _f(cfg.get("minimum_payoff_quality", 0.40), 0.40),
    )
    min_ending = _f(cfg.get("minimum_ending_quality", 0.45), 0.45)
    min_weakest = _f(cfg.get("minimum_weakest_quarter_interest", 0.30), 0.30)
    max_residual = _f(cfg.get("maximum_unexplained_low_interest_run_seconds", 0.75), 0.75)

    if _f(plan.retention_quality) < min_retention:
        failures.append("continuation retention below independent floor")
    if _f(plan.payoff_quality) < min_payoff:
        failures.append("continuation payoff below independent floor")
    if _f(plan.ending_quality) < min_ending:
        failures.append("continuation ending below independent floor")
    if _f(plan.weakest_quarter_interest) < min_weakest:
        failures.append("continuation weakest quarter below independent floor")
    if _f(plan.max_unexplained_low_interest_run_seconds) > max_residual:
        failures.append("continuation contains excessive unexplained inactivity")

    hostile_events = [
        event
        for engagement in getattr(plan, "engagements", ())
        for event in getattr(engagement, "events", ())
    ]
    has_verified_payoff = any(
        any(kind in PAYOFF_KINDS for kind in getattr(event, "kinds", ()))
        for event in hostile_events
    )
    minimum_sustained_anchors = int(cfg.get("minimum_verified_hostile_anchors_without_payoff", 2))
    minimum_sustained_retention = _f(cfg.get("minimum_retention_without_payoff", 0.50), 0.50)
    if not has_verified_payoff and (
        len(hostile_events) < minimum_sustained_anchors
        or _f(plan.retention_quality) < minimum_sustained_retention
    ):
        failures.append("continuation has neither a verified hostile payoff nor sustained independently strong hostile anchors")
    if any(getattr(segment, "reason", "") == "compressed_traversal" for segment in getattr(plan, "segments", ())):
        failures.append("continuation relies on compressed traversal")
    return failures


def install(legacy: Any) -> None:
    original_analyze = legacy.analyze_source
    original_diagnose = legacy.diagnose_source

    def analyze_source(source: Any, config: dict[str, Any]) -> Any:
        return refine_timeline(original_analyze(source, config), config, legacy)

    def diagnose_source(timeline: Any, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> dict[str, Any]:
        result = original_diagnose(timeline, config, excluded, source_key)
        result["combat_state_verifier"] = dict(getattr(timeline, "_combat_state_diagnostics", {}))
        result["combat_state_verifier"]["qualified_chain_policy"] = "verified hostile anchors; continuity breaks terminate chains"
        return result

    legacy.analyze_source = analyze_source
    legacy.diagnose_source = diagnose_source
    legacy.refined._candidate_engagement_chains = lambda timeline, config: verified_candidate_chains(timeline, config, legacy)


def self_test() -> None:
    cfg = {
        "combat_state_verifier": {
            "enabled": True,
            "minimum_hostile_event_score": 0.64,
            "outcome_minimum": 0.52,
            "payoff_combat_minimum": 0.42,
            "impact_minimum": 0.64,
            "impact_combat_minimum": 0.55,
            "strong_combat_minimum": 0.70,
            "strong_combat_audio_transient_minimum": 0.62,
            "strong_combat_center_motion_minimum": 0.40,
            "finishing_continuation": {
                "minimum_retention_quality": 0.40,
                "minimum_payoff_quality": 0.40,
                "minimum_ending_quality": 0.45,
                "minimum_weakest_quarter_interest": 0.30,
                "maximum_unexplained_low_interest_run_seconds": 0.75,
            },
        },
        "performance_targets": {"retention_quality_min": 0.36, "payoff_quality_min": 0.34},
    }
    contact_only = SimpleNamespace(kinds=("contact",), confidence=0.90, evidence={"contact": 0.90, "center_motion": 1.0, "hud_change": 1.0})
    if _event_hostile_decision(contact_only, cfg).hostile:
        raise AssertionError("contact-only teammate-like motion became hostile")
    weak_burst = SimpleNamespace(kinds=("combat_burst", "contact"), confidence=0.73, evidence={"combat": 0.63, "audio_transient": 0.90, "center_motion": 1.0})
    if _event_hostile_decision(weak_burst, cfg).hostile:
        raise AssertionError("weak combat burst became hostile")
    payoff = SimpleNamespace(kinds=("combat_burst", "contact", "outcome_like"), confidence=0.83, evidence={"combat": 0.71, "outcome": 0.67, "audio_transient": 1.0, "center_motion": 0.91})
    if not _event_hostile_decision(payoff, cfg).hostile:
        raise AssertionError("combat-supported payoff was rejected")
    strong_burst = SimpleNamespace(kinds=("combat_burst",), confidence=0.80, evidence={"combat": 0.78, "audio_transient": 0.75, "center_motion": 0.80})
    if not _event_hostile_decision(strong_burst, cfg).hostile:
        raise AssertionError("strong combat burst was rejected")

    poor = SimpleNamespace(
        retention_quality=0.53,
        payoff_quality=0.1603,
        ending_quality=0.55,
        weakest_quarter_interest=0.33,
        max_unexplained_low_interest_run_seconds=0.33,
        engagements=(),
        segments=(SimpleNamespace(reason="keep"),),
    )
    if not continuation_failures(poor, cfg):
        raise AssertionError("weak sustained-pressure continuation was accepted")
    good = SimpleNamespace(
        retention_quality=0.58,
        payoff_quality=0.74,
        ending_quality=0.72,
        weakest_quarter_interest=0.36,
        max_unexplained_low_interest_run_seconds=0.20,
        engagements=(SimpleNamespace(events=(SimpleNamespace(kinds=("outcome_like",)),)),),
        segments=(SimpleNamespace(reason="keep"),),
    )
    if continuation_failures(good, cfg):
        raise AssertionError("independently strong payoff continuation was rejected")
    print("MW4 combat-state verifier self-test: PASS")


def plan_combat_state_failures(plan: Any, timeline: Any, config: dict[str, Any]) -> list[str]:
    """Reject kept body footage that contains a sustained reload/search break."""
    if not hasattr(timeline, "signals") or "combat_break" not in timeline.signals:
        return []
    cfg = _cfg(config)
    max_break = _f(cfg.get("maximum_continuity_break_run_seconds", 0.55), 0.55)
    failures: list[str] = []
    verified = np.asarray(timeline.signals.get("verified_hostile", []), dtype=np.float32)
    combat_break = np.asarray(timeline.signals.get("combat_break", []), dtype=np.float32)
    for index, segment in enumerate(getattr(plan, "segments", ()), 1):
        reason = str(getattr(segment, "reason", ""))
        if reason == "finishing_move_open_hero":
            continue
        if reason == "compressed_traversal":
            failures.append(f"segment {index} compresses traversal instead of ending/splitting the combat story")
            continue
        i0 = max(0, int(math.floor(float(segment.start) * timeline.fps)))
        i1 = min(len(combat_break), int(math.ceil(float(segment.end) * timeline.fps)))
        if i1 <= i0:
            failures.append(f"segment {index} has no combat-state samples")
            continue
        break_run = _longest_true_run(combat_break[i0:i1] > 0.5, float(timeline.fps))
        if break_run > max_break:
            failures.append(f"segment {index} contains {break_run:.3f}s sustained reload/traversal break")
        if verified.size:
            v1 = min(len(verified), i1)
            if v1 <= i0 or float(np.max(verified[i0:v1])) <= 0.0:
                failures.append(f"segment {index} contains no verified hostile anchor")
    return failures
