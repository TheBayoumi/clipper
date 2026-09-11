from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import mw4_semantic_gameplay_v3_1 as core

# Re-export the V3.1 data contracts so the renderer can use this module directly.
ShotSpan = core.ShotSpan
ConsolidatedEvent = core.ConsolidatedEvent
Engagement = core.Engagement
FinishingMoveSpan = core.FinishingMoveSpan
EditSegment = core.EditSegment
SemanticPlanV31 = core.SemanticPlanV31
SemanticTimelineV31 = core.SemanticTimelineV31
PAYOFF_KINDS = core.PAYOFF_KINDS


def _event_has_payoff(event: ConsolidatedEvent) -> bool:
    return any(kind in PAYOFF_KINDS for kind in event.kinds)


def _engagement_has_payoff(engagement: Engagement) -> bool:
    return any(_event_has_payoff(event) for event in engagement.events)


def _last_payoff(chain: tuple[Engagement, ...]) -> ConsolidatedEvent | None:
    payoffs = [event for engagement in chain for event in engagement.events if _event_has_payoff(event)]
    return payoffs[-1] if payoffs else None


def _scene_cut_times(source: Path, config: dict[str, Any]) -> list[float]:
    """Detect actual source-stringout edits, rather than treating gameplay motion as a cut.

    FFmpeg's scene score runs on a small 6 fps analysis stream. This is intentionally
    separate from combat/activity normalization so fast camera movement is much less
    likely to fragment a gameplay sequence into fake shots.
    """
    cfg = config.get("semantic_analysis", {})
    threshold = float(cfg.get("scene_change_threshold", 0.34))
    analysis_fps = float(cfg.get("scene_change_fps", 6.0))
    vf = (
        f"fps={analysis_fps},scale=320:-1:flags=fast_bilinear,"
        f"select='gt(scene,{threshold})',showinfo"
    )
    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(source),
            "-an", "-vf", vf, "-f", "null", "-",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    times: list[float] = []
    for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr):
        value = float(match.group(1))
        if value > 0.35 and all(abs(value - prior) >= 0.70 for prior in times):
            times.append(value)
    return sorted(times)


def _build_shots(source: Path, duration: float, config: dict[str, Any]) -> tuple[ShotSpan, ...]:
    cuts = _scene_cut_times(source, config)
    guard = float(config.get("semantic_analysis", {}).get("scene_change_guard_seconds", 0.12))
    boundaries = [0.0] + cuts + [duration]
    shots: list[ShotSpan] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        start = left if index == 0 else min(right, left + guard)
        end = right if index == len(boundaries) - 2 else max(start, right - guard)
        if end - start >= 2.0:
            shots.append(ShotSpan(round(start, 3), round(end, 3)))
    if not shots:
        shots.append(ShotSpan(0.0, duration))
    return tuple(shots)


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimelineV31:
    base_timeline = core.base.analyze_source(source, config)
    shots = _build_shots(source, base_timeline.duration, config)
    consolidated = core.consolidate_events(base_timeline, shots)
    engagements = core.cluster_engagements(base_timeline, shots, consolidated, config)
    finishing = core.detect_finishing_moves(base_timeline, shots, engagements, config)
    return SemanticTimelineV31(base_timeline, shots, consolidated, engagements, finishing)


def _candidate_engagement_chains(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
) -> list[tuple[Engagement, ...]]:
    """Construct stories from engagements, not arbitrary duration windows.

    A candidate is only emitted when the semantic sequence can naturally support the
    campaign's >=10 s duration while still ending on a payoff or active continuation.
    We deliberately allow a longer engagement chain (up to 7 engagements) so the
    planner does not repeatedly collapse onto the same single strongest kill.
    """
    editor = config["semantic_editor"]
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 20.0))
    max_gap = float(editor.get("maximum_inter_engagement_gap_seconds", 4.0))
    max_engagements = int(editor.get("maximum_engagements_per_story", 7))
    lead = float(config["editorial"].get("finishing_move_opening_lead_seconds", 0.28))
    tail = float(editor["ending"].get("preferred_payoff_tail_seconds", 0.45))

    chains: list[tuple[Engagement, ...]] = []
    engagements = timeline.engagements
    finishing_by_shot: dict[int, list[FinishingMoveSpan]] = {}
    for span in timeline.finishing_moves:
        finishing_by_shot.setdefault(span.shot_index, []).append(span)

    for start_index, first in enumerate(engagements):
        shot = timeline.shots[first.shot_index]
        chain: list[Engagement] = []
        for candidate in engagements[start_index:]:
            if candidate.shot_index != first.shot_index:
                break
            if chain and candidate.start - chain[-1].end > max_gap:
                break
            chain.append(candidate)
            if len(chain) > max_engagements:
                break

            first_event = chain[0].events[0].time
            start = max(shot.start, first_event - 0.30)
            payoff = _last_payoff(tuple(chain))
            if payoff is not None:
                natural_end = min(shot.end, payoff.time + tail)
                natural_duration = natural_end - start
                if minimum <= natural_duration <= maximum:
                    chains.append(tuple(chain))
            else:
                # Active cliffhanger is eligible only when a genuinely extended
                # engagement arc already reaches campaign duration.
                natural_end = min(shot.end, chain[-1].end)
                if minimum <= natural_end - start <= maximum:
                    chains.append(tuple(chain))

    # Dedicated finishing-move stories: the move itself must be the opening hero
    # event, then the story is allowed to continue through later engagements until
    # a natural >=10 s ending/payoff is available.
    for span in timeline.finishing_moves:
        same_shot = [eng for eng in engagements if eng.shot_index == span.shot_index and eng.end >= span.start]
        if not same_shot:
            continue
        for index, engagement in enumerate(same_shot):
            if engagement.start - 0.25 <= span.start <= engagement.end + 0.25:
                same_shot = same_shot[index:]
                break
        chain: list[Engagement] = []
        shot = timeline.shots[span.shot_index]
        start = max(shot.start, span.start - lead)
        for engagement in same_shot:
            if chain and engagement.start - chain[-1].end > max_gap:
                break
            chain.append(engagement)
            if len(chain) > max_engagements:
                break
            payoff = _last_payoff(tuple(chain))
            if payoff is None:
                continue
            end = min(shot.end, payoff.time + tail)
            if minimum <= end - start <= maximum:
                chains.append(tuple(chain))

    unique: list[tuple[Engagement, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for chain in chains:
        key = tuple(round(item.start, 2) for item in chain)
        if key not in seen:
            seen.add(key)
            unique.append(chain)
    return unique


def _find_finishing_for_chain(
    timeline: SemanticTimelineV31,
    chain: tuple[Engagement, ...],
) -> FinishingMoveSpan | None:
    if not chain:
        return None
    first = chain[0]
    # A finishing move only qualifies if it belongs to the FIRST engagement. If it
    # appears later, this chain remains a normal story; a separate finishing-move
    # candidate will start at that move instead.
    candidates = [
        span for span in timeline.finishing_moves
        if span.shot_index == first.shot_index
        and first.start - 0.25 <= span.start <= first.end + 0.25
    ]
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def _boundary_for_chain(
    timeline: SemanticTimelineV31,
    chain: tuple[Engagement, ...],
    config: dict[str, Any],
    finishing: FinishingMoveSpan | None,
) -> tuple[float, float] | None:
    editor = config["semantic_editor"]
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 20.0))
    shot = timeline.shots[chain[0].shot_index]

    if finishing is not None:
        lead = float(config["editorial"].get("finishing_move_opening_lead_seconds", 0.28))
        start = max(shot.start, finishing.start - lead)
    else:
        start = max(shot.start, chain[0].events[0].time - 0.30)

    payoff = _last_payoff(chain)
    tail = float(editor["ending"].get("preferred_payoff_tail_seconds", 0.45))
    if payoff is not None:
        end = min(shot.end, payoff.time + tail)
    else:
        end = min(shot.end, chain[-1].end)

    # Critical V3.1 fix: do NOT pad the tail with dead footage just to hit 10 s.
    # If the semantic story does not naturally reach campaign duration, reject it.
    duration = end - start
    if duration < minimum or duration > maximum:
        return None
    return round(start, 3), round(end, 3)


def _quality_metrics(
    timeline: SemanticTimelineV31,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    residual_run: float,
    config: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    sig = timeline.signals
    fps = timeline.fps
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * fps)))
    interest = sig["interest"][i0:i1]
    if interest.size == 0:
        return (0.0,) * 7

    quarters = np.array_split(interest, 4)
    weakest = min(float(np.mean(part)) for part in quarters if len(part))
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    low_fraction = float(np.mean(interest < low_threshold))

    first_event = finishing.start if finishing is not None else engagements[0].events[0].time
    delay = max(0.0, first_event - start)
    opening_mean = core._mean(sig["interest"], timeline, start, min(end, start + 1.25))
    opening_slice = sig["interest"][i0:min(i1, i0 + max(1, int(round(1.1 * fps))))]
    opening_dead = core._longest_true_run(opening_slice < low_threshold, fps)
    opening_quality = np.clip(
        0.43 * opening_mean
        + 0.34 * (1.0 - min(1.0, delay / 0.90))
        + 0.23 * (1.0 - min(1.0, opening_dead / 0.50)),
        0,
        1,
    )
    if finishing is not None:
        opening_quality = min(1.0, float(opening_quality) + 0.18 * finishing.confidence)

    payoff_events = [event for engagement in engagements for event in engagement.events if _event_has_payoff(event)]
    last_payoff_time = payoff_events[-1].time if payoff_events else None
    if finishing is not None and (last_payoff_time is None or finishing.payoff > last_payoff_time):
        last_payoff_time = finishing.payoff

    if last_payoff_time is not None:
        tail = max(0.0, end - last_payoff_time)
        tail_quality = float(np.clip(1.0 - abs(tail - 0.45) / 0.75, 0, 1))
        ending_interest = core._mean(sig["interest"], timeline, max(start, end - 1.0), end)
        ending_quality = 0.66 * tail_quality + 0.34 * ending_interest
    else:
        active = max(
            core._mean(sig["combat"], timeline, max(start, end - 0.8), end),
            core._mean(sig["contact"], timeline, max(start, end - 0.8), end),
        )
        ending_quality = 0.72 * active + 0.28 * core._mean(sig["interest"], timeline, max(start, end - 1.0), end)

    payoff_strength = max(
        [event.confidence for event in payoff_events]
        + ([min(1.0, finishing.confidence + 0.12)] if finishing is not None else [0.0])
    )
    coherence = np.clip(
        0.54 * float(np.mean([eng.confidence for eng in engagements]))
        + 0.26 * min(1.0, len(engagements) / 4.0)
        + 0.20 * (1.0 - min(1.0, low_fraction / 0.45)),
        0,
        1,
    )
    retention = np.clip(
        0.27 * float(np.mean(interest))
        + 0.18 * weakest
        + 0.22 * opening_quality
        + 0.18 * ending_quality
        + 0.09 * coherence
        + 0.06 * (1.0 - min(1.0, residual_run / 0.90)),
        0,
        1,
    )
    payoff_quality = np.clip(0.72 * payoff_strength + 0.28 * ending_quality, 0, 1)
    return (
        float(opening_quality), float(ending_quality), float(coherence),
        float(retention), float(payoff_quality), weakest, low_fraction,
    )


def diagnose_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> dict[str, Any]:
    chains = _candidate_engagement_chains(timeline, config)
    counts: dict[str, int] = {
        "chains": len(chains),
        "boundary_reject": 0,
        "excluded_reject": 0,
        "edit_reject": 0,
        "opening_reject": 0,
        "ending_reject": 0,
        "retention_reject": 0,
        "payoff_reject": 0,
        "weakest_reject": 0,
        "low_fraction_reject": 0,
        "pass": 0,
    }
    samples: list[dict[str, Any]] = []
    for chain in chains:
        finishing = _find_finishing_for_chain(timeline, chain)
        boundary = _boundary_for_chain(timeline, chain, config, finishing)
        record: dict[str, Any] = {
            "engagements": [[item.start, item.end] for item in chain],
            "finishing": asdict(finishing) if finishing is not None else None,
        }
        if boundary is None:
            counts["boundary_reject"] += 1
            record["reject"] = "boundary"
            samples.append(record)
            continue
        start, end = boundary
        record["boundary"] = [start, end]
        if core._intersects_excluded(start, end, excluded):
            counts["excluded_reject"] += 1
            record["reject"] = "excluded"
            samples.append(record)
            continue
        edited = core._editable_segments(timeline, start, end, chain, finishing, config)
        if edited is None:
            counts["edit_reject"] += 1
            record["reject"] = "edit/no-dull"
            samples.append(record)
            continue
        segments, _, residual = edited
        opening, ending, coherence, retention, payoff, weakest, low_fraction = _quality_metrics(
            timeline, start, end, chain, finishing, residual, config
        )
        record["metrics"] = {
            "opening": round(opening, 4), "ending": round(ending, 4),
            "retention": round(retention, 4), "payoff": round(payoff, 4),
            "weakest": round(weakest, 4), "low_fraction": round(low_fraction, 4),
            "residual": round(residual, 3),
        }
        editor = config["semantic_editor"]
        checks = [
            (opening >= float(editor["opening"].get("minimum_quality", 0.42)), "opening"),
            (ending >= float(editor["ending"].get("minimum_quality", 0.40)), "ending"),
            (retention >= float(config["performance_targets"].get("retention_quality_min", 0.36)), "retention"),
            (payoff >= float(config["performance_targets"].get("payoff_quality_min", 0.34)), "payoff"),
            (weakest >= float(editor["dull"].get("minimum_weak_quarter_interest", 0.30)), "weakest"),
            (low_fraction <= float(editor["dull"].get("maximum_low_interest_fraction", 0.38)), "low_fraction"),
        ]
        failed = next((name for passed, name in checks if not passed), None)
        if failed:
            counts[f"{failed}_reject"] += 1
            record["reject"] = failed
        else:
            counts["pass"] += 1
            record["reject"] = None
        samples.append(record)
    return {
        "shot_count": len(timeline.shots),
        "engagement_count": len(timeline.engagements),
        "finishing_move_like_count": len(timeline.finishing_moves),
        "shot_spans": [asdict(item) for item in timeline.shots],
        "counts": counts,
        "samples": samples[:80],
    }


def build_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlanV31]:
    plans: list[SemanticPlanV31] = []
    for chain in _candidate_engagement_chains(timeline, config):
        finishing = _find_finishing_for_chain(timeline, chain)
        boundary = _boundary_for_chain(timeline, chain, config, finishing)
        if boundary is None:
            continue
        start, end = boundary
        if finishing is not None and finishing.start - start > 0.48:
            continue
        if core._intersects_excluded(start, end, excluded):
            continue
        edited = core._editable_segments(timeline, start, end, chain, finishing, config)
        if edited is None:
            continue
        segments, edit_reasons, residual = edited
        output_duration = sum((segment.end - segment.start) / segment.speed for segment in segments)
        opening, ending, coherence, retention, payoff, weakest, low_fraction = _quality_metrics(
            timeline, start, end, chain, finishing, residual, config
        )
        editor = config["semantic_editor"]
        if opening < float(editor["opening"].get("minimum_quality", 0.42)):
            continue
        if ending < float(editor["ending"].get("minimum_quality", 0.40)):
            continue
        if retention < float(config["performance_targets"].get("retention_quality_min", 0.36)):
            continue
        if payoff < float(config["performance_targets"].get("payoff_quality_min", 0.34)):
            continue
        if weakest < float(editor["dull"].get("minimum_weak_quarter_interest", 0.30)):
            continue
        if low_fraction > float(editor["dull"].get("maximum_low_interest_fraction", 0.38)):
            continue

        story, profile, effect_events, route_reasons = core._route(timeline, chain, finishing)
        weights = editor["selection"]
        score = (
            float(weights.get("retention_quality_weight", 0.24)) * retention
            + float(weights.get("payoff_quality_weight", 0.17)) * payoff
            + float(weights.get("opening_weight", 0.20)) * opening
            + float(weights.get("ending_weight", 0.17)) * ending
            + float(weights.get("story_coherence_weight", 0.12)) * coherence
            + float(weights.get("weakest_section_weight", 0.10)) * weakest
        )
        if finishing is not None:
            score += float(weights.get("finishing_move_bonus", 0.14))
        plans.append(
            SemanticPlanV31(
                start=start,
                end=end,
                raw_duration=round(end - start, 3),
                output_duration=round(output_duration, 3),
                score=round(float(score), 5),
                retention_quality=round(retention, 4),
                payoff_quality=round(payoff, 4),
                opening_quality=round(opening, 4),
                ending_quality=round(ending, 4),
                story_coherence=round(coherence, 4),
                weakest_quarter_interest=round(weakest, 4),
                low_interest_fraction=round(low_fraction, 4),
                max_unexplained_low_interest_run_seconds=round(residual, 3),
                story_type=story,
                effect_profile=profile,
                segments=segments,
                effect_events=effect_events,
                engagements=chain,
                finishing_move=finishing,
                editorial_reasons=tuple(route_reasons) + tuple(edit_reasons),
            )
        )

    # One semantic story per core opening engagement. Different duration/ending
    # permutations around the same opening no longer dominate the ranking.
    plans.sort(key=lambda item: item.score, reverse=True)
    unique: list[SemanticPlanV31] = []
    for plan in plans:
        anchor = plan.finishing_move.start if plan.finishing_move is not None else plan.engagements[0].events[0].time
        if any(
            abs(anchor - (prior.finishing_move.start if prior.finishing_move is not None else prior.engagements[0].events[0].time)) < 2.4
            and plan.engagements[0].shot_index == prior.engagements[0].shot_index
            for prior in unique
        ):
            continue
        unique.append(plan)
    return unique


def select_plans(plans: list[SemanticPlanV31], config: dict[str, Any]) -> list[SemanticPlanV31]:
    return core.select_plans(plans, config)


def _self_test() -> None:
    assert _event_has_payoff(ConsolidatedEvent(1.0, ("outcome_like",), 0.8, {}))
    assert not _event_has_payoff(ConsolidatedEvent(1.0, ("contact",), 0.8, {}))
    print(json.dumps({
        "self_test": "PASS",
        "candidate_mode": "engagement_driven_natural_boundaries",
        "finishing_move_must_open": True,
        "dead_tail_padding": False,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    parser.error("Use this module through mw4_fullframe_retention_v3_1.py or mw4_v3_1_shadow_preview.py")


if __name__ == "__main__":
    main()
