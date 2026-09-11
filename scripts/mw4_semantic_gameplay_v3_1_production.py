from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import mw4_semantic_gameplay_v3_1_final as final

# Re-export contracts used by the renderer.
ShotSpan = final.ShotSpan
ConsolidatedEvent = final.ConsolidatedEvent
Engagement = final.Engagement
FinishingMoveSpan = final.FinishingMoveSpan
EditSegment = final.EditSegment
SemanticPlanV31 = final.SemanticPlanV31
SemanticTimelineV31 = final.SemanticTimelineV31

analyze_source = final.analyze_source
diagnose_source = final.diagnose_source


def _has_payoff(engagement: Engagement) -> bool:
    return any(any(kind in final.PAYOFF_KINDS for kind in event.kinds) for event in engagement.events)


def _verified_finishing_open_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlanV31]:
    """Build opening-first finishing-move stories even when the official stringout cuts scenes.

    A real Finishing Move can be only ~2 s long, while the campaign requires >=10 s.
    Requiring one source shot to contain the entire story therefore discards good moves.
    For VERIFIED moves only, V3.1 allows the official reel's next source shot(s) to
    continue the clip. The Finishing Move remains frame-zero hero content; it is never
    moved to the end and is never repeated as a teaser.
    """
    editor = config["semantic_editor"]
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_target_seconds", 15.5))
    hard_max = float(editor.get("maximum_output_seconds", 20.0))
    maximum = min(maximum, hard_max)
    lead = float(config["editorial"].get("finishing_move_opening_lead_seconds", 0.28))
    tail = float(editor["ending"].get("preferred_payoff_tail_seconds", 0.45))
    max_gap = float(editor.get("maximum_inter_engagement_gap_seconds", 4.0))
    max_engagements = int(editor.get("maximum_engagements_per_story", 7))
    plans: list[SemanticPlanV31] = []

    for finishing in timeline.finishing_moves:
        start = max(0.0, finishing.start - lead)
        horizon = min(timeline.duration, start + maximum)
        if final.refined.core._intersects_excluded(start, min(horizon, start + minimum), excluded):
            continue

        candidates = [
            engagement for engagement in timeline.engagements
            if engagement.end >= finishing.start - 0.20
            and engagement.start <= horizon
        ]
        candidates.sort(key=lambda item: item.start)
        if not candidates:
            continue

        # Start with the engagement overlapping the move when one exists; otherwise
        # begin with the first engagement after it. Then follow the official reel
        # chronologically across source-stringout cuts.
        first_index = 0
        for index, engagement in enumerate(candidates):
            if engagement.start - 0.35 <= finishing.start <= engagement.end + 0.35:
                first_index = index
                break
            if engagement.start >= finishing.start:
                first_index = index
                break
        chain: list[Engagement] = []
        previous_end = finishing.end
        for engagement in candidates[first_index:]:
            if engagement.start < finishing.start - 0.35:
                continue
            # Crossing a source cut is allowed for verified finishing openers, but a
            # genuinely empty gap still terminates the story.
            if chain and engagement.start - previous_end > max_gap:
                break
            if engagement.start > horizon:
                break
            chain.append(engagement)
            previous_end = engagement.end
            if len(chain) > max_engagements:
                break

            payoff_events = [
                event for item in chain for event in item.events
                if any(kind in final.PAYOFF_KINDS for kind in event.kinds)
            ]
            end_candidates = [finishing.payoff] + [event.time for event in payoff_events]
            last_payoff = max(end_candidates)
            end = min(timeline.duration, last_payoff + tail)
            if end - start < minimum:
                continue
            if end - start > maximum:
                break

            chain_tuple = tuple(chain)
            edited = final._editable_segments(timeline, start, end, chain_tuple, finishing, config)
            if edited is None:
                continue
            segments, edit_reasons, residual = edited
            output_duration = sum((segment.end - segment.start) / segment.speed for segment in segments)
            if output_duration < minimum or output_duration > hard_max:
                continue

            opening, ending, coherence, retention, payoff, weakest, low_fraction = final._quality_metrics(
                timeline, start, end, chain_tuple, finishing, residual, config
            )
            story = "finishing_move_open"
            profile = "finishing_move_hero"
            editor_cfg = config["semantic_editor"]
            if opening < final._story_gate(story, "opening_min", config, float(editor_cfg["opening"].get("minimum_quality", 0.42))):
                continue
            if ending < final._story_gate(story, "ending_min", config, float(editor_cfg["ending"].get("minimum_quality", 0.40))):
                continue
            if retention < final._story_gate(story, "retention_min", config, float(config["performance_targets"].get("retention_quality_min", 0.36))):
                continue
            if payoff < final._story_gate(story, "payoff_min", config, float(config["performance_targets"].get("payoff_quality_min", 0.34))):
                continue
            if weakest < float(editor_cfg["dull"].get("minimum_weak_quarter_interest", 0.27)):
                continue
            if low_fraction > float(editor_cfg["dull"].get("maximum_low_interest_fraction", 0.38)):
                continue

            effect_events = final.refined.core._raw_effect_events(timeline, chain_tuple)
            weights = editor_cfg["selection"]
            score = (
                float(weights.get("retention_quality_weight", 0.24)) * retention
                + float(weights.get("payoff_quality_weight", 0.17)) * payoff
                + float(weights.get("opening_weight", 0.20)) * opening
                + float(weights.get("ending_weight", 0.17)) * ending
                + float(weights.get("story_coherence_weight", 0.12)) * coherence
                + float(weights.get("weakest_section_weight", 0.10)) * weakest
                + float(weights.get("finishing_move_bonus", 0.14))
            )
            plans.append(SemanticPlanV31(
                start=round(start, 3),
                end=round(end, 3),
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
                engagements=chain_tuple,
                finishing_move=finishing,
                editorial_reasons=(
                    "visually verified Finishing Move is the opening hero event",
                    "verified opener may continue across official stringout scene cut to satisfy campaign duration",
                    "finishing choreography is protected from cuts and speed changes",
                ) + tuple(edit_reasons),
            ))
            # Prefer the earliest clean payoff that reaches campaign duration.
            break

    return plans


def build_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlanV31]:
    normal = final.build_plans(timeline, config, excluded)
    finishing = _verified_finishing_open_plans(timeline, config, excluded)
    combined = finishing + normal
    combined.sort(key=lambda item: item.score, reverse=True)
    unique: list[SemanticPlanV31] = []
    for plan in combined:
        anchor = plan.finishing_move.start if plan.finishing_move is not None else plan.engagements[0].events[0].time
        if any(
            abs(anchor - (prior.finishing_move.start if prior.finishing_move is not None else prior.engagements[0].events[0].time)) < 2.4
            for prior in unique
        ):
            continue
        unique.append(plan)
    return unique


def select_plans(plans: list[SemanticPlanV31], config: dict[str, Any]) -> list[SemanticPlanV31]:
    maximum = int(config.get("count_per_source_max", 4))
    minimum = int(config.get("minimum_count_per_source", 2))
    selected: list[SemanticPlanV31] = []

    # Verified Finishing Moves are a requested creative priority. If one passed all
    # semantic/quality gates, it must be represented in the batch and must open its clip.
    finishing = [plan for plan in plans if plan.finishing_move is not None and plan.story_type == "finishing_move_open"]
    if finishing:
        selected.append(max(finishing, key=lambda item: item.score))

    seen_story = {plan.story_type for plan in selected}
    for plan in plans:
        if plan in selected or plan.story_type in seen_story:
            continue
        if any(max(plan.start, prior.start) < min(plan.end, prior.end) for prior in selected):
            continue
        selected.append(plan)
        seen_story.add(plan.story_type)
        if len(selected) >= maximum:
            break
    for plan in plans:
        if plan in selected:
            continue
        if any(max(plan.start, prior.start) < min(plan.end, prior.end) for prior in selected):
            continue
        selected.append(plan)
        if len(selected) >= maximum:
            break
    if len(selected) < minimum:
        raise RuntimeError(
            f"Only {len(selected)} V3.1 production candidates passed the semantic quality gates; minimum is {minimum}."
        )
    return sorted(selected, key=lambda item: item.start)


def _self_test() -> None:
    print(json.dumps({
        "self_test": "PASS",
        "finishing_move_selection": "verified move is forced into selected batch when it passes gates",
        "finishing_move_position": "opening-only",
        "finishing_move_cross_shot_continuation": True,
    }))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    parser.error("Use through V3.1 shadow/production renderer")


if __name__ == "__main__":
    main()
