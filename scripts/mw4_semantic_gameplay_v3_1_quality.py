from __future__ import annotations
import math
from typing import Any
import numpy as np
import mw4_semantic_gameplay_v3_1_refined as refined
import mw4_semantic_gameplay_v3_1_combat_state as combat
import mw4_semantic_gameplay_v3_1_combat_islands as islands
Engagement=refined.Engagement
FinishingMoveSpan=refined.FinishingMoveSpan
EditSegment=refined.EditSegment
SemanticPlanV31=refined.SemanticPlanV31
SemanticTimelineV31=refined.SemanticTimelineV31
PAYOFF_KINDS=refined.PAYOFF_KINDS
_EPS=1e-3
def _f(value: Any, default: float=0.0)->float:
    try: return float(value)
    except (TypeError,ValueError): return default
def _verified_cut_windows(config,source_key):
    out=[]
    for item in config.get("source_integrity",{}).get("verified_cut_windows",{}).get(source_key,[]):
        if len(item)!=2: continue
        a,b=sorted((float(item[0]),float(item[1])))
        if b>a: out.append((a,b))
    return tuple(sorted(out))
def _bridge_supported(timeline,left,right,config): return islands.bridge_supported(timeline,left,right,config)
def _segment_crosses_hard_gap(segment,timeline): return islands.segment_crosses_gap(timeline,float(segment.start),float(segment.end))
_event_hostile_decision=combat.hostile_decision
def _segment_duration(segment: EditSegment) -> float:
    return (float(segment.end) - float(segment.start)) / float(segment.speed)
def _source_segments_overlap(left: SemanticPlanV31, right: SemanticPlanV31) -> bool:
    return any((max(a.start, b.start) < min(a.end, b.end) for a in left.segments for b in right.segments))
def _explained_mask(timeline: SemanticTimelineV31, engagements: tuple[Engagement, ...], finishing: FinishingMoveSpan | None, config: dict[str, Any]) -> np.ndarray:
    cfg = config['semantic_editor']['dull']
    pre = float(cfg.get('engagement_explanation_lead_seconds', 0.35))
    post = float(cfg.get('engagement_explanation_tail_seconds', 0.3))
    mask = np.zeros(len(timeline.times), dtype=bool)
    for engagement in engagements:
        i0 = max(0, int(math.floor((engagement.start - pre) * timeline.fps)))
        i1 = min(len(mask), int(math.ceil((engagement.end + post) * timeline.fps)))
        mask[i0:i1] = True
    if finishing is not None:
        i0 = max(0, int(math.floor((finishing.start - 0.12) * timeline.fps)))
        i1 = min(len(mask), int(math.ceil((finishing.end + 0.18) * timeline.fps)))
        mask[i0:i1] = True
    return mask
def _longest_unexplained_low_run(timeline: SemanticTimelineV31, start: float, end: float, engagements: tuple[Engagement, ...], finishing: FinishingMoveSpan | None, config: dict[str, Any]) -> float:
    i0 = max(0, int(math.floor(start * timeline.fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * timeline.fps)))
    if i1 <= i0:
        return float('inf')
    threshold = float(config['semantic_editor']['dull'].get('low_interest_threshold', 0.3))
    explained = _explained_mask(timeline, engagements, finishing, config)
    low = np.asarray(timeline.signals['interest'] < threshold, dtype=bool)
    return refined.core._longest_true_run((low & ~explained)[i0:i1], float(timeline.fps))
def _quality_metrics(timeline: SemanticTimelineV31, start: float, end: float, engagements: tuple[Engagement, ...], finishing: FinishingMoveSpan | None, residual_run: float, config: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    sig = timeline.signals
    fps = timeline.fps
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * fps)))
    raw_interest = np.asarray(sig['interest'][i0:i1], dtype=np.float32)
    if raw_interest.size == 0:
        return (0.0,) * 7
    explained = _explained_mask(timeline, engagements, finishing, config)[i0:i1]
    floor = float(config['semantic_editor']['dull'].get('explained_interest_floor', 0.26))
    effective = np.where(explained, np.maximum(raw_interest, floor), raw_interest)
    quarters = np.array_split(effective, 4)
    weakest = min((float(np.mean(part)) for part in quarters if len(part)))
    low_threshold = float(config['semantic_editor']['dull'].get('low_interest_threshold', 0.3))
    low_fraction = float(np.mean((raw_interest < low_threshold) & ~explained))
    first_event = finishing.start if finishing is not None else engagements[0].events[0].time
    delay = max(0.0, float(first_event) - start)
    opening_mean = refined.core._mean(sig['interest'], timeline, start, min(end, start + 1.25))
    opening_slice = np.asarray(sig['interest'][i0:min(i1, i0 + max(1, int(round(1.1 * fps))))], dtype=np.float32)
    opening_explained = explained[:len(opening_slice)]
    opening_dead = refined.core._longest_true_run((opening_slice < low_threshold) & ~opening_explained, fps)
    opening = float(np.clip(0.43 * opening_mean + 0.34 * (1.0 - min(1.0, delay / 0.9)) + 0.23 * (1.0 - min(1.0, opening_dead / 0.5)), 0, 1))
    if finishing is not None:
        opening = min(1.0, opening + 0.2 * float(finishing.confidence))
    payoff_events = [event for engagement in engagements for event in engagement.events if any((kind in PAYOFF_KINDS for kind in event.kinds))]
    last_payoff = payoff_events[-1].time if payoff_events else None
    if finishing is not None and (last_payoff is None or finishing.payoff > last_payoff):
        last_payoff = finishing.payoff
    if last_payoff is not None:
        tail = max(0.0, end - float(last_payoff))
        tail_quality = float(np.clip(1.0 - abs(tail - 0.45) / 0.75, 0, 1))
        ending_interest = refined.core._mean(sig['interest'], timeline, max(start, end - 1.0), end)
        ending = 0.66 * tail_quality + 0.34 * ending_interest
    else:
        active = max(refined.core._mean(sig['combat'], timeline, max(start, end - 0.8), end), refined.core._mean(sig['contact'], timeline, max(start, end - 0.8), end))
        ending = 0.72 * active + 0.28 * refined.core._mean(sig['interest'], timeline, max(start, end - 1.0), end)
    payoff_strength = max([float(event.confidence) for event in payoff_events] + ([min(1.0, float(finishing.confidence) + 0.12)] if finishing is not None else [0.0]))
    coherence = float(np.clip(0.54 * float(np.mean([eng.confidence for eng in engagements])) + 0.26 * min(1.0, len(engagements) / 4.0) + 0.2 * (1.0 - min(1.0, low_fraction / 0.45)), 0, 1))
    retention = float(np.clip(0.27 * float(np.mean(effective)) + 0.18 * weakest + 0.22 * opening + 0.18 * ending + 0.09 * coherence + 0.06 * (1.0 - min(1.0, residual_run / 0.9)), 0, 1))
    payoff = float(np.clip(0.72 * payoff_strength + 0.28 * ending, 0, 1))
    return (opening, float(ending), coherence, retention, payoff, weakest, low_fraction)
def _story_gate(story: str, metric: str, config: dict[str, Any], default: float) -> float:
    table = config.get('story_quality_gates', {}).get(metric, {})
    return float(table.get(story, table.get('default', default)))
def _passes_story_gates(story: str, opening: float, ending: float, retention: float, payoff: float, weakest: float, low_fraction: float, residual: float, config: dict[str, Any]) -> bool:
    editor = config['semantic_editor']
    return bool(opening >= _story_gate(story, 'opening_min', config, float(editor['opening'].get('minimum_quality', 0.42))) and ending >= _story_gate(story, 'ending_min', config, float(editor['ending'].get('minimum_quality', 0.4))) and (retention >= _story_gate(story, 'retention_min', config, float(config['performance_targets'].get('retention_quality_min', 0.36)))) and (payoff >= _story_gate(story, 'payoff_min', config, float(config['performance_targets'].get('payoff_quality_min', 0.34)))) and (weakest >= float(editor['dull'].get('minimum_weak_quarter_interest', 0.27))) and (low_fraction <= float(editor['dull'].get('maximum_low_interest_fraction', 0.38))) and (residual <= float(editor['dull'].get('maximum_unexplained_low_interest_run_seconds', 0.9))))
def _score_plan(retention: float, payoff: float, opening: float, ending: float, coherence: float, weakest: float, finishing: bool, config: dict[str, Any]) -> float:
    weights = config['semantic_editor']['selection']
    score = float(weights.get('retention_quality_weight', 0.24)) * retention + float(weights.get('payoff_quality_weight', 0.17)) * payoff + float(weights.get('opening_weight', 0.2)) * opening + float(weights.get('ending_weight', 0.17)) * ending + float(weights.get('story_coherence_weight', 0.12)) * coherence + float(weights.get('weakest_section_weight', 0.1)) * weakest
    if finishing:
        score += float(weights.get('finishing_move_bonus', 0.14))
    return float(score)
def _effect_events_for_engagements(timeline: SemanticTimelineV31, engagements: tuple[Engagement, ...]) -> tuple[Any, ...]:
    verified_times = [float(event.time) for engagement in engagements for event in engagement.events]
    raw = [event for event in timeline.base.events if event.kind in {'outcome_like', 'impact', 'combat_burst'} and event.confidence >= 0.56 and any((abs(float(event.time) - time) <= 0.3 for time in verified_times))]
    unique: list[Any] = []
    for event in sorted(raw, key=lambda item: (-float(item.confidence), float(item.time))):
        if any((abs(float(event.time) - float(old.time)) <= 0.12 and event.kind == old.kind for old in unique)):
            continue
        unique.append(event)
    return tuple(sorted(unique[:3], key=lambda item: float(item.time)))
def _route_story(timeline: SemanticTimelineV31, chain: tuple[Engagement, ...]) -> tuple[str, str, tuple[Any, ...], tuple[str, ...]]:
    effects = _effect_events_for_engagements(timeline, chain)
    verified = [event for engagement in chain for event in engagement.events]
    impact = [event for event in verified if 'impact' in event.kinds]
    outcome = [event for event in verified if 'outcome_like' in event.kinds]
    if len(chain) >= 3 and len(verified) >= 3:
        return ('engagement_chain', 'chain_escalation', effects, ('continuous directly verified hostile engagement chain',))
    if impact and max((float(item.confidence) for item in impact)) >= 0.76:
        return ('impact_payoff', 'impact_flash', effects, ('directly verified high-confidence impact payoff',))
    if outcome:
        return ('precision_outcome', 'precision_punch', effects, ('directly verified outcome payoff',))
    return ('sustained_pressure', 'clean_pressure', effects, ('continuous directly verified hostile pressure',))
def _chronology_failures(plan: SemanticPlanV31) -> list[str]:
    segments = list(plan.segments)
    if not segments:
        return ['plan has no source segments']
    failures: list[str] = []
    if abs(float(plan.start) - float(segments[0].start)) > _EPS:
        failures.append('plan start does not match first segment')
    if abs(float(plan.end) - float(segments[-1].end)) > _EPS:
        failures.append('plan end does not match last segment')
    if float(plan.end) <= float(plan.start):
        failures.append('plan bounds are non-monotonic')
    for index, segment in enumerate(segments):
        if float(segment.end) <= float(segment.start):
            failures.append(f'segment {index + 1} has invalid bounds')
        if abs(float(segment.speed) - 1.0) > 1e-06:
            failures.append(f'segment {index + 1} is not source-native 1.0x')
        if index and float(segment.start) < float(segments[index - 1].end) - _EPS:
            failures.append(f'source chronology reversal/overlap at segment {index + 1}')
    return failures
def _planned_cross_shot_bridge(plan: SemanticPlanV31, left: EditSegment, right: EditSegment, bridge_index: int) -> bool:
    if plan.story_type != 'finishing_move_open':
        return False
    if bridge_index == 0 and left.reason == 'finishing_move_open_hero' and (right.reason == 'verified_combat_island_body'):
        return True
    return left.reason == 'verified_combat_island_body' and right.reason == 'verified_combat_island_body'
def plan_integrity_violations(plan: SemanticPlanV31, timeline: SemanticTimelineV31, config: dict[str, Any], source_key: str) -> list[str]:
    failures = _chronology_failures(plan)
    windows = _verified_cut_windows(config, source_key)
    for index, segment in enumerate(plan.segments):
        inside = any((shot.start - _EPS <= segment.start and segment.end <= shot.end + _EPS for shot in timeline.shots))
        if not inside:
            failures.append(f'segment {segment.start:.3f}-{segment.end:.3f} crosses a source-shot boundary')
        for left, right in windows:
            midpoint = (left + right) / 2.0
            if segment.start < midpoint < segment.end:
                failures.append(f'segment {segment.start:.3f}-{segment.end:.3f} crosses verified source cut {left:.3f}-{right:.3f}')
        if segment.reason != 'finishing_move_open_hero' and _segment_crosses_hard_gap(segment, timeline):
            failures.append(f'segment {index + 1} crosses a sustained reload/search/recovery break')
    for index, (left, right) in enumerate(zip(plan.segments, plan.segments[1:])):
        if right.start <= left.end + 0.02:
            continue
        left_shot = refined.core._shot_index(timeline.shots, max(left.start, left.end - _EPS))
        right_shot = refined.core._shot_index(timeline.shots, min(right.end, right.start + _EPS))
        if left_shot != right_shot and (not _planned_cross_shot_bridge(plan, left, right, index)):
            failures.append(f'unplanned cross-shot bridge {left.end:.3f}->{right.start:.3f}')
    for engagement in plan.engagements:
        for event in engagement.events:
            if not _event_hostile_decision(event, config).hostile:
                failures.append(f'plan includes non-hostile/ambiguous anchor at {event.time:.3f}s')
    if plan.finishing_move is not None:
        if plan.story_type != 'finishing_move_open' or plan.effect_profile != 'finishing_move_hero':
            failures.append('verified Finishing Move is not routed as finishing_move_open/finishing_move_hero')
        if not plan.segments or plan.segments[0].reason != 'finishing_move_open_hero':
            failures.append('Finishing Move first segment is not the protected hero')
        else:
            first = plan.segments[0]
            if not first.start <= plan.finishing_move.start <= first.end:
                failures.append('Finishing Move is not contained in first output segment')
            if (plan.finishing_move.start - first.start) / first.speed > 0.48:
                failures.append('Finishing Move opening delay exceeds 0.48s')
        island_bounds = {(round(float(item.start), 3), round(float(item.end), 3)) for item in plan.engagements}
        for segment in plan.segments[1:]:
            if segment.reason != 'verified_combat_island_body':
                failures.append('Finishing Move body contains a non-island segment')
            if (round(float(segment.start), 3), round(float(segment.end), 3)) not in island_bounds:
                failures.append('Finishing Move body segment does not equal a verified combat-island boundary')
    return list(dict.fromkeys(failures))
