from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import combat
from . import interaction
from . import semantics as core

Engagement = core.Engagement
_EPS = 1e-3

def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _support_components(timeline: Any, config: dict[str, Any]) -> tuple[Engagement, ...]:
    """Build maximal same-shot, hard-gap-free support regions around verified hostile evidence.

    Coarse engagement envelopes remain discovery input only. They are intentionally
    not used as final edit bounds.
    """
    hard = np.asarray(
        timeline.signals.get("combat_island_gap", combat.gap_signal(timeline, config)),
        dtype=np.float32,
    )
    grouped: dict[tuple[int, float, float], list[tuple[Any, Any]]] = defaultdict(list)
    for event in timeline.consolidated_events:
        decision = interaction.hostile_decision(event, config)
        if not decision.hostile:
            continue
        component = combat._support_component_for_event(timeline, event, hard, core)
        if component is None:
            continue
        shot_index, start, end = component
        grouped[(int(shot_index), float(start), float(end))].append((event, decision))

    minimum = _f(
        config["combat_state_verifier"].get("minimum_combat_island_seconds", 1.0),
        1.0,
    )
    out: list[Engagement] = []
    for (shot_index, start, end), members in grouped.items():
        if end - start < minimum - _EPS:
            continue
        members = sorted(members, key=lambda item: float(item[0].time))
        out.append(
            Engagement(
                round(start, 3),
                round(end, 3),
                shot_index,
                round(float(np.mean([item[1].score for item in members])), 4),
                tuple(item[0] for item in members),
            )
        )
    return tuple(sorted(out, key=lambda item: (item.start, item.end)))

def _component_chains(
    components: tuple[Engagement, ...],
    config: dict[str, Any],
) -> tuple[tuple[Engagement, ...], ...]:
    max_gap = _f(
        config["semantic_editor"].get("maximum_inter_engagement_gap_seconds", 4.0),
        4.0,
    )
    max_items = int(config["semantic_editor"].get("maximum_engagements_per_story", 7))
    chains: list[tuple[Engagement, ...]] = []
    for index, first in enumerate(components):
        chain: list[Engagement] = []
        for candidate in components[index:]:
            if int(candidate.shot_index) != int(first.shot_index):
                break
            if chain and float(candidate.start) - float(chain[-1].end) > max_gap + _EPS:
                break
            chain.append(candidate)
            if len(chain) > max_items:
                break
            chains.append(tuple(chain))
    return tuple(chains)

