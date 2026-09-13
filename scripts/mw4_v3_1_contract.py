from __future__ import annotations

from typing import Any

import mw4_v3_1_contract_legacy as _legacy
from mw4_v3_1_contract_legacy import *  # noqa: F401,F403

_ORIGINAL_CONFLICT = _legacy._plans_conflict


def _semantic_anchor_times(plan: dict[str, Any]) -> list[float]:
    anchors: set[float] = set()
    finishing = plan.get("finishing_move")
    if finishing is not None:
        anchors.add(round(float(finishing.get("payoff", finishing["start"])), 3))
    for event in plan.get("effect_events") or []:
        if str(event.get("kind", "")) in {"outcome_like", "impact"} and "time" in event:
            anchors.add(round(float(event["time"]), 3))
    for engagement in plan.get("engagements") or []:
        for event in engagement.get("events") or []:
            kinds = {str(item) for item in (event.get("kinds") or [])}
            if kinds.intersection({"outcome_like", "impact"}) and "time" in event:
                anchors.add(round(float(event["time"]), 3))
    return sorted(anchors)


def _reuses_montage_moment(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_segments = [s for s in (left.get("segments") or []) if s.get("reason") == "semantic_montage_moment"]
    right_segments = [s for s in (right.get("segments") or []) if s.get("reason") == "semantic_montage_moment"]
    for a in left_segments:
        for b in right_segments:
            a0, a1 = float(a["start"]), float(a["end"])
            b0, b1 = float(b["start"]), float(b["end"])
            shared = max(0.0, min(a1, b1) - max(a0, b0))
            shorter = min(a1 - a0, b1 - b0)
            if shared >= 0.35 and shorter > 0 and shared / shorter >= 0.50:
                return True
    return False


def _plans_conflict(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> bool:
    return _reuses_montage_moment(left, right) or _ORIGINAL_CONFLICT(left, right, config)


_legacy._semantic_anchor_times = _semantic_anchor_times
_legacy._plans_conflict = _plans_conflict


def main() -> None:
    _legacy.main()


if __name__ == "__main__":
    main()
