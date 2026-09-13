from __future__ import annotations

from typing import Any

import mw4_v3_1_allocator_hardened as _allocator
import mw4_v3_1_contract_legacy as _legacy
from mw4_v3_1_contract_legacy import *  # noqa: F401,F403

_ORIGINAL_CONFLICT = _legacy._plans_conflict
_ORIGINAL_FINISHING_FAILURES = _legacy._finishing_plan_failures
_ORIGINAL_VALIDATE = _legacy.validate_plan
_EPS = 1e-3


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


def _ordering_failures(source: str, index: int, plan: dict[str, Any]) -> list[str]:
    segments = list(plan.get("segments") or [])
    if not segments:
        return [f"{source} clip {index}: plan has no source segments"]
    failures: list[str] = []
    start = float(plan.get("start", segments[0]["start"]))
    end = float(plan.get("end", segments[-1]["end"]))
    if abs(start - float(segments[0]["start"])) > _EPS:
        failures.append(f"{source} clip {index}: plan start does not match first segment")
    if abs(end - float(segments[-1]["end"])) > _EPS or end <= start:
        failures.append(f"{source} clip {index}: plan bounds are non-monotonic")
    for pos, segment in enumerate(segments):
        s0, s1 = float(segment["start"]), float(segment["end"])
        if s1 <= s0:
            failures.append(f"{source} clip {index}: segment {pos + 1} has invalid bounds")
        if pos and s0 < float(segments[pos - 1]["end"]) - _EPS:
            failures.append(f"{source} clip {index}: source chronology reverses/overlaps at segment {pos + 1}")
    return failures


def _finishing_plan_failures(source: str, index: int, plan: dict[str, Any], config: dict[str, Any]) -> list[str]:
    failures = list(_ORIGINAL_FINISHING_FAILURES(source, index, plan, config))
    segments = list(plan.get("segments") or [])
    if plan.get("finishing_move") is not None and len(segments) > 1:
        gap = float(segments[1]["start"]) - float(segments[0]["end"])
        maximum = float(config.get("editorial", {}).get("finishing_move_max_continuation_gap_seconds", 18.0))
        if gap < -_EPS:
            failures.append(f"{source} clip {index}: Finishing Move continuation runs backward/overlaps")
        elif gap > maximum + _EPS:
            failures.append(f"{source} clip {index}: Finishing Move continuation gap exceeds {maximum:.3f}s")
    return list(dict.fromkeys(failures))


def validate_plan(source: str, index: int, plan: dict[str, Any], config: dict[str, Any]) -> list[str]:
    failures = list(_ORIGINAL_VALIDATE(source, index, plan, config))
    failures.extend(_ordering_failures(source, index, plan))
    return list(dict.fromkeys(failures))


_legacy._semantic_anchor_times = _semantic_anchor_times
_legacy._plans_conflict = _plans_conflict
_legacy._finishing_plan_failures = _finishing_plan_failures
_legacy.validate_plan = validate_plan
_allocator.install(_legacy)


def main() -> None:
    _legacy.main()


if __name__ == "__main__":
    main()
