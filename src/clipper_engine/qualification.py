from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gameplay import contract


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_batch(
    root: Path,
    config: dict[str, Any],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    contract.validate_configuration(config)
    summaries = [_load(path) for path in sorted(root.rglob("*_pipeline_summary.json"))]
    source_order = tuple(str(item) for item in allocation.get("source_order") or [])
    expected = set(source_order)
    actual = {str(item.get("source_key", "")) for item in summaries}
    failures: list[str] = []

    if actual != expected:
        failures.append(f"batch summaries cover {sorted(actual)}, expected {sorted(expected)}")
    if len(summaries) != len(expected):
        failures.append(f"expected exactly {len(expected)} source summaries, found {len(summaries)}")

    total_selected = 0
    total_rendered = 0
    total_verified_finishers = 0
    total_selected_finishers = 0

    for item in summaries:
        source = str(item.get("source_key", ""))
        if item.get("failure"):
            failures.append(f"{source}: {item['failure']}")
        selected = int(item.get("selected_count", 0))
        rendered = int(item.get("rendered_count", 0))
        expected_count = int(
            allocation.get("source_allocations", {}).get(source, {}).get("count", -1)
        )
        if rendered != selected:
            failures.append(f"{source}: rendered {rendered} != selected {selected}")
        if selected != expected_count:
            failures.append(f"{source}: rendered count {selected} != allocation {expected_count}")
        if not bool(item.get("technical_qa_passed", False)):
            failures.append(f"{source}: technical QA did not pass")
        if int(item.get("unplanned_source_cut_count", -1)) != 0:
            failures.append(f"{source}: unplanned source cuts present")

        source_allocation = dict(allocation.get("source_allocations", {}).get(source) or {})
        if int(source_allocation.get("selected_pair_conflict_count", -1)) != 0:
            failures.append(f"{source}: allocation reports selected candidate conflicts")
        dispositions = list(source_allocation.get("outcome_dispositions") or [])
        unresolved = [
            entry for entry in dispositions
            if entry.get("disposition") not in {
                "covered_by_selected_clip",
                "no_admissible_candidate",
            }
        ]
        if unresolved:
            failures.append(f"{source}: unresolved outcome dispositions: {unresolved}")

        total_selected += selected
        total_rendered += rendered
        total_verified_finishers += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    derived_target = int(allocation.get("target_count", -1))
    if total_selected != derived_target:
        failures.append(f"batch selected {total_selected}; allocation target is {derived_target}")
    if total_rendered != total_selected:
        failures.append("batch rendered count does not equal selected count")
    if int(allocation.get("selected_count", -1)) != total_selected:
        failures.append("rendered batch does not match pre-render allocation total")
    if total_verified_finishers > 0 and total_selected_finishers < 1:
        failures.append("verified Finishing Move exists in batch but none was rendered")

    if failures:
        raise AssertionError("\n".join(failures))

    return {
        "schema_version": 1,
        "sources": list(source_order),
        "selected_count": total_selected,
        "rendered_count": total_rendered,
        "selection_mode": "coverage_conflict_before_render",
        "derived_target_count": derived_target,
        "verified_finishing_move_count": total_verified_finishers,
        "selected_finishing_move_count": total_selected_finishers,
        "allocation_enforced": True,
        "status": "PASS",
    }
