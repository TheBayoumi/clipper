from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.validate_editorial_lifecycle import validate_editorial_call_closure


def _start(invocation: str, lifecycle: str, task: str = "source_hazards:x") -> dict:
    return {
        "app": "clipper-production-pipeline",
        "authoritative": True,
        "event": "editorial_remote_call_start",
        "execution_id": "exec-129",
        "invocation_id": invocation,
        "producer_lifecycle_id": lifecycle,
        "task": task,
    }


def _terminal(invocation: str, lifecycle: str, task: str = "source_hazards:x") -> dict:
    return {
        "app": "clipper-production-pipeline",
        "authoritative": True,
        "event": "editorial_remote_call_terminal",
        "execution_id": "exec-129",
        "invocation_id": invocation,
        "producer_lifecycle_id": lifecycle,
        "task": task,
        "status": "COMPLETE",
    }


def _application_result(invocation: str) -> dict:
    return {
        "app": "clipper-open-editor",
        "authoritative": False,
        "event": "application_result",
        "execution_id": "exec-129",
        "invocation_id": invocation,
        "application_status": "CAPACITY_REJECTED",
        "error_type": "EditorialCapacityError",
        "recovery_action": "REPARTITION",
    }


def _reconciliation(
    invocation: str,
    old_lifecycle: str,
    replacement_lifecycle: str,
    task: str = "source_hazards:x",
) -> dict:
    return {
        "app": "modal-spy",
        "authoritative": False,
        "event": "editorial_remote_call_reconciled",
        "execution_id": "exec-129",
        "invocation_id": invocation,
        "task": task,
        "application_status": "CAPACITY_REJECTED",
        "error_type": "EditorialCapacityError",
        "recovery_action": "REPARTITION",
        "producer_lifecycle_id": old_lifecycle,
        "replacement_producer_lifecycle_id": replacement_lifecycle,
        "reason": "producer_lifecycle_replaced_after_recoverable_remote_result",
    }


def _summary(*, starts: int, terminals: int, reconciliations: list[dict]) -> dict:
    return {
        "authoritative_event_counts": {
            "editorial_remote_call_start": starts,
            "editorial_remote_call_terminal": terminals,
        },
        "reconciled_editorial_calls": reconciliations,
    }


def _hilp129_records() -> tuple[list[dict], dict]:
    reconciliation = _reconciliation("inv-old", "producer-a", "producer-b")
    return (
        [
            _start("inv-old", "producer-a"),
            _application_result("inv-old"),
            reconciliation,
            _start("inv-new", "producer-b"),
            _terminal("inv-new", "producer-b"),
        ],
        reconciliation,
    )


def test_lifecycle_gate_accepts_normal_one_to_one_terminal_closure() -> None:
    records = [_start("inv-1", "producer-a"), _terminal("inv-1", "producer-a")]
    result = validate_editorial_call_closure(
        summary=_summary(starts=1, terminals=1, reconciliations=[]),
        records=records,
    )
    assert result == {
        "starts": 1,
        "terminals": 1,
        "reconciled": 0,
        "reconciled_invocation_ids": [],
    }


def test_lifecycle_gate_accepts_exact_hilp129_recoverable_restart_closure() -> None:
    records, reconciliation = _hilp129_records()
    result = validate_editorial_call_closure(
        summary=_summary(starts=2, terminals=1, reconciliations=[reconciliation]),
        records=records,
    )
    assert result["reconciled_invocation_ids"] == ["inv-old"]
    assert result["starts"] == result["terminals"] + result["reconciled"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda records: [
                item for item in records if item.get("event") != "application_result"
            ],
            "lacks model-side application_result",
        ),
        (
            lambda records: [
                {
                    **item,
                    "replacement_producer_lifecycle_id": "producer-a",
                }
                if item.get("event") == "editorial_remote_call_reconciled"
                else item
                for item in records
            ],
            "distinct producer replacement",
        ),
        (
            lambda records: [
                item
                for item in records
                if not (
                    item.get("event") == "editorial_remote_call_start"
                    and item.get("producer_lifecycle_id") == "producer-b"
                )
            ],
            "replacement lifecycle has no authoritative producer start",
        ),
        (
            lambda records: [
                {
                    **item,
                    "application_status": "FAILED",
                    "recovery_action": "NONE",
                }
                if item.get("event") == "editorial_remote_call_reconciled"
                else item
                for item in records
            ],
            "not a permitted recoverable result",
        ),
    ],
)
def test_lifecycle_gate_rejects_unsafe_reconciliation(mutator, message: str) -> None:
    records, reconciliation = _hilp129_records()
    mutated = mutator(copy.deepcopy(records))
    mutated_reconciliations = [
        event
        for event in mutated
        if event.get("event") == "editorial_remote_call_reconciled"
    ]
    with pytest.raises(RuntimeError, match=message):
        validate_editorial_call_closure(
            summary=_summary(
                starts=2,
                terminals=1,
                reconciliations=mutated_reconciliations,
            ),
            records=mutated,
        )


def test_lifecycle_gate_rejects_unknown_unmatched_start() -> None:
    records = [
        _start("inv-old", "producer-a"),
        _start("inv-new", "producer-b"),
        _terminal("inv-new", "producer-b"),
    ]
    with pytest.raises(RuntimeError, match="not one-to-one closed"):
        validate_editorial_call_closure(
            summary=_summary(starts=2, terminals=1, reconciliations=[]),
            records=records,
        )


def test_lifecycle_gate_rejects_terminal_and_reconciliation_for_same_invocation() -> None:
    reconciliation = _reconciliation("inv-old", "producer-a", "producer-b")
    records = [
        _start("inv-old", "producer-a"),
        _terminal("inv-old", "producer-a"),
        _application_result("inv-old"),
        reconciliation,
        _start("inv-new", "producer-b"),
        _terminal("inv-new", "producer-b"),
    ]
    with pytest.raises(RuntimeError, match="both producer-terminal and restart-reconciled"):
        validate_editorial_call_closure(
            summary=_summary(starts=2, terminals=2, reconciliations=[reconciliation]),
            records=records,
        )


def test_production_workflow_uses_reconciliation_aware_lifecycle_gate() -> None:
    workflow = Path(".github/workflows/production-pipeline.yml").read_text(
        encoding="utf-8"
    )
    assert workflow.count(
        "validate_editorial_call_closure(summary=summary, records=records)"
    ) == 2
    assert "call_terminals != call_starts" not in workflow
