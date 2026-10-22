from __future__ import annotations

from typing import Any

_RECOVERABLE_RECONCILIATION_ACTIONS = {
    "CAPACITY_REJECTED": "REPARTITION",
    "OUTPUT_RETRY": "REGENERATE",
}
_RECONCILIATION_SHARED_FIELDS = (
    "event",
    "execution_id",
    "invocation_id",
    "task",
    "application_status",
    "error_type",
    "recovery_action",
    "producer_lifecycle_id",
    "replacement_producer_lifecycle_id",
    "reason",
)
_TERMINAL_IDENTITY_FIELDS = (
    "execution_id",
    "task",
    "producer_lifecycle_id",
)


def _invocation_id(event: dict[str, Any]) -> str:
    return str(event.get("invocation_id") or "")


def _normalized_fields(
    event: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, str]:
    return {field: str(event.get(field) or "") for field in fields}


def _unique_by_invocation(
    events: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for event in events:
        invocation_id = _invocation_id(event)
        if not invocation_id:
            raise RuntimeError(f"{label} omitted invocation identity: {event}")
        if invocation_id in indexed:
            raise RuntimeError(
                f"{label} repeated invocation identity: invocation_id={invocation_id}"
            )
        indexed[invocation_id] = event
    return indexed


def validate_editorial_call_closure(
    *,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate producer terminals plus tightly evidenced restart reconciliations."""
    authoritative_counts = summary.get("authoritative_event_counts")
    if not isinstance(authoritative_counts, dict):
        raise RuntimeError("Modal spy authoritative event counts are missing")

    expected_starts = int(authoritative_counts.get("editorial_remote_call_start") or 0)
    expected_terminals = int(authoritative_counts.get("editorial_remote_call_terminal") or 0)
    if expected_starts <= 0:
        raise RuntimeError("live editorial acceptance observed no producer-side calls")

    producer_starts = [
        event
        for event in records
        if event.get("event") == "editorial_remote_call_start"
        and event.get("authoritative") is True
    ]
    producer_terminals = [
        event
        for event in records
        if event.get("event") == "editorial_remote_call_terminal"
        and event.get("authoritative") is True
    ]
    reconciliations = [
        event
        for event in records
        if event.get("event") == "editorial_remote_call_reconciled"
        and event.get("authoritative") is False
    ]
    application_results = [
        event
        for event in records
        if event.get("event") == "application_result" and event.get("authoritative") is False
    ]

    if len(producer_starts) != expected_starts:
        raise RuntimeError(
            "producer start telemetry count disagrees with spy summary: "
            f"records={len(producer_starts)} summary={expected_starts}"
        )
    if len(producer_terminals) != expected_terminals:
        raise RuntimeError(
            "producer terminal telemetry count disagrees with spy summary: "
            f"records={len(producer_terminals)} summary={expected_terminals}"
        )

    starts_by_id = _unique_by_invocation(producer_starts, label="editorial producer start")
    terminals_by_id = _unique_by_invocation(producer_terminals, label="editorial producer terminal")
    reconciled_by_id = _unique_by_invocation(
        reconciliations, label="editorial producer reconciliation"
    )

    summary_reconciliations = summary.get("reconciled_editorial_calls")
    if not isinstance(summary_reconciliations, list):
        raise RuntimeError("Modal spy reconciliation summary is missing")
    normalized_summary_reconciliations: list[dict[str, Any]] = []
    for item in summary_reconciliations:
        if not isinstance(item, dict):
            raise RuntimeError(f"invalid Modal spy reconciliation summary item: {item!r}")
        normalized_summary_reconciliations.append(item)
    summary_reconciled_by_id = _unique_by_invocation(
        normalized_summary_reconciliations,
        label="Modal spy reconciliation summary",
    )
    if set(summary_reconciled_by_id) != set(reconciled_by_id):
        raise RuntimeError(
            "Modal spy reconciliation summary disagrees with NDJSON evidence: "
            f"summary={sorted(summary_reconciled_by_id)} "
            f"records={sorted(reconciled_by_id)}"
        )
    for invocation_id, reconciliation in reconciled_by_id.items():
        summary_reconciliation = summary_reconciled_by_id[invocation_id]
        summary_fields = _normalized_fields(
            summary_reconciliation,
            _RECONCILIATION_SHARED_FIELDS,
        )
        record_fields = _normalized_fields(
            reconciliation,
            _RECONCILIATION_SHARED_FIELDS,
        )
        if summary_fields != record_fields:
            raise RuntimeError(
                "Modal spy reconciliation summary disagrees with NDJSON evidence: "
                f"invocation_id={invocation_id} summary={summary_fields} records={record_fields}"
            )

    overlap = set(terminals_by_id) & set(reconciled_by_id)
    if overlap:
        raise RuntimeError(
            "editorial invocation was both producer-terminal and restart-reconciled: "
            f"{sorted(overlap)}"
        )

    for invocation_id, terminal in terminals_by_id.items():
        start = starts_by_id.get(invocation_id)
        if start is None:
            raise RuntimeError(
                f"editorial producer terminal has no authoritative producer start: {terminal}"
            )
        for field in _TERMINAL_IDENTITY_FIELDS:
            start_value = str(start.get(field) or "")
            terminal_value = str(terminal.get(field) or "")
            if not start_value or not terminal_value:
                raise RuntimeError(
                    "editorial producer terminal identity is incomplete: "
                    f"field={field} start={start} terminal={terminal}"
                )
            if start_value != terminal_value:
                raise RuntimeError(
                    "editorial producer terminal identity does not match its start: "
                    f"field={field} start={start} terminal={terminal}"
                )

    for invocation_id, reconciliation in reconciled_by_id.items():
        start = starts_by_id.get(invocation_id)
        if start is None:
            raise RuntimeError(
                f"editorial reconciliation has no authoritative producer start: {reconciliation}"
            )

        application_status = str(reconciliation.get("application_status") or "")
        expected_action = _RECOVERABLE_RECONCILIATION_ACTIONS.get(application_status)
        recovery_action = str(reconciliation.get("recovery_action") or "")
        if expected_action is None or recovery_action != expected_action:
            raise RuntimeError(
                f"editorial reconciliation is not a permitted recoverable result: {reconciliation}"
            )

        producer_lifecycle_id = str(reconciliation.get("producer_lifecycle_id") or "")
        replacement_lifecycle_id = str(
            reconciliation.get("replacement_producer_lifecycle_id") or ""
        )
        if (
            not producer_lifecycle_id
            or not replacement_lifecycle_id
            or producer_lifecycle_id == replacement_lifecycle_id
        ):
            raise RuntimeError(
                f"editorial reconciliation lacks a distinct producer replacement: {reconciliation}"
            )
        if str(start.get("producer_lifecycle_id") or "") != producer_lifecycle_id:
            raise RuntimeError(
                "editorial reconciliation lifecycle does not match its producer start: "
                f"start={start} reconciliation={reconciliation}"
            )
        if str(start.get("task") or "") != str(reconciliation.get("task") or ""):
            raise RuntimeError(
                "editorial reconciliation task does not match its producer start: "
                f"start={start} reconciliation={reconciliation}"
            )
        if str(start.get("execution_id") or "") != str(reconciliation.get("execution_id") or ""):
            raise RuntimeError(
                "editorial reconciliation execution does not match its producer start: "
                f"start={start} reconciliation={reconciliation}"
            )

        same_invocation_results = [
            result for result in application_results if _invocation_id(result) == invocation_id
        ]
        if not same_invocation_results:
            raise RuntimeError(
                "editorial reconciliation lacks model-side application_result evidence: "
                f"{reconciliation}"
            )
        if not any(
            str(result.get("application_status") or "") == application_status
            and str(result.get("recovery_action") or "") == recovery_action
            and str(result.get("error_type") or "") == str(reconciliation.get("error_type") or "")
            and str(result.get("execution_id") or "")
            == str(reconciliation.get("execution_id") or "")
            for result in same_invocation_results
        ):
            raise RuntimeError(
                "editorial reconciliation does not match model-side recoverable evidence: "
                f"reconciliation={reconciliation} results={same_invocation_results}"
            )

        replacement_starts = [
            candidate
            for candidate in producer_starts
            if _invocation_id(candidate) != invocation_id
            and str(candidate.get("producer_lifecycle_id") or "") == replacement_lifecycle_id
            and str(candidate.get("execution_id") or "")
            == str(reconciliation.get("execution_id") or "")
        ]
        if not replacement_starts:
            raise RuntimeError(
                "editorial reconciliation replacement lifecycle has no authoritative "
                f"producer start: {reconciliation}"
            )

    start_ids = set(starts_by_id)
    terminal_ids = set(terminals_by_id)
    reconciled_ids = set(reconciled_by_id)
    effectively_closed_ids = terminal_ids | reconciled_ids
    if start_ids != effectively_closed_ids:
        raise RuntimeError(
            "producer-side editorial invocation IDs are not one-to-one closed: "
            f"starts={sorted(start_ids)} terminals={sorted(terminal_ids)} "
            f"reconciled={sorted(reconciled_ids)}"
        )
    if expected_starts != expected_terminals + len(reconciled_ids):
        raise RuntimeError(
            "producer-side editorial lifecycle counts do not close after reconciliation: "
            f"starts={expected_starts} terminals={expected_terminals} "
            f"reconciled={len(reconciled_ids)}"
        )

    return {
        "starts": expected_starts,
        "terminals": expected_terminals,
        "reconciled": len(reconciled_ids),
        "reconciled_invocation_ids": sorted(reconciled_ids),
    }
