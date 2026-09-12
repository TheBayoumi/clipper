from __future__ import annotations

from pathlib import Path


def replace_exact(path: Path, old: str, new: str, *, expected: int) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"expected {expected} patch targets in {path}, found {count}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


helper = Path("scripts/validate_editorial_lifecycle.py")
replace_exact(
    helper,
    '    "OUTPUT_RETRY": "OUTPUT_RETRY",\n',
    '    "OUTPUT_RETRY": "REGENERATE",\n',
    expected=1,
)

workflow = Path(".github/workflows/production-pipeline.yml")
old_count = '''          call_starts = int(authoritative_counts.get("editorial_remote_call_start") or 0)
          call_terminals = int(authoritative_counts.get("editorial_remote_call_terminal") or 0)
          if call_starts <= 0 or call_terminals != call_starts:
              raise RuntimeError(
                  "live editorial acceptance lacks a closed producer-side editorial call set: "
                  f"starts={call_starts} terminals={call_terminals}"
              )
'''
new_count = '''          call_starts = int(authoritative_counts.get("editorial_remote_call_start") or 0)
          call_terminals = int(authoritative_counts.get("editorial_remote_call_terminal") or 0)
          if call_starts <= 0:
              raise RuntimeError("live editorial acceptance observed no producer-side calls")
'''
replace_exact(workflow, old_count, new_count, expected=2)

old_ids = '''          start_ids = {
              str(event.get("invocation_id") or "")
              for event in producer_starts
          }
          terminal_ids = {
              str(event.get("invocation_id") or "")
              for event in producer_terminals
          }
          if "" in start_ids or "" in terminal_ids or start_ids != terminal_ids:
              raise RuntimeError(
                  "producer-side editorial invocation IDs are not one-to-one closed: "
                  f"starts={sorted(start_ids)} terminals={sorted(terminal_ids)}"
              )
'''
new_ids = '''          from scripts.validate_editorial_lifecycle import validate_editorial_call_closure

          validate_editorial_call_closure(summary=summary, records=records)
'''
replace_exact(workflow, old_ids, new_ids, expected=2)

test = Path("tests/test_hilp129_editorial_lifecycle_gate.py")
marker = '''def test_lifecycle_gate_accepts_exact_hilp129_recoverable_restart_closure() -> None:
    records, reconciliation = _hilp129_records()
    result = validate_editorial_call_closure(
        summary=_summary(starts=2, terminals=1, reconciliations=[reconciliation]),
        records=records,
    )
    assert result["reconciled_invocation_ids"] == ["inv-old"]
    assert result["starts"] == result["terminals"] + result["reconciled"]


'''
addition = marker + '''def test_lifecycle_gate_accepts_output_retry_regeneration_restart_closure() -> None:
    records, reconciliation = _hilp129_records()
    for event in records:
        if event.get("event") in {
            "application_result",
            "editorial_remote_call_reconciled",
        }:
            event["application_status"] = "OUTPUT_RETRY"
            event["error_type"] = "EditorialOutputInvalid"
            event["recovery_action"] = "REGENERATE"
    result = validate_editorial_call_closure(
        summary=_summary(starts=2, terminals=1, reconciliations=[reconciliation]),
        records=records,
    )
    assert result["reconciled_invocation_ids"] == ["inv-old"]
    assert result["reconciled"] == 1


'''
replace_exact(test, marker, addition, expected=1)
