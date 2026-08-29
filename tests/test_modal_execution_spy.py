from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/modal_execution_spy.py")
    spec = importlib.util.spec_from_file_location("modal_execution_spy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load modal execution spy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spy_parses_prefixed_modal_structured_events(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("clipper-open-editor",), tmp_path / "spy.ndjson")
    payload = spy._parse_json(
        "2026-08-28T02:21:28Z fc-123 ta-456 "
        '{"event":"editorial_request_plan","task":"source_hazards:x","input_tokens":100}'
    )
    assert payload == {
        "event": "editorial_request_plan",
        "task": "source_hazards:x",
        "input_tokens": 100,
    }


def test_spy_summary_surfaces_projection_repartition_and_oom(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "spy.ndjson",
    )
    spy.events_seen = 3
    spy.event_counts = {
        "editorial_evidence_projection": 1,
        "editorial_repartition": 1,
        "editorial_oom": 1,
    }
    spy.latest = {
        "editorial_evidence_projection": {
            "stage": "source_hazards:x",
            "raw_event_count": 400,
            "projected_event_count": 20,
            "raw_serialized_bytes": 400_000,
            "projected_serialized_bytes": 20_000,
        },
        "editorial_repartition": {
            "stage": "source_hazards:x",
            "reason": "context_exhausted",
            "observed_input_tokens": 4_000_000,
            "target_input_tokens": 250_000,
            "partition_count": 16,
        },
        "editorial_oom": {
            "task": "source_hazards:y",
            "cache_implementation": "dynamic",
            "input_tokens": 150_000,
        },
    }
    body = spy._comment_body()
    assert module.MARKER in body
    assert "Evidence projection" in body
    assert "Token-aware repartition" in body
    assert "Last OOM" in body
    assert "400000" in body
    assert "20000" in body


def test_spy_tracks_only_known_structured_fields(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy.ndjson")
    compact = spy._compact_event(
        {
            "event": "editorial_repartition",
            "stage": "source_hazards:x",
            "observed_input_tokens": 1000,
            "target_input_tokens": 500,
            "partition_count": 2,
            "secret_payload": "must-not-be-exposed",
        }
    )
    assert compact["event"] == "editorial_repartition"
    assert "secret_payload" not in compact


def test_spy_treats_capacity_rejection_as_recoverable(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy.ndjson")
    spy._record(
        "app",
        '{"event":"application_result","application_status":"CAPACITY_REJECTED",'
        '"recovery_action":"REPARTITION"}',
    )
    assert spy.abort_reason is None


def test_spy_aborts_nonrecoverable_application_failure(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy.ndjson")
    spy._record(
        "app",
        '{"event":"application_result","application_status":"FAILED","error_type":"ValueError"}',
    )
    assert spy.abort_reason is not None
    assert "non-recoverable" in spy.abort_reason


def test_spy_aborts_underpartitioned_measured_request(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy.ndjson")
    spy._record(
        "app",
        '{"event":"editorial_repartition","observed_input_tokens":1000000,'
        '"target_input_tokens":250000,"partition_count":2,'
        '"ranges":[[0,50],[50,100]]}',
    )
    assert spy.abort_reason is not None
    assert "under-partitioned" in spy.abort_reason


def test_spy_accepts_valid_multiway_repartition(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy.ndjson")
    spy._record(
        "app",
        '{"event":"editorial_repartition","observed_input_tokens":1000000,'
        '"target_input_tokens":250000,"partition_count":4,'
        '"ranges":[[0,25],[25,50],[50,75],[75,100]]}',
    )
    assert spy.abort_reason is None


def test_spy_aborts_projection_expansion_and_nonshrinking_context(tmp_path: Path) -> None:
    module = _module()
    projection = module.ModalExecutionSpy(("app",), tmp_path / "projection.ndjson")
    projection._record(
        "app",
        '{"event":"editorial_evidence_projection","raw_event_count":10,'
        '"projected_event_count":11,"raw_serialized_bytes":100,'
        '"projected_serialized_bytes":101}',
    )
    assert projection.abort_reason is not None

    context = module.ModalExecutionSpy(("app",), tmp_path / "context.ndjson")
    context._record(
        "app",
        '{"event":"editorial_context_repartition","previous_range":[0,100],"next_range":[0,100]}',
    )
    assert context.abort_reason is not None


def test_spy_aborts_repeated_identical_request_plan_without_progress(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "repeat.ndjson")
    plan = (
        '{"event":"editorial_request_plan","task":"source_hazards:w0-w9",'
        '"input_tokens":1000,"context_limit_tokens":2000,'
        '"available_output_tokens":1000,"generation_budget_tokens":100,'
        '"serialized_request_bytes":5000}'
    )
    spy._record("app", plan)
    assert spy.abort_reason is None
    spy._record("app", plan)
    assert spy.abort_reason is not None
    assert "without forward progress" in spy.abort_reason


def test_spy_allows_same_plan_after_producer_terminal_progress(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "progress.ndjson")
    plan = (
        '{"event":"editorial_request_plan","task":"semantic_cores:w0-w9",'
        '"input_tokens":1000,"context_limit_tokens":2000,'
        '"available_output_tokens":1000,"generation_budget_tokens":100,'
        '"serialized_request_bytes":5000}'
    )
    spy._record(
        "app",
        '{"event":"editorial_remote_call_start","invocation_id":"inv-1",'
        '"task":"semantic_cores:w0-w9"}',
    )
    spy._record("app", plan)
    spy._record(
        "app",
        '{"event":"editorial_remote_call_terminal","invocation_id":"inv-1",'
        '"task":"semantic_cores:w0-w9","status":"COMPLETE"}',
    )
    spy._record(
        "app",
        '{"event":"editorial_remote_call_start","invocation_id":"inv-2",'
        '"task":"semantic_cores:w0-w9"}',
    )
    spy._record("app", plan)
    assert spy.abort_reason is None


def test_spy_aborts_invalid_generation_plan(tmp_path: Path) -> None:
    module = _module()
    context = module.ModalExecutionSpy(("app",), tmp_path / "context-plan.ndjson")
    context._record(
        "app",
        '{"event":"editorial_request_plan","task":"x","input_tokens":100,'
        '"context_limit_tokens":100,"available_output_tokens":1,'
        '"generation_budget_tokens":1,"serialized_request_bytes":100}',
    )
    assert context.abort_reason is not None

    output = module.ModalExecutionSpy(("app",), tmp_path / "output-plan.ndjson")
    output._record(
        "app",
        '{"event":"editorial_request_plan","task":"x","input_tokens":50,'
        '"context_limit_tokens":100,"available_output_tokens":20,'
        '"generation_budget_tokens":21,"serialized_request_bytes":100}',
    )
    assert output.abort_reason is not None


def test_spy_log_follow_uses_live_stream_without_historical_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy.ndjson")

    class _Process:
        stdout = iter(())
        returncode = 0

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("completed process should not be terminated")

    observed: list[list[str]] = []

    def popen(command, **_kwargs):
        observed.append(command)
        return _Process()

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    spy.request_stop()
    spy._follow("app")
    assert observed
    command = observed[0]
    assert "--follow" in command
    assert "--since" not in command
    assert "--until" not in command
    assert "--show-function-call-id" in command


def test_spy_aborts_if_active_log_follower_reaches_clean_eof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "spy-eof.ndjson")

    class _Process:
        stdout = iter(())
        returncode = 0

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("completed process should not be terminated")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())

    spy._follow("app")

    assert spy.abort_reason is not None
    assert "exited before explicit stop" in spy.abort_reason
    assert "returncode=0" in spy.abort_reason
    assert spy.stop.is_set()


def test_spy_accepts_measured_capacity_probe(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "probe.ndjson")
    spy._record(
        "app",
        '{"event":"editorial_capacity_probe","status":"CAPACITY_REJECTED",'
        '"task":"source_hazards:acceptance_probe:video","input_tokens":4239373,'
        '"context_limit_tokens":262144,"serialized_request_bytes":10000000}',
    )
    assert spy.abort_reason is None
    assert spy.event_counts["editorial_capacity_probe"] == 1


def test_spy_rejects_capacity_probe_without_measured_context(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "bad-probe.ndjson")
    spy._record(
        "app",
        '{"event":"editorial_capacity_probe","status":"CAPACITY_REJECTED","input_tokens":4239373}',
    )
    assert spy.abort_reason is not None
    assert "omitted measured token/context" in spy.abort_reason


def test_spy_scopes_pipeline_and_model_events_to_one_execution(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "scoped.ndjson",
        root_function_call_id="fc-ABC123",
        execution_id="exec-123",
    )

    spy._record(
        "clipper-production-pipeline",
        "2026-08-28T12:00:00Z fc-OTHER "
        '{"event":"editorial_evidence_projection","raw_event_count":10,'
        '"projected_event_count":1,"raw_serialized_bytes":100,'
        '"projected_serialized_bytes":10}',
    )
    spy._record(
        "clipper-production-pipeline",
        "2026-08-28T12:00:01Z fc-ABC123 "
        '{"event":"editorial_evidence_projection","raw_event_count":10,'
        '"projected_event_count":1,"raw_serialized_bytes":100,'
        '"projected_serialized_bytes":10}',
    )
    spy._record(
        "clipper-open-editor",
        "2026-08-28T12:00:02Z fc-CHILD1 "
        '{"event":"editorial_generation_complete","execution_id":"other",'
        '"task":"source_hazards:x","duration_seconds":1}',
    )
    spy._record(
        "clipper-open-editor",
        "2026-08-28T12:00:03Z fc-CHILD2 "
        '{"event":"editorial_generation_complete","execution_id":"exec-123",'
        '"task":"source_hazards:x","duration_seconds":1}',
    )

    assert spy.event_counts == {
        "editorial_evidence_projection": 1,
        "editorial_generation_complete": 1,
    }


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_spy_rejects_non_finite_generation_stall_timeout(
    tmp_path: Path,
    monkeypatch,
    value: str,
) -> None:
    module = _module()
    monkeypatch.setenv("CLIPPER_MODAL_GENERATION_STALL_SECONDS", value)
    with pytest.raises(ValueError, match="finite and positive"):
        module.ModalExecutionSpy(("app",), tmp_path / "bad-stall.ndjson")


def test_spy_aborts_editorial_remote_call_that_exceeds_progress_deadline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "stall.ndjson",
        generation_stall_seconds=10,
        execution_id="exec-123",
    )
    clock = {"now": 100.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"source_hazards:x"}',
    )
    assert spy.abort_reason is None
    clock["now"] = 111.0
    spy._check_stalled_editorial_calls()
    assert spy.abort_reason is not None
    assert "no terminal progress" in spy.abort_reason
    assert spy.abort_event["invocation_id"] == "inv-1"


def test_spy_aborts_explicit_editorial_execution_timeout(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "timeout.ndjson")
    spy._record(
        "app",
        '{"event":"editorial_execution_timeout","task":"source_hazards:x",'
        '"timeout_seconds":900,"recovery_action":"REPARTITION"}',
    )
    assert spy.abort_reason is not None
    assert "runtime timeout" in spy.abort_reason


def test_spy_requires_closed_pipeline_editorial_calls_before_terminal_barrier(
    tmp_path: Path,
) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "terminal.ndjson",
        execution_id="exec-123",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"source_hazards:x"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_terminal","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"source_hazards:x","status":"COMPLETE"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"production_cycle_terminal","execution_id":"exec-123",'
        '"status":"PASS","pipeline_status":"SUCCESS","review_status":"NOT_RENDERED"}',
    )

    assert spy.wait_for_producer_barrier(timeout_seconds=0.1) is True
    summary = spy.summary()
    assert summary["terminal_seen"] is True
    assert summary["terminal_event"]["status"] == "PASS"
    assert summary["active_editorial_calls"] == []


def test_spy_rejects_production_terminal_while_editorial_call_is_active(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "active-at-terminal.ndjson",
        execution_id="exec-123",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"quality_windows:x"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"production_cycle_terminal","execution_id":"exec-123","status":"PASS"}',
    )

    assert spy.abort_reason is not None
    assert "active editorial calls" in spy.abort_reason
    assert spy.summary()["terminal_seen"] is False


def test_late_model_diagnostic_cannot_invalidate_closed_pipeline_barrier(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "late-diagnostic.ndjson",
        execution_id="exec-123",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"quality_windows:x"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_terminal","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"quality_windows:x","status":"COMPLETE"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"production_cycle_terminal","execution_id":"exec-123","status":"PASS"}',
    )
    assert spy.wait_for_producer_barrier(timeout_seconds=0.1) is True

    spy._record(
        "clipper-open-editor",
        '{"event":"editorial_execution_timeout","execution_id":"exec-123",'
        '"task":"quality_windows:x","timeout_seconds":900,"recovery_action":"REPARTITION"}',
    )

    assert spy.abort_reason is None
    assert spy.summary()["diagnostic_event_counts"]["editorial_execution_timeout"] == 1
    assert spy.wait_for_producer_barrier(timeout_seconds=0.1) is True


def test_spy_cannot_pass_terminal_drain_without_terminal_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    clock = {"now": 10.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    spy = module.ModalExecutionSpy(("app",), tmp_path / "missing-terminal.ndjson")

    assert spy.wait_for_producer_barrier(timeout_seconds=0.5) is False


def test_spy_retains_execution_scoped_editorial_runtime_metadata(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "runtime-metadata.ndjson",
        execution_id="exec-123",
    )
    spy._record(
        "clipper-open-editor",
        '{"event":"editorial_request_plan","execution_id":"exec-123",'
        '"invocation_id":"inv-runtime","task":"source_hazards:x",'
        '"input_tokens":12345,"context_limit_tokens":262144,'
        '"available_output_tokens":249799,"generation_budget_tokens":512,'
        '"runtime_safe_input_tokens":32768,"capacity_repartitionable":true,'
        '"serialized_request_bytes":50000,"outlines_version":"1.3.0",'
        '"transformers_version":"4.57.3","generation_deadline_seconds":300,'
        '"execution_timeout_seconds":900}',
    )

    summary = spy.summary()
    latest = summary["latest"]["editorial_request_plan"]
    assert latest["outlines_version"] == "1.3.0"
    assert latest["transformers_version"] == "4.57.3"
    assert latest["runtime_safe_input_tokens"] == 32768
    assert latest["generation_deadline_seconds"] == 300
    assert latest["execution_timeout_seconds"] == 900
    assert summary["diagnostic_event_counts"]["editorial_request_plan"] == 1


def test_spy_stall_snapshot_is_safe_during_concurrent_terminal_record(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(
        ("clipper-production-pipeline",),
        tmp_path / "concurrent-stall.ndjson",
        generation_stall_seconds=999,
        execution_id="exec-123",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-123",'
        '"invocation_id":"inv-1","task":"source_hazards:x"}',
    )

    started = threading.Event()
    finished = threading.Event()

    class SlowItemsDict(dict):
        def items(self):
            started.set()
            time.sleep(0.02)
            return super().items()

    with spy.lock:
        spy._active_editorial_calls = SlowItemsDict(spy._active_editorial_calls)

    def terminal() -> None:
        started.wait(timeout=1)
        spy._record(
            "clipper-production-pipeline",
            '{"event":"editorial_remote_call_terminal","execution_id":"exec-123",'
            '"invocation_id":"inv-1","task":"source_hazards:x","status":"COMPLETE"}',
        )
        finished.set()

    worker = threading.Thread(target=terminal)
    worker.start()
    spy._check_stalled_editorial_calls()
    worker.join(timeout=1)

    assert finished.is_set()
    assert spy.abort_reason is None
    assert spy.summary()["active_editorial_calls"] == []
