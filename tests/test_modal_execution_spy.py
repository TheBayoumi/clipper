from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_spy_allows_same_plan_after_generation_completion(tmp_path: Path) -> None:
    module = _module()
    spy = module.ModalExecutionSpy(("app",), tmp_path / "progress.ndjson")
    plan = (
        '{"event":"editorial_request_plan","task":"semantic_cores:w0-w9",'
        '"input_tokens":1000,"context_limit_tokens":2000,'
        '"available_output_tokens":1000,"generation_budget_tokens":100,'
        '"serialized_request_bytes":5000}'
    )
    spy._record("app", plan)
    spy._record(
        "app",
        '{"event":"editorial_generation_complete","task":"semantic_cores:w0-w9"}',
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
