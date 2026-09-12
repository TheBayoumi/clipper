from __future__ import annotations

import importlib.util
from pathlib import Path


def _spy_module():
    path = Path("scripts/modal_execution_spy.py")
    spec = importlib.util.spec_from_file_location("modal_execution_spy_hilp129", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Modal execution spy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_assigns_one_editorial_producer_lifecycle_per_root_attempt() -> None:
    pipeline = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    provider = Path("src/clipper/providers/modal.py").read_text(encoding="utf-8")
    assert "editorial_producer_lifecycle_id = uuid.uuid4().hex" in pipeline
    assert '"CLIPPER_EDITORIAL_PRODUCER_LIFECYCLE_ID": editorial_producer_lifecycle_id' in pipeline
    assert 'os.getenv("CLIPPER_EDITORIAL_PRODUCER_LIFECYCLE_ID", "").strip()' in provider
    assert '"producer_lifecycle_id": invocation.producer_lifecycle_id' in provider
    assert '"producer_lifecycle_id": self.producer_lifecycle_id' in provider


def test_spy_reconciles_hilp129_orphan_only_after_recoverable_result_and_replacement(
    tmp_path: Path,
) -> None:
    module = _spy_module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "hilp129.ndjson",
        execution_id="exec-129",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-old","producer_lifecycle_id":"producer-a",'
        '"task":"source_hazards:w0011662-w0012128"}',
    )
    spy._record(
        "clipper-open-editor",
        '{"event":"application_result","execution_id":"exec-129",'
        '"invocation_id":"inv-old","application_status":"CAPACITY_REJECTED",'
        '"error_type":"EditorialCapacityError","recovery_action":"REPARTITION"}',
    )
    assert spy.summary()["active_editorial_calls"] == ["inv-old"]

    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-new","producer_lifecycle_id":"producer-b",'
        '"task":"source_hazards:w0000000-w0012594"}',
    )

    summary = spy.summary()
    assert spy.abort_reason is None
    assert summary["active_editorial_calls"] == ["inv-new"]
    assert len(summary["reconciled_editorial_calls"]) == 1
    reconciled = summary["reconciled_editorial_calls"][0]
    assert reconciled["invocation_id"] == "inv-old"
    assert reconciled["application_status"] == "CAPACITY_REJECTED"
    assert reconciled["producer_lifecycle_id"] == "producer-a"
    assert reconciled["replacement_producer_lifecycle_id"] == "producer-b"

    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_terminal","execution_id":"exec-129",'
        '"invocation_id":"inv-new","producer_lifecycle_id":"producer-b",'
        '"task":"source_hazards:w0000000-w0012594","status":"CAPACITY_REJECTED"}',
    )
    assert spy.summary()["active_editorial_calls"] == []


def test_spy_does_not_reconcile_unknown_orphan_on_producer_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _spy_module()
    clock = {"now": 100.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "unknown.ndjson",
        execution_id="exec-129",
        generation_stall_seconds=10.0,
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-unknown","producer_lifecycle_id":"producer-a",'
        '"task":"source_hazards:unknown"}',
    )
    clock["now"] = 101.0
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-new","producer_lifecycle_id":"producer-b",'
        '"task":"source_hazards:new"}',
    )
    assert spy.summary()["active_editorial_calls"] == ["inv-new", "inv-unknown"]
    assert spy.summary()["reconciled_editorial_calls"] == []

    clock["now"] = 111.0
    spy._check_stalled_editorial_calls()
    assert spy.abort_reason is not None
    assert spy.abort_event["invocation_id"] == "inv-unknown"


def test_spy_same_producer_lifecycle_cannot_hide_unclosed_call(
    tmp_path: Path,
) -> None:
    module = _spy_module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "same-producer.ndjson",
        execution_id="exec-129",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-old","producer_lifecycle_id":"producer-a",'
        '"task":"source_hazards:old"}',
    )
    spy._record(
        "clipper-open-editor",
        '{"event":"application_result","execution_id":"exec-129",'
        '"invocation_id":"inv-old","application_status":"CAPACITY_REJECTED",'
        '"error_type":"EditorialCapacityError","recovery_action":"REPARTITION"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-new","producer_lifecycle_id":"producer-a",'
        '"task":"source_hazards:new"}',
    )
    assert spy.abort_reason is None
    assert spy.summary()["active_editorial_calls"] == ["inv-new", "inv-old"]
    assert spy.summary()["reconciled_editorial_calls"] == []


def test_spy_rejects_terminal_from_wrong_producer_lifecycle(tmp_path: Path) -> None:
    module = _spy_module()
    spy = module.ModalExecutionSpy(
        ("clipper-production-pipeline",),
        tmp_path / "mismatch.ndjson",
        execution_id="exec-129",
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_start","execution_id":"exec-129",'
        '"invocation_id":"inv-1","producer_lifecycle_id":"producer-a",'
        '"task":"source_hazards:x"}',
    )
    spy._record(
        "clipper-production-pipeline",
        '{"event":"editorial_remote_call_terminal","execution_id":"exec-129",'
        '"invocation_id":"inv-1","producer_lifecycle_id":"producer-b",'
        '"task":"source_hazards:x","status":"COMPLETE"}',
    )
    assert spy.abort_reason is not None
    assert "lifecycle does not match" in spy.abort_reason
