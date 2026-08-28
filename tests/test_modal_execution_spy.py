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
