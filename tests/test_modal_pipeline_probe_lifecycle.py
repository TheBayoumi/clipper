from __future__ import annotations

import ast
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_probe_runner() -> Callable[..., dict[str, Any]]:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_editorial_capacity_probe"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "Any": Any,
        "json": json,
        "os": os,
        "uuid": uuid,
    }
    exec(compile(isolated, "scripts/modal_pipeline.py", "exec"), namespace)  # noqa: S102
    loaded = namespace["_run_editorial_capacity_probe"]
    assert callable(loaded)
    return loaded


class _Remote:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def remote(self, request: dict[str, Any]) -> object:
        self.requests.append(dict(request))
        if self.error is not None:
            raise self.error
        return self.response


def test_acceptance_capacity_probe_is_closed_by_producer_barrier(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_probe_runner()
    remote = _Remote(
        {
            "value": {
                "status": "CAPACITY_REJECTED",
                "reason": "context_exhausted",
                "input_tokens": 761_756,
                "context_limit_tokens": 262_144,
                "serialized_request_bytes": 8_954_610,
            }
        }
    )
    worker = SimpleNamespace(capacity_probe=remote)
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-probe")
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", "a" * 40)

    details = runner(
        worker,
        task="source_hazards:acceptance_probe:video",
        raw_payload={"capacity_repartitionable": True},
    )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert [event["event"] for event in events] == [
        "editorial_remote_call_start",
        "editorial_remote_call_terminal",
    ]
    start, terminal = events
    assert start["execution_id"] == terminal["execution_id"] == "exec-probe"
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "CAPACITY_REJECTED"
    assert terminal["input_tokens"] == 761_756
    assert terminal["context_limit_tokens"] == 262_144
    assert details["status"] == "CAPACITY_REJECTED"

    request = remote.requests[0]
    assert request["execution_id"] == "exec-probe"
    assert request["expected_git_sha"] == "a" * 40
    assert request["editorial_invocation_id"] == start["invocation_id"]


def test_acceptance_capacity_probe_emits_error_terminal_on_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_probe_runner()
    remote = _Remote(error=RuntimeError("synthetic probe transport failure"))
    worker = SimpleNamespace(capacity_probe=remote)
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-failure")
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", "b" * 40)

    with pytest.raises(RuntimeError, match="synthetic probe transport failure"):
        runner(
            worker,
            task="source_hazards:acceptance_probe:video",
            raw_payload={"capacity_repartitionable": True},
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert [event["event"] for event in events] == [
        "editorial_remote_call_start",
        "editorial_remote_call_terminal",
    ]
    start, terminal = events
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "ERROR"
    assert terminal["error_type"] == "RuntimeError"
    assert terminal["reason"] == "acceptance_capacity_probe_failed"


def test_acceptance_capacity_probe_emits_error_terminal_on_structured_remote_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_probe_runner()
    remote = _Remote(
        {
            "error": {
                "type": "RuntimeError",
                "message": "synthetic model error",
                "details": {},
            }
        }
    )
    worker = SimpleNamespace(capacity_probe=remote)
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-structured")
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", "c" * 40)

    with pytest.raises(RuntimeError, match=r"EditorialModel\.capacity_probe failed"):
        runner(
            worker,
            task="source_hazards:acceptance_probe:video",
            raw_payload={"capacity_repartitionable": True},
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start, terminal = events
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "ERROR"
