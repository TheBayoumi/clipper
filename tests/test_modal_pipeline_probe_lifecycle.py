from __future__ import annotations

import json
from typing import Any

import pytest

from clipper.providers.modal import invoke_editorial_capacity_probe


class _Remote:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> object:
        self.requests.append(dict(request))
        if self.error is not None:
            raise self.error
        return self.response


def test_acceptance_capacity_probe_is_closed_by_shared_producer_barrier(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-probe")

    response = invoke_editorial_capacity_probe(
        remote,
        task="source_hazards:acceptance_probe:video",
        payload={"capacity_repartitionable": True},
        expected_git_sha="a" * 40,
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
    assert response["value"]["status"] == "CAPACITY_REJECTED"

    request = remote.requests[0]
    assert request["execution_id"] == "exec-probe"
    assert request["expected_git_sha"] == "a" * 40
    assert request["editorial_invocation_id"] == start["invocation_id"]


def test_acceptance_capacity_probe_emits_error_terminal_on_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = _Remote(error=RuntimeError("synthetic probe transport failure"))
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-failure")

    with pytest.raises(RuntimeError, match="synthetic probe transport failure"):
        invoke_editorial_capacity_probe(
            remote,
            task="source_hazards:acceptance_probe:video",
            payload={"capacity_repartitionable": True},
            expected_git_sha="b" * 40,
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start, terminal = events
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "ERROR"
    assert terminal["error_type"] == "RuntimeError"
    assert terminal["reason"] == "capacity_probe_remote_exception"


def test_acceptance_capacity_probe_emits_timeout_terminal_for_modal_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FunctionTimeoutError(Exception):
        pass

    remote = _Remote(error=FunctionTimeoutError("synthetic timeout"))
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-timeout")

    with pytest.raises(FunctionTimeoutError, match="synthetic timeout"):
        invoke_editorial_capacity_probe(
            remote,
            task="source_hazards:acceptance_probe:video",
            payload={"capacity_repartitionable": True},
            expected_git_sha="c" * 40,
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start, terminal = events
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "TIMEOUT"
    assert terminal["error_type"] == "FunctionTimeoutError"


def test_acceptance_capacity_probe_emits_error_terminal_on_structured_remote_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = _Remote(
        {
            "error": {
                "type": "RuntimeError",
                "message": "synthetic model error",
                "details": {},
            }
        }
    )
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-structured")

    with pytest.raises(RuntimeError, match=r"EditorialModel\.capacity_probe failed"):
        invoke_editorial_capacity_probe(
            remote,
            task="source_hazards:acceptance_probe:video",
            payload={"capacity_repartitionable": True},
            expected_git_sha="d" * 40,
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start, terminal = events
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "ERROR"
    assert terminal["reason"] == "capacity_probe_remote_error"


def test_acceptance_capacity_probe_rejects_invalid_measurement_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = _Remote({"value": {"status": "UNKNOWN"}})
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-invalid")

    with pytest.raises(RuntimeError, match="returned invalid status"):
        invoke_editorial_capacity_probe(
            remote,
            task="source_hazards:acceptance_probe:video",
            payload={"capacity_repartitionable": True},
            expected_git_sha="e" * 40,
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start, terminal = events
    assert start["invocation_id"] == terminal["invocation_id"]
    assert terminal["status"] == "ERROR"
    assert terminal["reason"] == "capacity_probe_invalid_status"
