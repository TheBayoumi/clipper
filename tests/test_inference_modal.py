"""Modal execution must close bounded calls and refuse unknown terminal states."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from clipper_engine.inference.modal import ModalVisionProvider
from clipper_engine.inference.vision import VisionSettings


class InputCancellation(RuntimeError):
    pass


class FakeCall:
    def __init__(self, responses: list[Any], *, cancel_error: Exception | None = None) -> None:
        self.responses = responses
        self.cancelled = False
        self.terminate_containers = False
        self.cancel_error = cancel_error

    def get(self, timeout: float) -> dict[str, Any]:
        _ = timeout
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, dict)
        return value

    def cancel(self, *, terminate_containers: bool = False) -> None:
        self.cancelled = True
        self.terminate_containers = terminate_containers
        if self.cancel_error is not None:
            raise self.cancel_error


def _provider(monkeypatch: pytest.MonkeyPatch, call: FakeCall) -> ModalVisionProvider:
    method = SimpleNamespace(spawn=lambda request: call)
    worker = SimpleNamespace(inspect=method)
    fake = SimpleNamespace(Cls=SimpleNamespace(from_name=lambda _app, _name: lambda: worker))
    old = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake if name == "modal" else old(name),
    )
    return ModalVisionProvider(VisionSettings(enabled=True, deployed_sha="a" * 40))


def test_modal_vision_returns_valid_object(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, FakeCall([{"observations": []}]))
    assert provider.inspect({"task": "source_gameplay_visual_scout"}) == {"observations": []}


def test_modal_vision_cancelled_deadline_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    call = FakeCall([TimeoutError(), InputCancellation("cancelled")])
    provider = _provider(monkeypatch, call)
    with pytest.raises(RuntimeError, match="bounded inference deadline"):
        provider.inspect({"task": "source_gameplay_visual_scout"})
    assert call.cancelled
    assert call.terminate_containers


def test_modal_vision_refuses_unconfirmed_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    call = FakeCall([TimeoutError(), TimeoutError()])
    provider = _provider(monkeypatch, call)
    with pytest.raises(RuntimeError, match="terminal state unconfirmed"):
        provider.inspect({"task": "source_gameplay_visual_scout"})
    assert call.cancelled
    assert call.terminate_containers


def test_modal_vision_refuses_unknown_cancellation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    call = FakeCall([TimeoutError(), RuntimeError("lost control plane")])
    provider = _provider(monkeypatch, call)
    with pytest.raises(RuntimeError, match="terminal state unconfirmed"):
        provider.inspect({"task": "source_gameplay_visual_scout"})


def test_modal_vision_refuses_failed_cancel_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    call = FakeCall([TimeoutError()], cancel_error=RuntimeError("cancel failed"))
    provider = _provider(monkeypatch, call)
    with pytest.raises(RuntimeError, match="terminal state unconfirmed"):
        provider.inspect({"task": "source_gameplay_visual_scout"})


def test_modal_vision_refuses_non_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    call = FakeCall([{"other": True}])
    provider = _provider(monkeypatch, call)
    assert provider.inspect({}) == {"other": True}  # Parser owns schema validation.


def test_modal_readiness_is_bounded_and_source_stamped() -> None:
    from clipper_engine.inference.modal import bounded_modal_readiness

    call = FakeCall([{"deployed_git_sha": "a" * 40, "gpu_count": 2}])
    worker = SimpleNamespace(ready=SimpleNamespace(spawn=lambda: call))
    ready = bounded_modal_readiness(worker, stage="gpu_smoke", deadline_seconds=120)
    assert ready["gpu_count"] == 2
    assert not call.cancelled


def test_modal_readiness_timeout_cancels_container_and_fails_closed() -> None:
    from clipper_engine.inference.modal import bounded_modal_readiness

    call = FakeCall([TimeoutError(), InputCancellation("cancelled")])
    worker = SimpleNamespace(ready=SimpleNamespace(spawn=lambda: call))
    with pytest.raises(RuntimeError, match="gpu_smoke readiness exceeded 120s"):
        bounded_modal_readiness(worker, stage="gpu_smoke", deadline_seconds=120)
    assert call.cancelled and call.terminate_containers


def test_modal_readiness_unconfirmed_cancellation_is_rejected() -> None:
    from clipper_engine.inference.modal import bounded_modal_readiness

    call = FakeCall([TimeoutError(), TimeoutError()])
    worker = SimpleNamespace(ready=SimpleNamespace(spawn=lambda: call))
    with pytest.raises(RuntimeError, match="readiness cancellation terminal state unconfirmed"):
        bounded_modal_readiness(worker, stage="source_wide", deadline_seconds=120)
    assert call.cancelled and call.terminate_containers


def test_modal_readiness_rejects_non_object_and_unbounded_deadline() -> None:
    from clipper_engine.inference.modal import bounded_modal_readiness

    worker = SimpleNamespace(ready=SimpleNamespace(spawn=lambda: FakeCall([{"ok": True}])))
    with pytest.raises(ValueError, match="readiness deadline"):
        bounded_modal_readiness(worker, stage="production_source", deadline_seconds=901)
    bad = SimpleNamespace(
        ready=SimpleNamespace(spawn=lambda: SimpleNamespace(get=lambda timeout: None))
    )
    with pytest.raises(RuntimeError, match="non-object"):
        bounded_modal_readiness(bad, stage="gpu_smoke", deadline_seconds=60)


def test_modal_readiness_reports_wall_time_without_changing_remote_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clipper_engine.inference.modal import bounded_modal_readiness

    call = FakeCall([{"deployed_git_sha": "a" * 40, "gpu_count": 2}])
    worker = SimpleNamespace(ready=SimpleNamespace(spawn=lambda: call))
    result = bounded_modal_readiness(worker, stage="source_wide", deadline_seconds=120)
    assert result["deployed_git_sha"] == "a" * 40
    assert 0 <= result["readiness_wall_seconds"] <= 120
    assert "wall_seconds=" in capsys.readouterr().out


def test_modal_readiness_unconfirmed_cancellation_has_structured_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clipper_engine.inference.modal import bounded_modal_readiness

    call = FakeCall([TimeoutError(), TimeoutError()])
    worker = SimpleNamespace(ready=SimpleNamespace(spawn=lambda: call))
    with pytest.raises(RuntimeError, match="terminal state unconfirmed"):
        bounded_modal_readiness(worker, stage="source_wide", deadline_seconds=120)
    output = capsys.readouterr().out
    assert '"event": "modal_readiness_timeout"' in output
    assert '"event": "modal_readiness_cancellation_unconfirmed"' in output
    assert call.cancelled and call.terminate_containers
