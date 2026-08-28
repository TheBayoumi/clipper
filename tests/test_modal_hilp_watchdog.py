from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    scripts = str(Path("scripts").resolve())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = Path("scripts/modal_hilp_watchdog.py")
    spec = importlib.util.spec_from_file_location("modal_hilp_watchdog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load modal HILP watchdog")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment(tmp_path: Path, monkeypatch) -> None:
    evidence = tmp_path / "source-master.json"
    evidence.write_text('{"sha256":"source"}\n', encoding="utf-8")
    brief = tmp_path / "brief.yaml"
    brief.write_text("campaign_id: test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "open-evidence").mkdir(exist_ok=True)
    (tmp_path / "open-evidence" / "source-master.json").write_text(
        evidence.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIPPER_TARGET_VIDEO_ID", "video")
    monkeypatch.setenv("CLIPPER_TARGET_CHANNEL_ID", "channel")
    monkeypatch.setenv("CLIPPER_TARGET_VIDEO_URL", "https://example.test/video")
    monkeypatch.setenv("CLIPPER_CAMPAIGN_BRIEF", str(brief))
    monkeypatch.setenv("CLIPPER_FRESH_INFERENCE", "false")
    monkeypatch.setenv("CLIPPER_RESUME_FROM_RUN_ID", "prior-run")
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", "a" * 40)
    monkeypatch.setenv("CLIPPER_EXECUTION_MODE", "resume")
    monkeypatch.setenv("CLIPPER_MAX_GPU_SECONDS", "1000")
    monkeypatch.setenv("CLIPPER_MAX_ESTIMATED_USD", "100")
    monkeypatch.setenv("CLIPPER_MODAL_APP", "models")
    monkeypatch.setenv("CLIPPER_MODAL_PIPELINE_APP", "pipeline")
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "0.001")


class _Spy:
    instance = None

    def __init__(self, apps, output, **kwargs) -> None:
        self.apps = apps
        self.output = output
        self.abort_reason = None
        self.stopped = False
        self.root_function_call_id = kwargs.get("root_function_call_id")
        self.execution_id = kwargs.get("execution_id")
        _Spy.instance = self

    def run(self) -> int:
        return 0

    def request_stop(self, *_args) -> None:
        self.stopped = True

    def wait_for_producer_barrier(self, **_kwargs) -> bool:
        return self.abort_reason is None

    def summary(self) -> dict[str, object]:
        return {
            "status": "ABORT" if self.abort_reason else "PASS",
            "abort_reason": self.abort_reason,
            "events_seen": 1,
            "event_counts": {"production_cycle_terminal": 1},
            "terminal_seen": True,
            "terminal_event": {"status": "PASS"},
            "active_editorial_calls": [],
        }


class _Call:
    object_id = "fc-test"

    def __init__(self, result: dict[str, object], *, abort_spy: bool = False) -> None:
        self.result = result
        self.abort_spy = abort_spy
        self.cancel_args: list[bool] = []
        self.polls = 0

    def hydrate(self) -> None:
        return None

    def get(self, *, timeout: float):
        assert timeout > 0
        self.polls += 1
        if self.abort_spy and _Spy.instance is not None:
            _Spy.instance.abort_reason = "synthetic bad telemetry"
            raise TimeoutError
        return self.result

    def cancel(self, *, terminate_containers: bool) -> None:
        self.cancel_args.append(terminate_containers)


def _modal(call: _Call):
    class _Function:
        @staticmethod
        def from_name(app: str, function: str):
            assert app == "pipeline"
            assert function == "run_full_cycle"
            return SimpleNamespace(spawn=lambda request: call)

    return SimpleNamespace(Function=_Function)


def test_watchdog_returns_successful_editorial_only_result(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))

    call = _Call(
        {
            "status": "PASS",
            "execution_mode": "resume",
            "execution_id": "e" * 32,
            "deployed_git_sha": "a" * 40,
            "pipeline_status": "SUCCESS",
            "review_status": "NOT_RENDERED",
            "run_volume": "volume",
            "run_path": "/run",
        }
    )
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    result = module.run(render=False)
    assert result["status"] == "PASS"
    metadata = json.loads(
        (tmp_path / "open-evidence" / "modal-function-call.json").read_text(encoding="utf-8")
    )
    assert metadata["function_call_id"] == "fc-test"
    assert metadata["resume_from_run_id"] == "prior-run"
    assert metadata["fresh_inference"] is False
    assert metadata["render"] is False
    assert metadata["editorial_acceptance_probe"] is True
    assert call.cancel_args == []


def test_watchdog_cancels_exact_call_without_terminating_containers_on_spy_abort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))

    call = _Call({}, abort_spy=True)
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    try:
        module.run(render=False)
    except RuntimeError as exc:
        assert "Modal spy aborted production early" in str(exc)
    else:
        raise AssertionError("watchdog should abort")

    assert call.cancel_args == [False]
    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["call_cancelled"] is True
    assert summary["function_call_id"] == "fc-test"


def test_watchdog_rechecks_budget_after_successful_final_poll(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    monkeypatch.setenv("CLIPPER_MAX_GPU_SECONDS", "1")
    clock = {"now": 0.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    class BudgetCall(_Call):
        def get(self, *, timeout: float):
            assert timeout > 0
            clock["now"] = 1.0
            return {
                "status": "PASS",
                "execution_mode": "resume",
                "execution_id": "e" * 32,
                "deployed_git_sha": "a" * 40,
                "pipeline_status": "SUCCESS",
                "review_status": "NOT_RENDERED",
                "run_volume": "volume",
                "run_path": "/run",
            }

    call = BudgetCall({})
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    try:
        module.run(render=False)
    except RuntimeError as exc:
        assert "GPU budget exceeded by completed production call" in str(exc)
    else:
        raise AssertionError("final successful poll must still enforce the GPU budget")

    assert call.cancel_args == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CLIPPER_MAX_GPU_SECONDS", "nan"),
        ("CLIPPER_MAX_GPU_SECONDS", "inf"),
        ("CLIPPER_MAX_GPU_SECONDS", "-inf"),
        ("CLIPPER_MAX_ESTIMATED_USD", "nan"),
        ("CLIPPER_MAX_ESTIMATED_USD", "inf"),
        ("CLIPPER_MAX_ESTIMATED_USD", "-inf"),
    ],
)
def test_watchdog_rejects_nonfinite_production_budgets(
    tmp_path: Path,
    monkeypatch,
    name: str,
    value: str,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "modal", _modal(_Call({})))
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="finite and positive"):
        module.run(render=False)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_watchdog_rejects_nonfinite_poll_interval(
    tmp_path: Path,
    monkeypatch,
    value: str,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    monkeypatch.setitem(sys.modules, "modal", _modal(_Call({})))
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", value)

    with pytest.raises(ValueError, match="finite and positive"):
        module.run(render=False)
