from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from clipper.modal_execution import ProductionCallSubmissionFailed


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

    def synthetic_recoverable_call(function, request, *, budget, **_kwargs):
        call = function._test_call
        return call, module.time.monotonic(), None

    module._spawn_recoverable_modal_call = synthetic_recoverable_call
    return module


def _environment(tmp_path: Path, monkeypatch) -> None:
    evidence = tmp_path / "source-master.json"
    evidence.write_text(
        json.dumps(
            {
                "video_id": "video",
                "channel_id": "channel",
                "canonical_url": "https://www.youtube.com/watch?v=video",
                "quality_policy": "highest_available_no_transcode",
                "sha256": "s" * 64,
                "volume_path": "/inputs/source.mkv",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    brief = tmp_path / "brief.yaml"
    brief.write_text(
        "campaign_id: test\n"
        "title: Test\n"
        "objective: Test production target\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: video\n"
        "      channel_id: channel\n"
        "      url: https://www.youtube.com/watch?v=video\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [channel]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "open-evidence").mkdir(exist_ok=True)
    (tmp_path / "open-evidence" / "source-master.json").write_text(
        evidence.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIPPER_TARGET_VIDEO_ID", "video")
    monkeypatch.setenv("CLIPPER_TARGET_CHANNEL_ID", "channel")
    monkeypatch.setenv("CLIPPER_TARGET_VIDEO_URL", "https://www.youtube.com/watch?v=video")
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


def test_scoped_brief_keeps_only_selected_authorized_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    brief_path = Path(str(module.os.environ["CLIPPER_CAMPAIGN_BRIEF"]))
    payload = yaml.safe_load(brief_path.read_text(encoding="utf-8"))
    payload["targets"]["videos"].append(
        {
            "video_id": "other",
            "channel_id": "channel",
            "url": "https://www.youtube.com/watch?v=other",
        }
    )
    brief_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    scoped = yaml.safe_load(module._scoped_brief_yaml())
    assert [item["video_id"] for item in scoped["targets"]["videos"]] == ["video"]
    assert scoped["rights"]["authorized_channels"] == ["channel"]


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
        while not self.stopped:
            time.sleep(0.001)
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
            "terminal_event": {
                "status": "PASS",
                "execution_id": self.execution_id,
                "pipeline_status": "SUCCESS",
                "review_status": "NOT_RENDERED",
            },
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
            return SimpleNamespace(_test_call=call)

    return SimpleNamespace(Function=_Function)


def test_source_payload_persists_failed_attempt_and_budget_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    (tmp_path / "open-evidence" / "source-master.json").unlink()

    class _Function:
        @staticmethod
        def from_name(app: str, function: str):
            assert app == "pipeline"
            assert function == "acquire_source"
            return object()

    def fail_acquisition(
        _function,
        _candidate,
        *,
        budget,
        attempt_evidence,
        **_kwargs,
    ):
        attempt_evidence.append(
            {
                "egress": "cloud:gcp",
                "status": "FAIL",
                "phase": "invoke",
                "error_type": "RuntimeError",
            }
        )
        budget.charge(2.0, gpu_count=0.0, estimated_usd_per_second=0.01)
        raise RuntimeError("synthetic acquisition failure")

    monkeypatch.setattr(module, "_acquire_remote_source", fail_acquisition)
    budget = module._BudgetLedger(100.0, 1.0)

    with pytest.raises(RuntimeError, match="synthetic acquisition failure"):
        module._source_payload(
            SimpleNamespace(Function=_Function),
            budget,
            execution_id="e" * 32,
        )

    attempts = json.loads(
        (tmp_path / "open-evidence" / "source-egress-attempts.json").read_text(encoding="utf-8")
    )
    budget_evidence = json.loads(
        (tmp_path / "open-evidence" / "source-budget.json").read_text(encoding="utf-8")
    )
    assert attempts["reused_evidence"] is False
    assert attempts["attempts"][0]["status"] == "FAIL"
    assert budget_evidence["estimated_usd"] == pytest.approx(0.02)


def test_watchdog_marks_pre_root_failure_as_abort(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(
        module,
        "_source_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic pre-root failure")),
    )
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace())

    with pytest.raises(RuntimeError, match="synthetic pre-root failure"):
        module.run(render=False)

    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ABORT"
    assert "synthetic pre-root failure" in summary["abort_reason"]
    assert summary["function_call_id"] == ""


def test_watchdog_reconciles_root_submission_with_lost_ack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    call = _Call({})

    def lost_ack(function, request, *, budget, **_kwargs):
        assert function._test_call is call
        assert request["resume_from_run_id"] == "prior-run"
        return (
            call,
            module.time.monotonic(),
            ProductionCallSubmissionFailed(
                "fc-test",
                "ServiceError",
                "root input acknowledgement lost",
            ),
        )

    monkeypatch.setattr(module, "_spawn_recoverable_modal_call", lost_ack)
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(ProductionCallSubmissionFailed, match="fc-test"):
        module.run(render=False)

    assert call.cancel_args == [False]
    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ABORT"
    assert summary["function_call_id"] == "fc-test"
    assert summary["call_cancelled"] is True


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


def test_watchdog_rejects_terminal_evidence_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)

    class DriftSpy(_Spy):
        def summary(self) -> dict[str, object]:
            value = super().summary()
            terminal = dict(value["terminal_event"])
            terminal["execution_id"] = "wrong-execution"
            value["terminal_event"] = terminal
            return value

    monkeypatch.setattr(module, "ModalExecutionSpy", DriftSpy)
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

    with pytest.raises(RuntimeError, match="terminal evidence does not match"):
        module.run(render=False)

    summary = json.loads(
        (tmp_path / "open-evidence" / "modal-spy-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ABORT"
    assert "field=execution_id" in summary["abort_reason"]


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


def test_watchdog_counts_successful_hydration_against_compute_budget(
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

    class SlowHydrateCall(_Call):
        def hydrate(self) -> None:
            clock["now"] = 0.6

        def get(self, *, timeout: float):
            raise AssertionError(f"budget must fail before polling, got timeout={timeout}")

    call = SlowHydrateCall({})
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="production budget reached"):
        module.run(render=False)

    assert call.cancel_args == [False]


def test_watchdog_caps_poll_timeout_to_remaining_compute_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    monkeypatch.setenv("CLIPPER_MAX_GPU_SECONDS", "1")
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")
    clock = {"now": 0.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    class BudgetPollCall(_Call):
        def __init__(self) -> None:
            super().__init__({})
            self.timeouts: list[float] = []

        def get(self, *, timeout: float):
            self.timeouts.append(timeout)
            clock["now"] = timeout
            raise TimeoutError

    call = BudgetPollCall()
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="production budget reached"):
        module.run(render=False)

    assert call.timeouts == [pytest.approx(0.5)]
    assert call.cancel_args == [False]


def test_watchdog_retries_exact_call_cancellation_before_marking_cancelled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))

    class RetryCancelCall(_Call):
        def __init__(self) -> None:
            super().__init__({}, abort_spy=True)
            self.cancel_attempts = 0

        def cancel(self, *, terminate_containers: bool) -> None:
            assert terminate_containers is False
            self.cancel_attempts += 1
            if self.cancel_attempts == 1:
                raise RuntimeError("transient cancel failure")
            self.cancel_args.append(terminate_containers)

    call = RetryCancelCall()
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="Modal spy aborted production early"):
        module.run(render=False)

    assert call.cancel_attempts == 2
    assert call.cancel_args == [False]


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CLIPPER_MODAL_SPY_POLL_SECONDS", "nan"),
        ("CLIPPER_MODAL_SPY_POLL_SECONDS", "inf"),
        ("CLIPPER_MODAL_SPY_POLL_SECONDS", "-inf"),
        ("CLIPPER_MODAL_SPY_POLL_SECONDS", "0"),
        ("CLIPPER_MODAL_SPY_POLL_SECONDS", "-1"),
        ("CLIPPER_MODAL_SPY_POLL_SECONDS", "not-a-number"),
        ("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "nan"),
        ("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "inf"),
        ("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "-inf"),
        ("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "0"),
        ("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "-1"),
        ("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "not-a-number"),
    ],
)
def test_watchdog_rejects_invalid_timing_before_spawning(
    tmp_path: Path,
    monkeypatch,
    name: str,
    value: str,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setenv(name, value)

    class _ForbiddenFunction:
        @staticmethod
        def from_name(*_args, **_kwargs):
            raise AssertionError("Modal function lookup must not occur before timing validation")

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=_ForbiddenFunction))

    with pytest.raises(ValueError, match="finite and positive"):
        module.run(render=False)


def test_watchdog_cancels_spawned_call_if_hydration_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "ModalExecutionSpy", _Spy)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))

    class HydrationFailureCall(_Call):
        def hydrate(self) -> None:
            raise RuntimeError("synthetic hydration failure")

    call = HydrationFailureCall({})
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="synthetic hydration failure"):
        module.run(render=False)

    assert call.cancel_args == [False]


def test_watchdog_cancels_exact_call_if_spy_thread_dies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="e" * 32))

    class DeadSpy(_Spy):
        def run(self) -> int:
            raise RuntimeError("synthetic spy thread crash")

    class PollingCall(_Call):
        def get(self, *, timeout: float):
            assert timeout > 0
            time.sleep(0.01)
            raise TimeoutError

    call = PollingCall({})
    monkeypatch.setattr(module, "ModalExecutionSpy", DeadSpy)
    monkeypatch.setitem(sys.modules, "modal", _modal(call))

    with pytest.raises(RuntimeError, match="spy thread exited unexpectedly"):
        module.run(render=False)

    assert call.cancel_args == [False]
