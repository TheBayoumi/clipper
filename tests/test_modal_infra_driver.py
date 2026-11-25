"""Driver E2E through the real SDKBackend, with inert Modal-shaped test doubles."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import clipper.modal_infra_driver as driver
from clipper.modal_infra_qualification import (
    DATA,
    DIGEST,
    READ_SCRIPT,
    WRITE_SCRIPT,
    Intent,
    Ledger,
)


class FakeSDKState:
    """No provider import, network operation, hidden client or host allocation."""

    def __init__(self) -> None:
        self.client = object()
        self.events: list[str] = []
        self.sandboxes: list[Any] = []
        self.payload = b""
        self.workspace = "approved-workspace"
        self.hold_create = False
        self.hold_terminate = False
        self.entered_create = threading.Event()
        self.release_create = threading.Event()
        self.release_create.set()
        self.entered_terminate = threading.Event()
        self.release_terminate = threading.Event()
        self.release_terminate.set()
        self.sdk = self._sdk()

    def _sdk(self) -> Any:
        state = self

        class Workspace:
            name = ""

            @classmethod
            def from_context(cls, *, client: object) -> Workspace:
                assert client is state.client
                state.events.append("workspace")
                workspace = cls()
                workspace.name = state.workspace
                return workspace

            def hydrate(self) -> None:
                state.events.append("hydrate")

        class Context:
            def __init__(self, kind: str, value: object) -> None:
                self.kind = kind
                self.value = value

            async def __aenter__(self) -> object:
                state.events.append(self.kind + "_enter")
                return self.value

            async def __aexit__(self, *_args: object) -> None:
                state.events.append(self.kind + "_exit")

        class App:
            app_id = "ap-driver-synthetic"

            def run(self, *, client: object, environment_name: str, detach: bool) -> Context:
                assert client is state.client
                assert environment_name == "qualification"
                assert detach is False
                return Context("app", self)

        class Volume:
            object_id = "vo-driver-synthetic"

            @classmethod
            def ephemeral(cls, *, client: object, environment_name: str) -> Context:
                assert client is state.client
                assert environment_name == "qualification"
                return Context("volume", cls())

            def read_file(self, path: str) -> list[bytes]:
                assert path == "infra-check.bin"
                state.events.append("volume_read")
                return [state.payload]

        class Image:
            @staticmethod
            def debian_slim() -> str:
                return "inert-image"

        class Sandbox:
            def __init__(self, index: int, script: str) -> None:
                self.object_id = f"sb-driver-{index}"
                self.script = script
                if script == WRITE_SCRIPT:
                    self.stdout = ["READY:" + DIGEST + "\n"]
                else:
                    self.stdout = ["READ:" + str(len(DATA)) + ":" + DIGEST + "\n"]

            @staticmethod
            def create(*args: object, **kwargs: object) -> Sandbox:
                assert args[:2] == ("python3", "-c")
                assert kwargs["client"] is state.client
                assert kwargs["cpu"] == (0.25, 0.25)
                assert kwargs["memory"] == (256, 256)
                assert kwargs["gpu"] is None
                assert kwargs["block_network"] is True
                assert kwargs["timeout"] == 60
                assert kwargs["secrets"] == []
                assert kwargs["volumes"]["/data"].object_id == "vo-driver-synthetic"
                script = args[2]
                assert script in (WRITE_SCRIPT, READ_SCRIPT)
                state.events.append("create_enter")
                state.entered_create.set()
                if state.hold_create and not state.release_create.wait(3):
                    raise TimeoutError("synthetic create wait exceeded")
                sandbox = Sandbox(len(state.sandboxes) + 1, script)
                state.sandboxes.append(sandbox)
                if script == WRITE_SCRIPT:
                    state.payload = DATA
                state.events.append("create_return")
                return sandbox

            def terminate(self, *, wait: bool) -> None:
                assert wait is True
                state.events.append("terminate_" + self.object_id)
                state.entered_terminate.set()
                if state.hold_terminate and not state.release_terminate.wait(3):
                    raise TimeoutError("synthetic terminate wait exceeded")

            def poll(self) -> int:
                state.events.append("poll_" + self.object_id)
                return 137

        return SimpleNamespace(
            Workspace=Workspace, App=App, Volume=Volume, Image=Image, Sandbox=Sandbox
        )


def _intent(workspace: str = "approved-workspace") -> Intent:
    return Intent(workspace, "qualification", "a" * 40, "modal-cpu-synthetic-persistence")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.skipif(os.name != "posix", reason="Driver explicitly requires POSIX signals")
def test_driver_true_adapter_owner_roundtrip_and_restoration(tmp_path: Path) -> None:
    state = FakeSDKState()
    path = tmp_path / "private" / "journal.jsonl"
    previous = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    result = driver.run_infrastructure_experiment(
        state.sdk, state.client, _intent(), path, decision_seconds=1, reconciliation_seconds=1
    )
    assert result["status"] == "SUCCESS"
    assert result["exact_bytes"] is True
    assert result["sandbox_terminal"] == ["137", "137"]
    assert result["evidence_durability"] == "OWNER_ACKNOWLEDGED"
    assert result["provider_removal"] == "UNKNOWN"
    assert result["unknown_outcomes"] == result["known_unconfirmed_sandboxes"] == []
    assert driver.result_exit_code(result) == 0
    assert len(state.sandboxes) == 2
    assert state.events.index("poll_sb-driver-1") < state.events.index(
        "create_enter", state.events.index("poll_sb-driver-1")
    )
    assert state.events[-2:] == ["volume_exit", "app_exit"]
    assert previous == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    assert path.stat().st_mode & 0o077 == 0
    assert _rows(path)[-1]["result"]["status"] == "UNCONFIRMED"


@pytest.mark.skipif(os.name != "posix", reason="Driver explicitly requires POSIX signals")
def test_driver_workspace_mismatch_has_no_context_or_allocation(tmp_path: Path) -> None:
    state = FakeSDKState()
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    result = driver.run_infrastructure_experiment(
        state.sdk, state.client, _intent("wrong-workspace"), tmp_path / "ledger"
    )
    assert result["status"] == "BLOCKED"
    assert driver.result_exit_code(result) == 2
    assert state.sandboxes == []
    assert "app_enter" not in state.events
    assert "volume_enter" not in state.events
    assert before == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))


@pytest.mark.skipif(os.name != "posix", reason="Driver explicitly requires POSIX signals")
@pytest.mark.parametrize("point", ["create", "terminate"])
def test_driver_real_repeated_signals_wait_for_late_cleanup(tmp_path: Path, point: str) -> None:
    state = FakeSDKState()
    if point == "create":
        state.hold_create = True
        state.release_create.clear()
        entered = state.entered_create
        release = state.release_create
    else:
        state.hold_terminate = True
        state.release_terminate.clear()
        entered = state.entered_terminate
        release = state.release_terminate
    errors: list[str] = []

    def interrupt() -> None:
        if not entered.wait(2):
            errors.append("owner did not reach held operation")
            release.set()
            return
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)
        release.set()

    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    worker = threading.Thread(target=interrupt, daemon=True)
    worker.start()
    try:
        result = driver.run_infrastructure_experiment(
            state.sdk,
            state.client,
            _intent(),
            tmp_path / "private" / "journal.jsonl",
            decision_seconds=0.05,
            reconciliation_seconds=0.05,
        )
    finally:
        release.set()
        worker.join(timeout=3)
    assert not errors
    assert not worker.is_alive()
    assert result["status"] == "BLOCKED"
    assert result["provider_removal"] == "UNKNOWN"
    assert driver.result_exit_code(result) == 2
    assert len(state.sandboxes) == 1
    assert "terminate_sb-driver-1" in state.events
    assert "poll_sb-driver-1" in state.events
    assert state.events[-2:] == ["volume_exit", "app_exit"]
    assert before == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))


@pytest.mark.skipif(os.name != "posix", reason="Driver explicitly requires POSIX signals")
@pytest.mark.parametrize("step", ["run_start", "sandbox_return"])
def test_driver_journal_failure_closes_fd_and_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    state = FakeSDKState()
    opened: list[int] = []

    class FailingLedger(Ledger):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            opened.append(self._fd)

        def record(self, **fields: object) -> None:
            fail = (step == "run_start" and fields.get("operation") == "RUN_START") or (
                step == "sandbox_return"
                and fields.get("operation") == "RETURN"
                and fields.get("step") == "sandbox_1_create"
            )
            if fail:
                raise OSError("permanent synthetic journal failure")
            super().record(**fields)

    monkeypatch.setattr(driver, "Ledger", FailingLedger)
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    if step == "run_start":
        with pytest.raises(OSError, match="journal"):
            driver.run_infrastructure_experiment(
                state.sdk, state.client, _intent(), tmp_path / "journal"
            )
        assert not state.sandboxes
    else:
        result = driver.run_infrastructure_experiment(
            state.sdk, state.client, _intent(), tmp_path / "journal"
        )
        assert result["status"] == "BLOCKED"
        assert result["evidence_errors"]
        assert len(state.sandboxes) == 1
        assert "terminate_sb-driver-1" in state.events
    assert before == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("bad", ["budget", "intent", "client", "path"])
def test_driver_preflight_blocks_invalid_inputs(tmp_path: Path, bad: str) -> None:
    state = FakeSDKState()
    opts: dict[str, Any] = {
        "sdk": state.sdk,
        "client": state.client,
        "intent": _intent(),
        "ledger_path": tmp_path / "ledger",
    }
    if bad == "budget":
        opts["decision_seconds"] = float("inf")
    elif bad == "intent":
        opts["intent"] = _intent("")
    elif bad == "client":
        opts["client"] = None
    else:
        opts["ledger_path"] = "not-a-path"
    with pytest.raises((ValueError, RuntimeError)):
        driver.run_infrastructure_experiment(**opts)
    assert state.events == []
    assert not (tmp_path / "ledger").exists()


def test_driver_nested_loop_fails_before_provider_work(tmp_path: Path) -> None:
    state = FakeSDKState()

    async def invoke() -> None:
        with pytest.raises(RuntimeError, match="event loop"):
            driver.run_infrastructure_experiment(
                state.sdk, state.client, _intent(), tmp_path / "ledger"
            )

    asyncio.run(invoke())
    assert state.events == []


def test_driver_nonmain_thread_fails_before_provider_work(tmp_path: Path) -> None:
    state = FakeSDKState()
    failures: list[str] = []

    def invoke() -> None:
        try:
            driver.run_infrastructure_experiment(
                state.sdk, state.client, _intent(), tmp_path / "ledger"
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert failures and "main-thread" in failures[0]
    assert state.events == []


def test_exit_code_never_promotes_file_only_candidate() -> None:
    assert driver.result_exit_code({"status": "SUCCESS"}) == 2
    assert (
        driver.result_exit_code(
            {"status": "UNCONFIRMED", "evidence_durability": "OWNER_ACKNOWLEDGED"}
        )
        == 2
    )
    assert driver.result_exit_code({"status": "SUCCESS", "evidence_durability": "UNKNOWN"}) == 2


@pytest.mark.skipif(os.name != "posix", reason="Driver requires POSIX signals")
@pytest.mark.parametrize("held", ["create", "terminate"])
def test_driver_observer_cancellation_keeps_late_owner_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: str
) -> None:
    state = FakeSDKState()
    if held == "create":
        state.hold_create = True
        state.release_create.clear()
        entered = state.entered_create
        release = state.release_create
    else:
        state.hold_terminate = True
        state.release_terminate.clear()
        entered = state.entered_terminate
        release = state.release_terminate
    observed = threading.Event()
    cancelled = threading.Event()
    original_observe = driver.QualificationOwner.observe

    async def cancel_observer(owner: Any) -> dict[str, object]:
        assert await asyncio.to_thread(entered.wait, 2)
        observed.set()
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        try:
            return await original_observe(owner)
        finally:
            cancelled.set()

    monkeypatch.setattr(driver.QualificationOwner, "observe", cancel_observer)

    def release_after_cancellation() -> None:
        if cancelled.wait(2):
            time.sleep(0.03)
        release.set()

    worker = threading.Thread(target=release_after_cancellation, daemon=True)
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    worker.start()
    try:
        result = driver.run_infrastructure_experiment(
            state.sdk,
            state.client,
            _intent(),
            tmp_path / "journal",
            decision_seconds=0.1,
            reconciliation_seconds=0.1,
        )
    finally:
        release.set()
        worker.join(timeout=3)
    assert observed.is_set() and cancelled.is_set()
    assert not worker.is_alive()
    assert result["driver_cancelled"] is True
    assert result["driver_interrupted"] is True
    assert result["driver_disposition"] == "INTERRUPTED"
    assert result["status"] == "BLOCKED"
    assert driver.result_exit_code(result) == 2
    assert len(state.sandboxes) == 1
    assert "terminate_sb-driver-1" in state.events
    assert "poll_sb-driver-1" in state.events
    assert state.events[-2:] == ["volume_exit", "app_exit"]
    assert before == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))


@pytest.mark.skipif(os.name != "posix", reason="Driver requires POSIX signals")
def test_driver_repeated_terminal_waiter_cancellation_waits_for_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = FakeSDKState()
    state.hold_terminate = True
    state.release_terminate.clear()
    original_terminal = driver.QualificationOwner.terminal
    cancellations: list[int] = []
    cancel_processed = threading.Event()

    async def repeated_cancel(owner: Any) -> dict[str, object]:
        if len(cancellations) < 2:
            assert await asyncio.to_thread(state.entered_terminate.wait, 2)
            cancellations.append(1)
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            try:
                return await original_terminal(owner)
            finally:
                if len(cancellations) == 2:
                    cancel_processed.set()
        return await original_terminal(owner)

    monkeypatch.setattr(driver.QualificationOwner, "terminal", repeated_cancel)

    def release_after_cancel() -> None:
        if cancel_processed.wait(2):
            time.sleep(0.03)
        state.release_terminate.set()

    worker = threading.Thread(target=release_after_cancel, daemon=True)
    worker.start()
    try:
        result = driver.run_infrastructure_experiment(
            state.sdk,
            state.client,
            _intent(),
            tmp_path / "journal",
            decision_seconds=0.01,
            reconciliation_seconds=0.01,
        )
    finally:
        state.release_terminate.set()
        worker.join(timeout=3)
    assert len(cancellations) == 2
    assert cancel_processed.is_set()
    assert result["status"] == "BLOCKED"
    assert result["driver_cancelled"] is True
    assert driver.result_exit_code(result) == 2
    assert len(state.sandboxes) == 1
    assert "terminate_sb-driver-1" in state.events
    assert "poll_sb-driver-1" in state.events
    assert state.events[-2:] == ["volume_exit", "app_exit"]


@pytest.mark.skipif(os.name != "posix", reason="Driver requires POSIX signals")
@pytest.mark.parametrize("phase", ["terminal_fsync", "ledger_close"])
def test_driver_sigterm_after_owner_success_blocks_driver_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    state = FakeSDKState()
    emitted: list[str] = []
    opened: list[int] = []

    class SignallingLedger(Ledger):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            opened.append(self._fd)

        def record(self, **fields: object) -> None:
            if phase == "terminal_fsync" and fields.get("operation") == "TERMINAL":
                real_fsync = os.fsync

                def signal_during_fsync(fd: int) -> None:
                    if fd == self._fd:
                        emitted.append("terminal_fsync")
                        os.kill(os.getpid(), signal.SIGTERM)
                    real_fsync(fd)

                with monkeypatch.context() as patch:
                    patch.setattr(os, "fsync", signal_during_fsync)
                    super().record(**fields)
                return
            super().record(**fields)

        def close(self) -> None:
            if phase == "ledger_close":
                emitted.append("ledger_close")
                os.kill(os.getpid(), signal.SIGTERM)
            super().close()

    monkeypatch.setattr(driver, "Ledger", SignallingLedger)
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    result = driver.run_infrastructure_experiment(
        state.sdk, state.client, _intent(), tmp_path / "journal"
    )
    assert emitted == [phase]
    assert result["owner_status"] == "SUCCESS"
    assert result["status"] == "BLOCKED"
    assert result["evidence_durability"] == "OWNER_ACKNOWLEDGED"
    assert result["driver_interrupted"] is True
    assert result["driver_disposition"] == "INTERRUPTED"
    assert result["driver_cancelled"] is False
    assert driver.result_exit_code(result) == 2
    assert len(state.sandboxes) == 2
    assert _rows(tmp_path / "journal")[-1]["result"]["status"] == "UNCONFIRMED"
    assert before == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_driver_exit_code_requires_complete_noninterrupted_handoff() -> None:
    owner_success = {"status": "SUCCESS", "evidence_durability": "OWNER_ACKNOWLEDGED"}
    assert driver.result_exit_code(owner_success) == 2
    assert driver.result_exit_code({**owner_success, "driver_disposition": "COMPLETE"}) == 2
    assert (
        driver.result_exit_code(
            {
                **owner_success,
                "driver_disposition": "COMPLETE",
                "driver_interrupted": True,
            }
        )
        == 2
    )
    assert (
        driver.result_exit_code(
            {
                **owner_success,
                "driver_disposition": "COMPLETE",
                "driver_interrupted": False,
            }
        )
        == 0
    )


@pytest.mark.skipif(os.name != "posix", reason="Driver requires POSIX signals")
def test_driver_pending_cancel_when_owner_already_done_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = FakeSDKState()
    original_terminal = driver.QualificationOwner.terminal
    pending_cancel = threading.Event()

    async def completed_then_cancel(owner: Any) -> dict[str, object]:
        result = await original_terminal(owner)
        if not pending_cancel.is_set():
            assert owner.task is not None and owner.task.done()
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            pending_cancel.set()
        return result

    monkeypatch.setattr(driver.QualificationOwner, "terminal", completed_then_cancel)
    original_handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    result = driver.run_infrastructure_experiment(
        state.sdk, state.client, _intent(), tmp_path / "journal"
    )
    assert pending_cancel.is_set()
    assert result["owner_status"] == "SUCCESS"
    assert result["status"] == "BLOCKED"
    assert result["driver_cancelled"] is True
    assert result["driver_interrupted"] is True
    assert driver.result_exit_code(result) == 2
    assert len(state.sandboxes) == 2
    assert state.events[-2:] == ["volume_exit", "app_exit"]
    assert original_handlers == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))


@pytest.mark.skipif(os.name != "posix", reason="Signal restoration requires POSIX")
@pytest.mark.parametrize(
    ("moment", "expected_status"),
    [("before_sigterm_restore", "BLOCKED"), ("after_sigterm_restore", "SUCCESS")],
)
def test_driver_sigterm_handoff_is_per_signal_not_joint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    moment: str,
    expected_status: str,
) -> None:
    state = FakeSDKState()
    set_handler = signal.signal
    previous_term = set_handler(signal.SIGTERM, signal.SIG_IGN)
    triggered: list[str] = []

    def intercept(signum: signal.Signals, handler: Any) -> Any:
        if signum == signal.SIGTERM and handler == signal.SIG_IGN:
            if moment == "before_sigterm_restore":
                assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
                os.kill(os.getpid(), signal.SIGTERM)
                triggered.append("before")
            old = set_handler(signum, handler)
            if moment == "after_sigterm_restore":
                # SIGTERM is caller-owned (ignored), although SIGINT is
                # still installed for the driver until its next restore.
                assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
                os.kill(os.getpid(), signal.SIGTERM)
                triggered.append("after")
            return old
        return set_handler(signum, handler)

    try:
        monkeypatch.setattr(driver.signal, "signal", intercept)
        result = driver.run_infrastructure_experiment(
            state.sdk, state.client, _intent(), tmp_path / "journal"
        )
    finally:
        monkeypatch.undo()
        set_handler(signal.SIGTERM, previous_term)
    assert triggered == ["before" if moment == "before_sigterm_restore" else "after"]
    assert result["owner_status"] == "SUCCESS"
    assert result["status"] == expected_status
    assert result["driver_interrupted"] is (expected_status == "BLOCKED")
    assert driver.result_exit_code(result) == (2 if expected_status == "BLOCKED" else 0)
    assert len(state.sandboxes) == 2



@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "SIGUSR1"),
    reason="Real SIGUSR1 subprocess contract requires POSIX",
)
def test_driver_rejects_actual_callable_sigusr1_before_any_work(tmp_path: Path) -> None:
    """Real installed throwing handler must be rejected, not executed or replaced."""
    script = """
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
import clipper.modal_infra_driver as driver
from test_modal_infra_driver import FakeSDKState, _intent

state = FakeSDKState()
attempts = []
class ForbiddenLedger:
    def __init__(self, *_args, **_kwargs):
        attempts.append("ledger")
        raise AssertionError("ledger constructed")
class ForbiddenRunner:
    def __init__(self, *_args, **_kwargs):
        attempts.append("runner")
        raise AssertionError("runner started")
driver.Ledger = ForbiddenLedger
driver.asyncio.Runner = ForbiddenRunner
real_signal = signal.signal

def no_handler_install(signum, handler):
    if signum in (signal.SIGINT, signal.SIGTERM):
        attempts.append("owned_handler_mutation")
        raise AssertionError("owned handler changed")
    return real_signal(signum, handler)

def throwing(_signum, _frame):
    attempts.append("throwing_host_handler_executed")
    raise KeyboardInterrupt("unacceptable SIGUSR1 execution")

previous = real_signal(signal.SIGUSR1, throwing)
before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
driver.signal.signal = no_handler_install
try:
    try:
        driver.run_infrastructure_experiment(
            state.sdk, state.client, _intent(), Path(sys.argv[1]) / "journal"
        )
    except RuntimeError as exc:
        assert "Unsupported host signal contract" in str(exc)
    else:
        raise AssertionError("callable host signal unexpectedly accepted")
    assert signal.getsignal(signal.SIGUSR1) is throwing
    assert before == (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    assert attempts == [], attempts
    assert state.events == [], state.events
    assert state.sandboxes == []
    assert not (Path(sys.argv[1]) / "journal").exists()
finally:
    driver.signal.signal = real_signal
    real_signal(signal.SIGUSR1, previous)
print(json.dumps({"rejected": True}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(Path(__file__).resolve().parent)],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert json.loads(completed.stdout.strip().splitlines()[-1])["rejected"] is True


@pytest.mark.parametrize("which", ["owned", "other"])
@pytest.mark.parametrize("value", [None, object()])
def test_driver_rejects_unknown_signal_dispositions_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str, value: object
) -> None:
    state = FakeSDKState()
    inspected = signal.SIGTERM if which == "owned" else signal.SIGUSR1
    actual_getsignal = signal.getsignal
    modified: list[object] = []

    def fake_getsignal(signum: signal.Signals) -> object:
        if signum == inspected:
            return value
        return actual_getsignal(signum)

    def no_mutation(*args: object) -> None:
        modified.append(args)
        raise AssertionError("changed handler before preflight")

    with monkeypatch.context() as patch:
        patch.setattr(driver.signal, "getsignal", fake_getsignal)
        patch.setattr(driver.signal, "signal", no_mutation)
        with pytest.raises(RuntimeError, match="Unsupported host signal contract"):
            driver.run_infrastructure_experiment(
                state.sdk, state.client, _intent(), tmp_path / "journal"
            )
    assert modified == []
    assert state.events == []
    assert not (tmp_path / "journal").exists()


@pytest.mark.parametrize("failure", ["valid_signals", "getsignal", "mask", "blocked"])
def test_driver_signal_preflight_fails_closed_without_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    state = FakeSDKState()
    touched: list[str] = []
    original_signal = signal.signal
    original_valid = signal.valid_signals
    original_get = signal.getsignal
    original_mask = signal.pthread_sigmask

    def fail(*_args: object) -> Any:
        raise OSError("synthetic inspection failure")

    def fake_get(signum: signal.Signals) -> object:
        if signum == signal.SIGTERM:
            return fail()
        return original_get(signum)

    def mask(*args: object) -> Any:
        if failure == "blocked":
            assert args == (signal.SIG_BLOCK, set())
            return {signal.SIGINT, signal.SIGTERM}
        return original_mask(*args)

    def spy_signal(signum: signal.Signals, handler: object) -> Any:
        touched.append("signal")
        return original_signal(signum, handler)

    with monkeypatch.context() as patch:
        patch.setattr(driver.signal, "signal", spy_signal)
        if failure == "valid_signals":
            patch.setattr(driver.signal, "valid_signals", fail)
        elif failure == "getsignal":
            patch.setattr(driver.signal, "getsignal", fake_get)
        else:
            patch.setattr(driver.signal, "pthread_sigmask", fail if failure == "mask" else mask)
        with pytest.raises(RuntimeError, match="Unsupported host signal contract"):
            driver.run_infrastructure_experiment(
                state.sdk, state.client, _intent(), tmp_path / "journal"
            )
    assert touched == []
    assert state.events == []
    assert not (tmp_path / "journal").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_driver_owned_callable_handlers_are_allowed_and_restored(tmp_path: Path) -> None:
    state = FakeSDKState()
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)

    def owned_handler(_signum: int, _frame: object) -> None:
        pass

    try:
        signal.signal(signal.SIGINT, owned_handler)
        signal.signal(signal.SIGTERM, owned_handler)
        result = driver.run_infrastructure_experiment(
            state.sdk, state.client, _intent(), tmp_path / "journal"
        )
        assert result["status"] == "SUCCESS"
        assert driver.result_exit_code(result) == 0
        assert signal.getsignal(signal.SIGINT) is owned_handler
        assert signal.getsignal(signal.SIGTERM) is owned_handler
        assert len(state.sandboxes) == 2
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_driver_partial_install_failure_restores_first_owned_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = FakeSDKState()
    original_signal = signal.signal
    before_int = signal.getsignal(signal.SIGINT)
    before_term = signal.getsignal(signal.SIGTERM)
    actions: list[signal.Signals] = []

    def fail_second(signum: signal.Signals, handler: object) -> Any:
        actions.append(signum)
        if signum == signal.SIGTERM and callable(handler):
            raise OSError("synthetic second handler install failure")
        return original_signal(signum, handler)

    with monkeypatch.context() as patch:
        patch.setattr(driver.signal, "signal", fail_second)
        with pytest.raises(OSError, match="synthetic second"):
            driver.run_infrastructure_experiment(
                state.sdk, state.client, _intent(), tmp_path / "journal"
            )
    assert actions == [signal.SIGINT, signal.SIGTERM, signal.SIGINT]
    assert before_int == signal.getsignal(signal.SIGINT)
    assert before_term == signal.getsignal(signal.SIGTERM)
    assert state.events == []
    assert not (tmp_path / "journal").exists()
