"""Explicit synchronous owner for the isolated Modal CPU infrastructure experiment.

No implicit Modal import, client initialization, CLI, or live invocation.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import threading
from pathlib import Path
from types import FrameType
from typing import Any

from clipper.modal_infra_qualification import Intent, Ledger, QualificationOwner, SDKBackend


def _preflight_host_signals() -> dict[signal.Signals, Any]:
    """Fail closed on unsupported dispositions or blocked owned signals."""
    owned = (signal.SIGINT, signal.SIGTERM)
    previous: dict[signal.Signals, Any] = {}
    try:
        available = signal.valid_signals()
        if any(signum not in available for signum in owned):
            raise ValueError("Missing owned signal")
        for signum in sorted(available, key=int):
            if signum in (signal.SIGKILL, signal.SIGSTOP):
                continue
            disposition = signal.getsignal(signum)
            if signum in owned:
                if (
                    disposition is not signal.SIG_DFL
                    and disposition is not signal.SIG_IGN
                    and not callable(disposition)
                ):
                    raise ValueError(f"Unrestorable owned signal: {signum}")
                previous[signum] = disposition
            elif disposition is not signal.SIG_DFL and disposition is not signal.SIG_IGN:
                # Reject callable, None, and unknown values; never execute handlers.
                raise ValueError(f"Unsupported host signal: {signum}")
        # Empty SIG_BLOCK input reads the main-thread mask without unblocking.
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        if any(signum in blocked for signum in owned):
            raise ValueError("Owned signal blocked")
    except Exception as exc:
        raise RuntimeError("Unsupported host signal contract") from exc
    return previous


def result_exit_code(result: dict[str, object]) -> int:
    """Map only live owner-acknowledged success to a successful process status."""
    if (
        result.get("driver_disposition") == "COMPLETE"
        and result.get("driver_interrupted") is False
        and result.get("status") == "SUCCESS"
        and result.get("evidence_durability") == "OWNER_ACKNOWLEDGED"
    ):
        return 0
    return 2


def run_infrastructure_experiment(
    sdk: Any,
    client: Any,
    intent: Intent,
    ledger_path: Path,
    *,
    decision_seconds: float = 8,
    reconciliation_seconds: float = 16,
) -> dict[str, object]:
    """Own one event loop and controlled POSIX signal lifecycle.

    Host dispositions and masks must remain stable throughout the run. Only
    explicit injected SDK/client handles are accepted; no spend authority.
    """
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Driver requires POSIX main-thread signal ownership")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Driver cannot run inside an existing event loop")
    if not isinstance(intent, Intent) or not isinstance(ledger_path, Path):
        raise ValueError("Explicit Intent and pathlib.Path ledger required")
    intent.validate()
    if (
        not math.isfinite(decision_seconds)
        or not math.isfinite(reconciliation_seconds)
        or decision_seconds <= 0
        or reconciliation_seconds <= 0
    ):
        raise ValueError("Finite positive observation budgets required")
    previous = _preflight_host_signals()
    backend = SDKBackend(sdk, client, intent.environment)
    installed: list[signal.Signals] = []
    owner: QualificationOwner | None = None
    stop_requested = False
    observation_cancelled = False
    ledger: Ledger | None = None

    def on_signal(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_requested
        stop_requested = True
        if owner is not None:
            owner.request_stop()

    async def exercise() -> dict[str, object]:
        nonlocal owner, observation_cancelled
        if ledger is None:
            raise RuntimeError("Driver ledger unavailable")
        owner = QualificationOwner(
            backend,
            intent,
            ledger,
            decision_seconds=decision_seconds,
            reconciliation_seconds=reconciliation_seconds,
        )
        owner.start()
        if stop_requested:
            owner.request_stop()
        observation_error: BaseException | None = None
        try:
            await owner.observe()
        except asyncio.CancelledError:
            observation_cancelled = True
            owner.request_stop()
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        except BaseException as exc:
            # An unexpected observation failure is never converted to success.
            observation_error = exc
            owner.request_stop()

        # Only the driver waiter is uncancelled; the owner task remains alive.
        # Repeated waiter cancellation cannot bypass late-handle cleanup.
        while True:
            try:
                result = await owner.terminal()
                # A completed owner await may return without delivering a
                # cancellation already pending on this driver task.
                await asyncio.sleep(0)
                break
            except asyncio.CancelledError:
                if owner.task is not None and owner.task.cancelled():
                    raise RuntimeError("Owner task cancelled unexpectedly") from None
                observation_cancelled = True
                owner.request_stop()
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        if observation_error is not None:
            raise observation_error
        return result

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, on_signal)
            installed.append(signum)
        ledger = Ledger(ledger_path)
        with asyncio.Runner() as runner:
            result = runner.run(exercise())
    finally:
        # Ledger remains open through terminal reconciliation. Noncooperative
        # SDK calls may block indefinitely; this is not a hard provider deadline.
        try:
            if ledger is not None:
                ledger.close()
        finally:
            for signum in reversed(installed):
                signal.signal(signum, previous[signum])

    # Ledger closes before sequential SIGTERM and SIGINT restoration.
    # Per-signal handoff is not atomic; host signal contract must remain stable.
    interrupted = stop_requested or observation_cancelled
    outcome = dict(result)
    outcome["owner_status"] = result["status"]
    outcome["driver_interrupted"] = interrupted
    outcome["driver_cancelled"] = observation_cancelled
    outcome["driver_disposition"] = "INTERRUPTED" if interrupted else "COMPLETE"
    if interrupted:
        outcome["status"] = "BLOCKED"
    return outcome
