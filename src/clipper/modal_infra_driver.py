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


def result_exit_code(result: dict[str, object]) -> int:
    """Map only live owner-acknowledged success to a successful process status."""
    if (
        result.get("status") == "SUCCESS"
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
    """Own one event loop, private ledger, and POSIX signals until owner terminal.

    Only an external, independently authorized host wrapper may supply an
    existing explicit SDK/client. This function grants no account/spend authority.
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
    backend = SDKBackend(sdk, client, intent.environment)

    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    installed: list[signal.Signals] = []
    owner: QualificationOwner | None = None
    stop_requested = False
    ledger: Ledger | None = None

    def on_signal(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_requested
        stop_requested = True
        if owner is not None:
            owner.request_stop()

    async def exercise() -> dict[str, object]:
        nonlocal owner
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
        try:
            await owner.observe()
        except Exception:
            # An observation error must not discard a started owner task.
            owner.request_stop()
        return await owner.terminal()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, on_signal)
            installed.append(signum)
        ledger = Ledger(ledger_path)
        with asyncio.Runner() as runner:
            return runner.run(exercise())
    finally:
        # Do not close this ledger before terminal() has settled.
        # A noncooperative provider call can prevent terminal() forever.
        try:
            if ledger is not None:
                ledger.close()
        finally:
            for signum in reversed(installed):
                signal.signal(signum, previous[signum])
