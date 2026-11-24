"""Modal CPU-only 64 KiB lifetime qualification; inert until explicitly invoked.

No Modal import, client creation, CLI entry point or implicit provider calls.
A caller MUST keep the owner event loop alive until terminal() completes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import uuid
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

DATA = bytes(range(256)) * 256
DIGEST = hashlib.sha256(DATA).hexdigest()
SIZE = len(DATA)
PATH = "/data/infra-check.bin"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

# No private payload is interpolated into either command.
WRITE_SCRIPT = (
    "import hashlib,os,pathlib,time\n"
    "b=bytes(range(256))*256\n"
    "p=pathlib.Path('/data/infra-check.bin')\n"
    "with p.open('wb') as f:\n"
    " f.write(b);f.flush();os.fsync(f.fileno())\n"
    "print('READY:'+hashlib.sha256(b).hexdigest(),flush=True)\n"
    "while True: time.sleep(10)\n"
)
READ_SCRIPT = (
    "import hashlib,pathlib,time\n"
    "p=pathlib.Path('/data/infra-check.bin')\n"
    "b=p.read_bytes()\n"
    "print('READ:'+str(len(b))+':'+hashlib.sha256(b).hexdigest(),flush=True)\n"
    "while True: time.sleep(10)\n"
)


@dataclass(frozen=True)
class Intent:
    expected_workspace: str
    environment: str
    source_sha: str
    explicit_run_intent: str

    def validate(self) -> None:
        if (
            not self.expected_workspace
            or not self.environment
            or self.explicit_run_intent != "modal-cpu-synthetic-persistence"
            or not SHA_PATTERN.fullmatch(self.source_sha)
        ):
            raise ValueError("Explicit workspace/environment/source SHA/run intent required")


class Ledger:
    """Private, append-only, fsynced JSONL; no provider tokens or book content."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink():
            raise ValueError("Symlink journal forbidden")
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        if os.fstat(self._fd).st_mode & 0o077:
            os.close(self._fd)
            raise ValueError("Journal must be owner-private")

    def record(self, **fields: object) -> None:
        payload = (json.dumps(fields, sort_keys=True, separators=(",", ":")) + "\n").encode()
        count = os.write(self._fd, payload)
        if count != len(payload):
            raise OSError("Incomplete journal record")
        os.fsync(self._fd)

    def close(self) -> None:
        os.close(self._fd)


class Backend(Protocol):
    async def verify(self, expected_workspace: str) -> str: ...
    def app_context(self) -> Any: ...
    def volume_context(self) -> Any: ...
    async def create(self, app: Any, volume: Any, script: str) -> Any: ...
    async def stdout_chunks(self, sandbox: Any) -> Any: ...
    async def volume_chunks(self, volume: Any) -> Any: ...
    async def terminate(self, sandbox: Any) -> None: ...
    async def poll(self, sandbox: Any) -> Any: ...


@dataclass(frozen=True)
class SDKBackend:
    """1.5.5 adapter. Caller supplies one explicit client and SDK module.

    No SDK import/client initialization occurs here. These methods are NEVER
    invoked by importing this module or by the offline fake tests.
    """

    sdk: Any
    client: Any
    environment: str

    def __post_init__(self) -> None:
        if self.sdk is None or self.client is None or not self.environment:
            raise ValueError("Explicit SDK/client/environment required")

    async def verify(self, expected_workspace: str) -> str:
        def check() -> str:
            workspace = self.sdk.Workspace.from_context(client=self.client)
            workspace.hydrate()
            # Modal 1.5.5 Workspace.name is an Optional[str] property.
            name = workspace.name
            if not isinstance(name, str) or not name:
                raise ValueError("Modal workspace identity unavailable")
            return name

        actual = await asyncio.to_thread(check)
        if actual != expected_workspace:
            raise ValueError("Modal workspace identity mismatch")
        return actual

    def app_context(self) -> Any:
        # The App object is local until context entry.
        app = self.sdk.App()
        return app.run(client=self.client, environment_name=self.environment, detach=False)

    def volume_context(self) -> Any:
        return self.sdk.Volume.ephemeral(client=self.client, environment_name=self.environment)

    async def create(self, app: Any, volume: Any, script: str) -> Any:
        return await asyncio.to_thread(
            self.sdk.Sandbox.create,
            "python3",
            "-c",
            script,
            app=app,
            image=self.sdk.Image.debian_slim(),
            volumes={"/data": volume},
            cpu=(0.25, 0.25),
            memory=(256, 256),
            gpu=None,
            secrets=[],
            encrypted_ports=[],
            unencrypted_ports=[],
            h2_ports=[],
            block_network=True,
            timeout=60,
            client=self.client,
        )

    async def stdout_chunks(self, sandbox: Any) -> Any:
        return await asyncio.to_thread(lambda: iter(sandbox.stdout))

    async def volume_chunks(self, volume: Any) -> Any:
        return await asyncio.to_thread(lambda: iter(volume.read_file("infra-check.bin")))

    async def terminate(self, sandbox: Any) -> None:
        await asyncio.to_thread(sandbox.terminate, wait=True)

    async def poll(self, sandbox: Any) -> Any:
        return await asyncio.to_thread(sandbox.poll)


async def _close_iterator(iterator: Any) -> None:
    closer = getattr(iterator, "aclose", None)
    if closer is not None:
        await closer()
    else:
        closer = getattr(iterator, "close", None)
        if closer is not None:
            await asyncio.to_thread(closer)


async def _next_chunk(iterator: Any) -> bytes | str | None:
    if hasattr(iterator, "__anext__"):
        try:
            async_piece: bytes | str | None = await iterator.__anext__()
            return async_piece
        except StopAsyncIteration:
            return None

    # Do not transfer StopIteration through an asyncio Future.
    piece: bytes | str | None = await asyncio.to_thread(lambda: next(iterator, None))
    return piece


async def read_first_line(iterator: Any, maximum: int = 128) -> bytes:
    """One logical line; do not examine or accumulate its trailing suffix."""
    prefix = bytearray()
    while True:
        chunk = await _next_chunk(iterator)
        if chunk is None:
            raise ValueError("Readiness EOF before newline")
        data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        if not isinstance(data, bytes):
            raise ValueError("Readiness must be text or bytes")
        end = data.find(b"\n")
        prefix.extend(data if end < 0 else data[:end])
        if len(prefix) > maximum:
            raise ValueError("Readiness first line overflow")
        if end >= 0:
            return bytes(prefix)


async def read_exact(iterator: Any, cap: int = SIZE) -> bytes:
    out = bytearray()
    while True:
        piece = await _next_chunk(iterator)
        if piece is None:
            return bytes(out)
        if not isinstance(piece, bytes) or len(out) + len(piece) > cap:
            raise ValueError("Volume data invalid or oversized")
        out.extend(piece)


class KnownSandbox(TypedDict):
    """Known sandbox ownership and separately observed cleanup state."""

    id: str
    step: str
    terminate_attempted: bool
    terminate_succeeded: bool
    poll_attempted: bool
    terminal_observed: bool
    cleanup_errors: list[str]


class QualificationOwner:
    """One in-loop owner; finite observation is NOT a hard provider deadline.

    Pending operations and settled uncertain acquisitions are different state.
    The caller must keep the loop alive and await terminal() before shutdown.
    """

    def __init__(
        self,
        backend: Backend,
        intent: Intent,
        ledger: Ledger,
        *,
        decision_seconds: float = 8,
        reconciliation_seconds: float = 16,
    ) -> None:
        intent.validate()
        # SDK environment is immutable and must agree with the sole journal intent
        # before workspace verification or any provider context is entered.
        if isinstance(backend, SDKBackend) and backend.environment != intent.environment:
            raise ValueError("SDK environment differs from immutable run intent")
        if (
            not math.isfinite(decision_seconds)
            or not math.isfinite(reconciliation_seconds)
            or decision_seconds <= 0
            or reconciliation_seconds <= 0
        ):
            raise ValueError("Finite positive observation budgets required")
        self.backend = backend
        self.intent = intent
        self.ledger = ledger
        self.decision_seconds = decision_seconds
        self.reconciliation_seconds = reconciliation_seconds
        self.run_id = uuid.uuid4().hex
        self.stopped = asyncio.Event()
        self.task: asyncio.Task[dict[str, object]] | None = None
        self.unresolved: set[str] = set()
        self.active_operations: set[str] = set()
        self.provider_entry_started: set[str] = set()
        self.unknown_outcomes: list[dict[str, str]] = []
        self.evidence_errors: list[str] = []
        self.cleanup_errors: list[str] = []
        self.live_sandboxes: list[Any] = []
        # Returned identities remain actionable until terminal observation.
        self.known_unconfirmed_sandboxes: dict[str, KnownSandbox] = {}
        self.sandbox_terminal: list[str] = []
        self._result: dict[str, object] = {
            "run_id": self.run_id,
            "status": "PENDING",
            "sandbox_terminal": self.sandbox_terminal,
            "exact_bytes": False,
            "app_context_closed": False,
            "volume_context_closed": False,
            "provider_removal": "UNKNOWN",
            "primary_error": None,
            "cleanup_errors": self.cleanup_errors,
            "evidence_errors": self.evidence_errors,
            "unknown_outcomes": self.unknown_outcomes,
            "active_operations": [],
            "known_unconfirmed_sandboxes": [],
            "evidence_durability": "UNCONFIRMED",
        }

    def _record(self, operation: str, **details: object) -> None:
        """Journal errors latch STOP; callers explicitly decide if fatal."""
        try:
            self.ledger.record(
                run_id=self.run_id,
                source_sha=self.intent.source_sha,
                workspace=self.intent.expected_workspace,
                environment=self.intent.environment,
                operation=operation,
                **details,
            )
        except OSError as exc:
            self.stopped.set()
            self.evidence_errors.append(f"{operation}:{type(exc).__name__}")
            # Never write an error about a journal error to the same sink.
            raise

    def _safe_record(self, operation: str, **details: object) -> None:
        with suppress(OSError):
            self._record(operation, **details)

    def _first_error(self, exc: BaseException) -> None:
        if self._result["primary_error"] is None:
            self._result["primary_error"] = type(exc).__name__
        self.stopped.set()

    def _unknown(self, kind: str, step: str, identifier: str = "") -> None:
        item = {
            "kind": kind,
            "step": step,
            "id": identifier,
            "outcome": "UNRESOLVED",
        }
        self.unknown_outcomes.append(item)
        self.unresolved.add(step)
        self.stopped.set()
        self._safe_record("UNKNOWN_ID_OUTCOME", **item)

    async def _await_operation(self, step: str, action: Any, *, acquisition: bool = False) -> Any:
        """Track every owned await, including streams and teardown."""
        self.active_operations.add(step)
        try:
            if acquisition:
                self._record("OPERATION_BEGIN", step=step)
                if self.stopped.is_set():
                    raise RuntimeError("STOP_LATCHED")
            else:
                self._safe_record("OPERATION_BEGIN", step=step)
            if acquisition:
                # Mark the exact boundary after the durable pre-entry barrier.
                self.provider_entry_started.add(step)
            return await action()
        finally:
            self.active_operations.discard(step)
            self._safe_record("OPERATION_SETTLED", step=step)

    async def _operation(self, name: str, action: Any, *, acquisition: bool = False) -> Any:
        if acquisition and self.stopped.is_set():
            raise RuntimeError("STOP_LATCHED")
        # Cleanup still runs on journal failure, but resource entry must not.
        if acquisition:
            self._record("INTENT", step=name)
        else:
            self._safe_record("INTENT", step=name)
        result = await self._await_operation(name, action)
        self._safe_record("RETURN", step=name, object_id=str(getattr(result, "object_id", "")))
        return result

    def start(self) -> None:
        if self.task is not None:
            raise ValueError("Owner cannot start twice")
        self._record("RUN_START", intent=self.intent.explicit_run_intent, sha256=DIGEST)
        self.task = asyncio.create_task(self._run())

    def request_stop(self) -> None:
        self.stopped.set()

    def _snapshot(self) -> dict[str, object]:
        # Isolate nested owner state before giving snapshots to the journal/task.
        return deepcopy({
            **self._result,
            "active_operations": sorted(self.active_operations),
            "unknown_outcomes": [dict(entry) for entry in self.unknown_outcomes],
            "unresolved_ids": sorted(self.unresolved | set(self.known_unconfirmed_sandboxes)),
            "known_unconfirmed_sandboxes": [
                dict(entry) for entry in self.known_unconfirmed_sandboxes.values()
            ],
            "evidence_errors": list(self.evidence_errors),
            "cleanup_errors": list(self.cleanup_errors),
        })

    async def observe(self) -> dict[str, object]:
        if self.task is None:
            raise ValueError("Owner not started")
        try:
            done, _ = await asyncio.wait({self.task}, timeout=self.decision_seconds)
        except asyncio.CancelledError:
            self.request_stop()
            self._safe_record("CALLER_CANCELLED_OWNER_RETAINED")
            raise
        if done:
            return deepcopy(self.task.result())
        self.request_stop()
        self._safe_record("DECISION_DEADLINE", disposition="OWNER_RETAINED_IN_LOOP")
        try:
            done, _ = await asyncio.wait({self.task}, timeout=self.reconciliation_seconds)
        except asyncio.CancelledError:
            self.request_stop()
            self._safe_record("CALLER_CANCELLED_OWNER_RETAINED")
            raise
        if done:
            return deepcopy(self.task.result())
        self._safe_record(
            "UNRESOLVED",
            active_operations=sorted(self.active_operations),
            unknown_allocations=[dict(item) for item in self.unknown_outcomes],
            disposition="OWNER_MONITORING_CONTINUES_IN_ACTIVE_LOOP",
        )
        return {
            **self._snapshot(),
            "status": "UNRESOLVED",
            "owner_task_alive": True,
        }

    async def terminal(self) -> dict[str, object]:
        if self.task is None:
            raise ValueError("Owner not started")
        try:
            return deepcopy(await asyncio.shield(self.task))
        except asyncio.CancelledError:
            self.request_stop()
            self._safe_record("CALLER_CANCELLED_OWNER_RETAINED", waiter="terminal")
            raise

    async def _drain(self, name: str, iterator: Any, *, line: bool) -> bytes:
        primary: BaseException | None = None
        try:
            consumed: bytes = await self._await_operation(
                name + "_consume",
                lambda: read_first_line(iterator) if line else read_exact(iterator),
            )
            return consumed
        except BaseException as exc:
            primary = exc
            self._first_error(exc)
            raise
        finally:
            try:
                await self._operation(name + "_iterator_close", lambda: _close_iterator(iterator))
            except BaseException as exc:
                self.cleanup_errors.append(f"{name}_iterator_close:{type(exc).__name__}")
                self._first_error(exc)
                self._safe_record("ITERATOR_CLOSE_FAILED", step=name, error=type(exc).__name__)
                if primary is None:
                    raise

    async def _sandbox(
        self, app: Any, volume: Any, script: str, expected: bytes, index: int
    ) -> None:
        step = f"sandbox_{index}_create"
        if self.stopped.is_set():
            raise RuntimeError("STOP_LATCHED")
        self._record("INTENT", step=step)
        try:
            sb = await self._await_operation(
                step, lambda: self.backend.create(app, volume, script), acquisition=True
            )
        except BaseException:
            # A rejected pre-entry evidence write cannot allocate remotely.
            if step in self.provider_entry_started:
                self._unknown("sandbox", step)
            raise
        # Handle registration precedes every potentially failing evidence write.
        self.live_sandboxes.append(sb)
        try:
            sandbox_id = getattr(sb, "object_id", None)
            if not isinstance(sandbox_id, str) or not sandbox_id:
                self._unknown("sandbox", step)
                raise ValueError("Sandbox missing object_id")
            # Register before journal failure. Context exit is not termination.
            resource: KnownSandbox = {
                "id": sandbox_id,
                "step": step,
                "terminate_attempted": False,
                "terminate_succeeded": False,
                "poll_attempted": False,
                "terminal_observed": False,
                "cleanup_errors": [],
            }
            self.known_unconfirmed_sandboxes[sandbox_id] = resource
            try:
                self._record("RETURN", step=step, object_id=sandbox_id)
            except OSError:
                self._unknown("sandbox", step, sandbox_id)
                raise
            if self.stopped.is_set():
                raise RuntimeError("STOP_LATCHED_AFTER_CREATE")
            iterator = await self._await_operation(
                f"sandbox_{index}_stdout_acquire",
                lambda: self.backend.stdout_chunks(sb),
            )
            first = await self._drain(f"sandbox_{index}_stdout", iterator, line=True)
            if first != expected:
                raise ValueError("Invalid first readiness line")
            self._record("READINESS", step=index)
        except BaseException as exc:
            self._first_error(exc)
        finally:
            # Attempt poll even when terminate fails: two separate observations.
            cleanup_id = getattr(sb, "object_id", None)
            cleanup_resource = (
                self.known_unconfirmed_sandboxes.get(cleanup_id)
                if isinstance(cleanup_id, str)
                else None
            )
            try:
                if cleanup_resource is not None:
                    cleanup_resource["terminate_attempted"] = True
                await self._operation(
                    f"sandbox_{index}_terminate", lambda: self.backend.terminate(sb)
                )
                if cleanup_resource is not None:
                    cleanup_resource["terminate_succeeded"] = True
            except BaseException as exc:
                if cleanup_resource is not None:
                    cleanup_resource["cleanup_errors"].append(f"terminate:{type(exc).__name__}")
                self.cleanup_errors.append(f"sandbox_{index}_terminate:{type(exc).__name__}")
                self._first_error(exc)
                self._safe_record("SANDBOX_CLEANUP_ERROR", step=index, error=type(exc).__name__)
            try:
                if cleanup_resource is not None:
                    cleanup_resource["poll_attempted"] = True
                terminal = await self._operation(
                    f"sandbox_{index}_poll", lambda: self.backend.poll(sb)
                )
                if terminal is None:
                    raise ValueError("No terminal observation")
                self._record("SANDBOX_TERMINAL", step=index, terminal=str(terminal))
                self.sandbox_terminal.append(str(terminal))
                if cleanup_resource is not None:
                    cleanup_resource["terminal_observed"] = True
                    self.known_unconfirmed_sandboxes.pop(cleanup_resource["id"], None)
            except BaseException as exc:
                if cleanup_resource is not None:
                    cleanup_resource["cleanup_errors"].append(f"poll:{type(exc).__name__}")
                self.cleanup_errors.append(f"sandbox_{index}_poll:{type(exc).__name__}")
                self._first_error(exc)
                self._safe_record("SANDBOX_CLEANUP_ERROR", step=index, error=type(exc).__name__)
            finally:
                if cleanup_resource is not None and not cleanup_resource["terminal_observed"]:
                    self._safe_record("KNOWN_SANDBOX_UNCONFIRMED", **cleanup_resource)
        if self.stopped.is_set():
            raise RuntimeError("STOP_LATCHED")

    async def _enter_context(self, kind: str, context: Any) -> Any:
        step = f"{kind}_context_enter"
        if self.stopped.is_set():
            raise RuntimeError("STOP_LATCHED")
        self._record("INTENT", step=step)
        try:
            handle = await self._await_operation(step, context.__aenter__, acquisition=True)
        except BaseException:
            if step in self.provider_entry_started:
                self._unknown(kind, step)
            raise
        # Caller stores the returned handle before validating/journaling ID.
        return handle

    async def _exit_context(self, kind: str, context: Any) -> None:
        step = f"{kind}_context_exit"
        try:
            await self._operation(step, lambda: context.__aexit__(None, None, None))
            self._result[step.replace("_exit", "_closed")] = True
        except BaseException as exc:
            self.cleanup_errors.append(f"{step}:{type(exc).__name__}")
            self._first_error(exc)
            self._safe_record("CONTEXT_EXIT_FAILED", step=step, error=type(exc).__name__)

    async def _run(self) -> dict[str, object]:
        app_context: Any = None
        volume_context: Any = None
        app_entered = False
        volume_entered = False
        try:
            verified = await self._await_operation(
                "workspace_verify",
                lambda: self.backend.verify(self.intent.expected_workspace),
            )
            if verified != self.intent.expected_workspace:
                raise ValueError("Client-bound workspace mismatch")
            self._record("WORKSPACE_VERIFIED")
            if self.stopped.is_set():
                raise RuntimeError("STOP_LATCHED")
            app_context = self.backend.app_context()
            app = await self._enter_context("app", app_context)
            app_entered = True
            app_id = getattr(app, "app_id", None)
            if not isinstance(app_id, str) or not app_id:
                self._unknown("app", "app_context_enter")
                raise ValueError("Modal App returned without an app_id")
            try:
                self._record("RETURN", step="app_context_enter", app_id=app_id)
            except OSError:
                self._unknown("app", "app_context_enter", app_id)
                raise
            if self.stopped.is_set():
                raise RuntimeError("STOP_LATCHED")
            volume_context = self.backend.volume_context()
            volume = await self._enter_context("volume", volume_context)
            volume_entered = True
            volume_id = getattr(volume, "object_id", None)
            if not isinstance(volume_id, str) or not volume_id:
                self._unknown("volume", "volume_context_enter")
                raise ValueError("Modal Volume returned without object_id")
            try:
                self._record("RETURN", step="volume_context_enter", object_id=volume_id)
            except OSError:
                self._unknown("volume", "volume_context_enter", volume_id)
                raise
            await self._sandbox(app, volume, WRITE_SCRIPT, ("READY:" + DIGEST).encode(), 1)
            if self.stopped.is_set():
                raise RuntimeError("STOP_LATCHED")
            await self._sandbox(
                app, volume, READ_SCRIPT, ("READ:" + str(SIZE) + ":" + DIGEST).encode(), 2
            )
            if self.stopped.is_set():
                raise RuntimeError("STOP_LATCHED")
            iterator = await self._await_operation(
                "volume_iterator_acquire", lambda: self.backend.volume_chunks(volume)
            )
            raw = await self._drain("volume", iterator, line=False)
            if raw != DATA or hashlib.sha256(raw).hexdigest() != DIGEST:
                raise ValueError("Volume byte/hash mismatch")
            self._record("EXACT_BYTES_VERIFIED", length=len(raw), sha256=DIGEST)
            self._result["exact_bytes"] = True
        except BaseException as exc:
            self._first_error(exc)
            self._safe_record("PRIMARY_ERROR", error=type(exc).__name__)
        finally:
            # Only successfully ENTERED contexts receive __aexit__.
            # Always attempt app exit even when volume exit fails.
            if volume_entered:
                await self._exit_context("volume", volume_context)
            if app_entered:
                await self._exit_context("app", app_context)
            complete = (
                not self.stopped.is_set()
                and self._result["primary_error"] is None
                and not self.evidence_errors
                and not self.cleanup_errors
                and not self.unknown_outcomes
                and not self.known_unconfirmed_sandboxes
                and not self.active_operations
                and len(self.sandbox_terminal) == 2
                and self._result["exact_bytes"] is True
                and self._result["app_context_closed"] is True
                and self._result["volume_context_closed"] is True
            )
            self._result["status"] = "SUCCESS" if complete else "BLOCKED"
            self._result["active_operations"] = sorted(self.active_operations)
            # No file-only recovery reader may accept a terminal candidate as SUCCESS:
            # post-write fsync failure leaves visible bytes with unknown durability.
            candidate = self._snapshot()
            candidate["status"] = "UNCONFIRMED"
            candidate["evidence_durability"] = "UNCONFIRMED_FROM_FILE"
            try:
                self._record("TERMINAL", result=candidate)
            except OSError:
                self._result["status"] = "BLOCKED"
                self._result["evidence_durability"] = "UNKNOWN"
            else:
                if self.evidence_errors:
                    self._result["status"] = "BLOCKED"
                    self._result["evidence_durability"] = "UNKNOWN"
                else:
                    # Only the live owner, after successful journal fsync, can
                    # acknowledge the local outcome. Never provider removal.
                    self._result["evidence_durability"] = "OWNER_ACKNOWLEDGED"
        return self._snapshot()
