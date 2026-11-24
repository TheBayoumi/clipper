"""Modal CPU-only 64 KiB lifetime qualification; inert until explicitly invoked.

No Modal import, client creation, CLI entry point or implicit provider calls.
A caller MUST keep the owner event loop alive until terminal() completes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DATA = bytes(range(256)) * 256
DIGEST = hashlib.sha256(DATA).hexdigest()
SIZE = len(DATA)
PATH = "/data/infra-check.bin"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

# No private payload is interpolated into either command.
WRITE_SCRIPT = (
    "import hashlib,os,pathlib,subprocess,time\n"
    "b=bytes(range(256))*256\n"
    "p=pathlib.Path('/data/infra-check.bin')\n"
    "with p.open('wb') as f:\n"
    " f.write(b);f.flush();os.fsync(f.fileno())\n"
    "subprocess.run(['sync','/data'],check=True,timeout=15)\n"
    "print('READY:'+hashlib.sha256(b).hexdigest(),flush=True)\n"
    "time.sleep(55)\n"
)
READ_SCRIPT = (
    "import hashlib,pathlib,time\n"
    "p=pathlib.Path('/data/infra-check.bin')\n"
    "b=p.read_bytes()\n"
    "print('READ:'+str(len(b))+':'+hashlib.sha256(b).hexdigest(),flush=True)\n"
    "time.sleep(55)\n"
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


class SDKBackend:
    """1.5.5 adapter. Caller supplies one explicit client and SDK module.

    No SDK import/client initialization occurs here. These methods are NEVER
    invoked by importing this module or by the offline fake tests.
    """

    def __init__(self, sdk: Any, client: Any, environment: str) -> None:
        if sdk is None or client is None or not environment:
            raise ValueError("Explicit SDK/client/environment required")
        self.sdk = sdk
        self.client = client
        self.environment = environment

    async def verify(self, expected_workspace: str) -> str:
        def check() -> str:
            workspace = self.sdk.Workspace.from_context(client=self.client)
            workspace.hydrate()
            return str(workspace.name())

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
            return await iterator.__anext__()
        except StopAsyncIteration:
            return None

    # Do not transfer StopIteration through an asyncio Future.
    return await asyncio.to_thread(lambda: next(iterator, None))


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


class QualificationOwner:
    """A single retained in-loop owner; no detached tasks or provider retries.

    Decision/reconciliation deadlines bound observation, NOT remote calls.
    A caller must await terminal() before closing its event loop. At deadline
    owner records UNRESOLVED_MONITORING_ACTIVE and continues in this loop.
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
        if decision_seconds <= 0 or reconciliation_seconds <= 0:
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
        self.live_sandboxes: list[Any] = []
        self._result: dict[str, object] = {
            "run_id": self.run_id,
            "sandbox_terminal": [],
            "exact_bytes": False,
            "app_context_closed": False,
            "volume_context_closed": False,
            "provider_removal": "UNKNOWN",
            "primary_error": None,
            "cleanup_errors": [],
        }

    def _record(self, operation: str, **details: object) -> None:
        self.ledger.record(
            run_id=self.run_id,
            source_sha=self.intent.source_sha,
            workspace=self.intent.expected_workspace,
            environment=self.intent.environment,
            operation=operation,
            **details,
        )

    def start(self) -> None:
        if self.task is not None:
            raise ValueError("Owner cannot start twice")
        self._record("RUN_START", intent=self.intent.explicit_run_intent, sha256=DIGEST)
        self.task = asyncio.create_task(self._run())

    def request_stop(self) -> None:
        self.stopped.set()

    async def observe(self) -> dict[str, object]:
        if self.task is None:
            raise ValueError("Owner not started")
        try:
            done, _ = await asyncio.wait({self.task}, timeout=self.decision_seconds)
        except asyncio.CancelledError:
            self.request_stop()
            self._record("CALLER_CANCELLED_OWNER_RETAINED")
            raise
        if done:
            return self.task.result()
        # Decision expiry stops *new* acquisitions, not the retained create
        # task or its subsequent termination/reconciliation.
        self.request_stop()
        self._record("DECISION_DEADLINE", disposition="OWNER_RETAINED_IN_LOOP")
        try:
            done, _ = await asyncio.wait({self.task}, timeout=self.reconciliation_seconds)
        except asyncio.CancelledError:
            self.request_stop()
            self._record("CALLER_CANCELLED_OWNER_RETAINED")
            raise
        if done:
            return self.task.result()
        self._record(
            "UNRESOLVED",
            pending=sorted(self.unresolved),
            disposition="OWNER_MONITORING_CONTINUES_IN_ACTIVE_LOOP",
        )
        return {"status": "UNRESOLVED", "run_id": self.run_id, "owner_task_alive": True}

    async def terminal(self) -> dict[str, object]:
        if self.task is None:
            raise ValueError("Owner not started")
        # No wait_for: cancellation of a waiter does not cancel owned work.
        return await asyncio.shield(self.task)

    async def _operation(self, name: str, action: Any, *, acquisition: bool = False) -> Any:
        if acquisition and self.stopped.is_set():
            raise RuntimeError("STOP_LATCHED")
        try:
            self._record("INTENT", step=name)
        except OSError:
            if acquisition:
                raise
            # Cleanup is mandatory even if the evidence sink is unavailable.
            self._result["cleanup_errors"].append("JournalIntentFailed")  # type: ignore[union-attr]
        self.unresolved.add(name)
        try:
            result = await action()
            try:
                self._record("RETURN", step=name, object_id=str(getattr(result, "object_id", "")))
            except OSError:
                if acquisition:
                    raise
                self._result["cleanup_errors"].append("JournalReturnFailed")  # type: ignore[union-attr]
            return result
        finally:
            self.unresolved.discard(name)

    async def _drain(self, name: str, iterator: Any, *, line: bool) -> bytes:
        error: BaseException | None = None
        try:
            return await (read_first_line(iterator) if line else read_exact(iterator))
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                await self._operation(name + "_iterator_close", lambda: _close_iterator(iterator))
            except BaseException as exc:
                self._result["cleanup_errors"].append(type(exc).__name__)  # type: ignore[union-attr]
                self._record("ITERATOR_CLOSE_FAILED", step=name, error=type(exc).__name__)
                if error is None:
                    raise

    async def _sandbox(
        self, app: Any, volume: Any, script: str, expected: bytes, index: int
    ) -> None:
        if self.stopped.is_set():
            raise RuntimeError("STOP_LATCHED")
        step = f"sandbox_{index}_create"
        self._record("INTENT", step=step)
        self.unresolved.add(step)
        try:
            sb = await self.backend.create(app, volume, script)
        except BaseException:
            # The provider may have allocated a Sandbox without returning
            # its ID. Never retry or falsely assert non-acquisition.
            self._record("UNKNOWN_ID_OUTCOME", step=step, outcome="UNRESOLVED")
            raise
        finally:
            self.unresolved.discard(step)
        # Retain the returned handle BEFORE attempting a second journal write.
        self.live_sandboxes.append(sb)
        primary: BaseException | None = None
        try:
            self._record("RETURN", step=step, object_id=str(getattr(sb, "object_id", "")))
            if not getattr(sb, "object_id", None):
                self._record("UNKNOWN_SANDBOX_ID", step=index)
            if self.stopped.is_set():
                raise RuntimeError("STOP_LATCHED_AFTER_CREATE")
            iterator = await self.backend.stdout_chunks(sb)
            first = await self._drain(f"sandbox_{index}_stdout", iterator, line=True)
            if first != expected:
                raise ValueError("Invalid first readiness line")
            self._record("READINESS", step=index)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                await self._operation(
                    f"sandbox_{index}_terminate", lambda: self.backend.terminate(sb)
                )
                terminal = await self._operation(
                    f"sandbox_{index}_poll", lambda: self.backend.poll(sb)
                )
                if terminal is None:
                    raise ValueError("No terminal observation")
                self._record("SANDBOX_TERMINAL", step=index, terminal=str(terminal))
                self._result["sandbox_terminal"].append(str(terminal))  # type: ignore[union-attr]
            except BaseException as exc:
                self._record("SANDBOX_CLEANUP_ERROR", step=index, error=type(exc).__name__)
                self._result["cleanup_errors"].append(type(exc).__name__)  # type: ignore[union-attr]
                if primary is None:
                    raise

    async def _run(self) -> dict[str, object]:
        try:
            verified = await self.backend.verify(self.intent.expected_workspace)
            if verified != self.intent.expected_workspace:
                raise ValueError("Client-bound workspace mismatch")
            self._record("WORKSPACE_VERIFIED")
            app_context = self.backend.app_context()
            volume_context: Any = None
            app: Any = None
            volume: Any = None
            try:
                if self.stopped.is_set():
                    raise RuntimeError("STOP_LATCHED")
                self._record("INTENT", step="app_context_enter")
                try:
                    app = await app_context.__aenter__()
                except BaseException:
                    self._record("UNKNOWN_ID_OUTCOME", step="app_context_enter", outcome="UNRESOLVED")
                    raise
                # Context return is held locally before its durable ID record.
                self._record(
                    "RETURN", step="app_context_enter", object_id=str(getattr(app, "object_id", ""))
                )
                volume_context = self.backend.volume_context()
                try:
                    if self.stopped.is_set():
                        raise RuntimeError("STOP_LATCHED")
                    self._record("INTENT", step="volume_context_enter")
                    try:
                        volume = await volume_context.__aenter__()
                    except BaseException:
                        self._record(
                            "UNKNOWN_ID_OUTCOME", step="volume_context_enter", outcome="UNRESOLVED"
                        )
                        raise
                    self._record(
                        "RETURN",
                        step="volume_context_enter",
                        object_id=str(getattr(volume, "object_id", "")),
                    )
                    await self._sandbox(
                        app,
                        volume,
                        WRITE_SCRIPT,
                        ("READY:" + DIGEST).encode(),
                        1,
                    )
                    if self.stopped.is_set():
                        raise RuntimeError("STOP_LATCHED")
                    await self._sandbox(
                        app,
                        volume,
                        READ_SCRIPT,
                        ("READ:" + str(SIZE) + ":" + DIGEST).encode(),
                        2,
                    )
                    if self.stopped.is_set():
                        raise RuntimeError("STOP_LATCHED")
                    iterator = await self.backend.volume_chunks(volume)
                    raw = await self._drain("volume", iterator, line=False)
                    if raw != DATA or hashlib.sha256(raw).hexdigest() != DIGEST:
                        raise ValueError("Volume byte/hash mismatch")
                    self._result["exact_bytes"] = True
                    self._record("EXACT_BYTES_VERIFIED", length=len(raw), sha256=DIGEST)
                finally:
                    if volume_context is not None:
                        await self._operation(
                            "volume_context_exit",
                            lambda: volume_context.__aexit__(None, None, None),
                        )
                        self._result["volume_context_closed"] = True
            finally:
                if app is not None:
                    await self._operation(
                        "app_context_exit", lambda: app_context.__aexit__(None, None, None)
                    )
                    self._result["app_context_closed"] = True
        except BaseException as exc:
            self._result["primary_error"] = type(exc).__name__
            self._record("PRIMARY_ERROR", error=type(exc).__name__)
        self._record("TERMINAL", result=self._result)
        return dict(self._result)
