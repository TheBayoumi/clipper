"""Offline Modal qualification: fake backend only, no SDK import or provider calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from clipper.modal_infra_qualification import (
    DATA,
    DIGEST,
    READ_SCRIPT,
    SIZE,
    WRITE_SCRIPT,
    Intent,
    Ledger,
    QualificationOwner,
    SDKBackend,
    read_exact,
    read_first_line,
)


class Iterator:
    def __init__(self, chunks: list[bytes | str], *, error: bool = False) -> None:
        self.chunks = iter(chunks)
        self.closed = False
        self.error = error

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> bytes | str:
        if self.error:
            raise OSError("synthetic stream error")
        return next(self.chunks)

    def close(self) -> None:
        self.closed = True


class Context:
    def __init__(self, backend: FakeBackend, name: str) -> None:
        self.backend = backend
        self.name = name
        # Match Modal 1.5.5: App.app_id, Volume.object_id.
        self.app_id = "ap-synthetic" if name == "app" else None
        self.object_id = "vo-synthetic" if name == "volume" else None

    async def __aenter__(self) -> Context:
        self.backend.events.append(self.name + "_enter")
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.backend.events.append(self.name + "_exit")


class Sandbox:
    def __init__(self, index: int) -> None:
        self.object_id = f"sb-synthetic-{index}"


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.sandboxes: list[Sandbox] = []
        self.stored = b""
        self.corrupt = False
        self.entered_create = asyncio.Event()
        self.release_create = asyncio.Event()
        self.hold_create = False
        self.hold_terminate = False
        self.entered_cleanup = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.mismatch = False
        self.iterators: list[Iterator] = []

    async def verify(self, expected_workspace: str) -> str:
        self.events.append("verify")
        return "wrong-workspace" if self.mismatch else expected_workspace

    def app_context(self) -> Context:
        return Context(self, "app")

    def volume_context(self) -> Context:
        return Context(self, "volume")

    async def create(self, _app: Any, _volume: Any, script: str) -> Sandbox:
        self.events.append("create")
        self.entered_create.set()
        if self.hold_create:
            await self.release_create.wait()
        sb = Sandbox(len(self.sandboxes) + 1)
        self.sandboxes.append(sb)
        if script == WRITE_SCRIPT:
            self.stored = bytes(range(256)) * 256
            assert hashlib.sha256(self.stored).hexdigest() == DIGEST
        else:
            assert script == READ_SCRIPT
        return sb

    async def stdout_chunks(self, sandbox: Sandbox) -> Iterator:
        if sandbox.object_id.endswith("1"):
            data = ("READY:" + DIGEST + "\n").encode()
        else:
            data = ("READ:" + str(SIZE) + ":" + DIGEST + "\n").encode()
        iterator = Iterator([data[:4], data[4:] + b"x" * 512_000])
        self.iterators.append(iterator)
        return iterator

    async def volume_chunks(self, _volume: Any) -> Iterator:
        raw = self.stored[:-1] if self.corrupt else self.stored
        iterator = Iterator([raw[:5], raw[5:]])
        self.iterators.append(iterator)
        return iterator

    async def terminate(self, sandbox: Sandbox) -> None:
        self.events.append(f"terminate_{sandbox.object_id}")
        self.entered_cleanup.set()
        if self.hold_terminate:
            await self.release_cleanup.wait()

    async def poll(self, sandbox: Sandbox) -> int:
        self.events.append(f"poll_{sandbox.object_id}")
        return 137


def _owner(tmp_path: Path, backend: FakeBackend) -> QualificationOwner:
    return QualificationOwner(
        backend,
        Intent("explicit-workspace", "qualification", "a" * 40, "modal-cpu-synthetic-persistence"),
        Ledger(tmp_path / "private" / "ledger.jsonl"),
        decision_seconds=0.02,
        reconciliation_seconds=0.02,
    )


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "private" / "ledger.jsonl").read_text().splitlines()
    ]


def test_offline_exact_bytes_sequential_termination_and_ledger(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FakeBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["exact_bytes"] is True
        assert result["primary_error"] is None
        assert result["provider_removal"] == "UNKNOWN"
        assert result["app_context_closed"] and result["volume_context_closed"]
        assert result["sandbox_terminal"] == ["137", "137"]
        assert backend.events.index("poll_sb-synthetic-1") < backend.events.index(
            "create", backend.events.index("poll_sb-synthetic-1")
        )
        assert all(iterator.closed for iterator in backend.iterators)
        assert hashlib.sha256(backend.stored).hexdigest() == DIGEST
        entries = _events(tmp_path)
        assert [x["step"] for x in entries if x["operation"] == "INTENT"][:3] == [
            "app_context_enter",
            "volume_context_enter",
            "sandbox_1_create",
        ]
        assert [
            x["object_id"]
            for x in entries
            if x["operation"] == "RETURN" and x["step"].endswith("_create")
        ] == ["sb-synthetic-1", "sb-synthetic-2"]
        app_returns = [
            x
            for x in entries
            if x["operation"] == "RETURN" and x.get("step") == "app_context_enter"
        ]
        assert len(app_returns) == 1
        assert app_returns[0]["app_id"] == "ap-synthetic"
        assert "object_id" not in app_returns[0]

    asyncio.run(exercise())


def test_mismatch_prevents_any_context_or_create(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FakeBackend()
        backend.mismatch = True
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert backend.events == ["verify"]

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_during", ["create", "cleanup"])
def test_repeated_caller_cancel_retains_owner_and_cleans_late_handle(
    tmp_path: Path, cancel_during: str
) -> None:
    async def exercise() -> None:
        backend = FakeBackend()
        backend.hold_create = cancel_during == "create"
        backend.hold_terminate = cancel_during == "cleanup"
        owner = _owner(tmp_path, backend)
        owner.start()
        if cancel_during == "create":
            await backend.entered_create.wait()
        else:
            await backend.entered_cleanup.wait()
        for _ in range(2):
            observer = asyncio.create_task(owner.observe())
            await asyncio.sleep(0)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        assert owner.task is not None and not owner.task.done()
        assert (await owner.observe())["status"] == "UNRESOLVED"
        if cancel_during == "create":
            backend.release_create.set()
        else:
            backend.release_cleanup.set()
        final = await owner.terminal()
        assert final["primary_error"] == "RuntimeError"
        assert len(backend.sandboxes) == 1
        assert "poll_sb-synthetic-1" in backend.events
        assert backend.events[-2:] == ["volume_exit", "app_exit"]
        assert any(x["operation"] == "UNRESOLVED" for x in _events(tmp_path))

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "chunks",
    [
        [b"READY:ok\n" + b"x" * 200000],
        [b"RE", b"ADY:ok\n", b"x" * 200000],
    ],
)
def test_readiness_ignores_suffix_and_closes_on_owner(tmp_path: Path, chunks: list[bytes]) -> None:
    async def exercise() -> None:
        iterator = Iterator(chunks)
        assert await read_first_line(iterator, maximum=32) == b"READY:ok"
        iterator.close()
        assert iterator.closed

    asyncio.run(exercise())


@pytest.mark.parametrize("chunks", [[b"abcd"], [b"a" * 129 + b"\n"], [], [b"abc", b"def"]])
def test_readiness_rejects_eof_overflow_and_missing_newline(chunks: list[bytes]) -> None:
    async def exercise() -> None:
        with pytest.raises(ValueError):
            await read_first_line(Iterator(chunks))

    asyncio.run(exercise())


def test_volume_cap_and_iterator_error() -> None:
    async def exercise() -> None:
        assert await read_exact(Iterator([DATA[:50], DATA[50:]])) == DATA
        with pytest.raises(ValueError):
            await read_exact(Iterator([DATA, b"overflow"]))
        with pytest.raises(OSError):
            await read_exact(Iterator([], error=True))

    asyncio.run(exercise())


def test_volume_corruption_fails_independently_of_readiness(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FakeBackend()
        backend.corrupt = True
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert result["exact_bytes"] is False
        assert len(result["sandbox_terminal"]) == 2

    asyncio.run(exercise())


def test_sdk_injection_has_no_provider_work_at_construction() -> None:
    class SDK:
        pass

    client = object()
    backend = SDKBackend(SDK(), client, "qualification")
    assert backend.client is client
    assert backend.environment == "qualification"


def test_ledger_preintent_fail_closed_and_private_permissions(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger")
    ledger.record(operation="INTENT", step="synthetic")
    assert ledger.path.stat().st_mode & 0o077 == 0
    ledger.close()
    assert json.loads(ledger.path.read_text())["operation"] == "INTENT"
    with pytest.raises(ValueError):
        Intent("w", "e", "incorrect", "modal-cpu-synthetic-persistence").validate()


def test_sdk_explicit_client_workspace_and_resource_wiring() -> None:
    async def exercise() -> None:
        from types import SimpleNamespace

        calls: list[tuple[str, object]] = []
        client = object()
        ambient = object()

        class Workspace:
            @staticmethod
            def from_context(*, client: object) -> object:
                calls.append(("workspace_client", client))
                return SimpleNamespace(
                    hydrate=lambda: None,
                    name="approved-workspace",
                )

        class ModalApp:
            def run(self, *, client: object, environment_name: str, detach: bool) -> object:
                calls.append(("app_client", client))
                calls.append(("app_environment", environment_name))
                assert detach is False
                return "app-context"

        class ModalVolume:
            @staticmethod
            def ephemeral(*, client: object, environment_name: str) -> object:
                calls.append(("volume_client", client))
                calls.append(("volume_environment", environment_name))
                return "volume-context"

        class ModalImage:
            @staticmethod
            def debian_slim() -> str:
                return "synthetic-image"

        class ModalSandbox:
            @staticmethod
            def create(*args: object, **kwargs: object) -> object:
                assert args[:2] == ("python3", "-c")
                assert args[2] == WRITE_SCRIPT
                assert kwargs["client"] is client
                assert kwargs["app"] == "owned-app"
                assert kwargs["volumes"] == {"/data": "owned-volume"}
                assert kwargs["cpu"] == (0.25, 0.25)
                assert kwargs["memory"] == (256, 256)
                assert kwargs["gpu"] is None
                assert kwargs["secrets"] == []
                assert kwargs["encrypted_ports"] == []
                assert kwargs["unencrypted_ports"] == []
                assert kwargs["h2_ports"] == []
                assert kwargs["block_network"] is True
                assert kwargs["timeout"] == 60
                assert "environment_name" not in kwargs
                return "sandbox"

        sdk = SimpleNamespace(
            Workspace=Workspace,
            App=ModalApp,
            Volume=ModalVolume,
            Image=ModalImage,
            Sandbox=ModalSandbox,
        )
        backend = SDKBackend(sdk, client, "qualification")
        assert await backend.verify("approved-workspace") == "approved-workspace"
        # Ambient identity is deliberately mutated after client-bound verification.
        ambient = object()
        assert ambient is not client
        assert backend.app_context() == "app-context"
        assert backend.volume_context() == "volume-context"
        assert await backend.create("owned-app", "owned-volume", WRITE_SCRIPT) == "sandbox"
        assert [x[1] for x in calls if x[0].endswith("_client")] == [client, client, client]
        assert [x[1] for x in calls if x[0].endswith("_environment")] == [
            "qualification",
            "qualification",
        ]
        with pytest.raises(ValueError, match="mismatch"):
            await backend.verify("other-workspace")

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failed_step",
    [
        ("INTENT", "app_context_enter"),
        ("RETURN", "app_context_enter"),
        ("INTENT", "volume_context_enter"),
        ("RETURN", "volume_context_enter"),
        ("INTENT", "sandbox_1_create"),
        ("RETURN", "sandbox_1_create"),
    ],
)
def test_journal_failure_does_not_skip_teardown(
    tmp_path: Path, failed_step: tuple[str, str]
) -> None:
    class FlakyLedger(Ledger):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.failed = False

        def record(self, **fields: object) -> None:
            if (
                not self.failed
                and fields.get("operation") == failed_step[0]
                and fields.get("step") == failed_step[1]
            ):
                self.failed = True
                raise OSError("synthetic fsync/write failure")
            super().record(**fields)

    async def exercise() -> None:
        backend = FakeBackend()
        owner = QualificationOwner(
            backend,
            Intent(
                "explicit-workspace", "qualification", "a" * 40, "modal-cpu-synthetic-persistence"
            ),
            FlakyLedger(tmp_path / "private" / "ledger.jsonl"),
        )
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "OSError"
        assert (
            backend.events[-1:] == ["app_exit"]
            if failed_step[1] != "app_context_enter" or failed_step[0] == "RETURN"
            else backend.events == ["verify"]
        )
        assert result["status"] == "BLOCKED"
        if failed_step[0] == "RETURN":
            expected = {
                "app_context_enter": ("app", "ap-synthetic"),
                "volume_context_enter": ("volume", "vo-synthetic"),
                "sandbox_1_create": ("sandbox", "sb-synthetic-1"),
            }[failed_step[1]]
            assert any(
                item["kind"] == expected[0] and item["id"] == expected[1]
                for item in result["unknown_outcomes"]
            )
        assert len(backend.sandboxes) <= 1
        if backend.sandboxes:
            assert "poll_sb-synthetic-1" in backend.events
        assert (
            not any(x == "create" for x in backend.events[backend.events.index("create") + 1 :])
            if "create" in backend.events
            else True
        )

    asyncio.run(exercise())


def test_unknown_create_no_retry_and_durable_intent(tmp_path: Path) -> None:
    class UnknownBackend(FakeBackend):
        async def create(self, _app: Any, _volume: Any, _script: str) -> Sandbox:
            self.events.append("create")
            raise OSError("ack lost; resource ID unknown")

    async def exercise() -> None:
        backend = UnknownBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "OSError"
        assert backend.events.count("create") == 1
        assert backend.events[-2:] == ["volume_exit", "app_exit"]
        assert any(
            x["operation"] == "INTENT" and x.get("step") == "sandbox_1_create"
            for x in _events(tmp_path)
        )

    asyncio.run(exercise())


def test_decision_expiry_stops_late_create_before_second_allocation(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FakeBackend()
        backend.hold_create = True
        owner = _owner(tmp_path, backend)
        owner.start()
        await backend.entered_create.wait()
        result = await owner.observe()
        assert result["status"] == "UNRESOLVED"
        assert result["owner_task_alive"] is True
        assert owner.task is not None and not owner.task.done()
        backend.release_create.set()
        final = await owner.terminal()
        assert final["primary_error"] == "RuntimeError"
        assert len(backend.sandboxes) == 1
        assert "poll_sb-synthetic-1" in backend.events
        assert backend.events[-2:] == ["volume_exit", "app_exit"]

    asyncio.run(exercise())


def test_real_local_synthetic_child_scripts_byte_exact(tmp_path: Path) -> None:
    # Run the actual immutable child snippets entirely offline with their mount
    # path redirected to a temporary directory; no Modal or /data writes.
    source = tmp_path / "infra-check.bin"
    writer = WRITE_SCRIPT.replace("/data/infra-check.bin", str(source))
    writer = writer.replace("while True: time.sleep(10)", "time.sleep(0)")
    reader = READ_SCRIPT.replace("/data/infra-check.bin", str(source))
    reader = reader.replace("while True: time.sleep(10)", "time.sleep(0)")
    written = subprocess.run(
        [sys.executable, "-c", writer], capture_output=True, check=True, timeout=5
    )
    assert written.stdout == ("READY:" + DIGEST + "\n").encode()
    assert source.read_bytes() == DATA
    assert source.stat().st_size == 65_536
    readback = subprocess.run(
        [sys.executable, "-c", reader], capture_output=True, check=True, timeout=5
    )
    assert readback.stdout == ("READ:" + str(SIZE) + ":" + DIGEST + "\n").encode()
    assert source.read_bytes() == DATA


@pytest.mark.parametrize("stage", ["stdout", "volume"])
def test_failing_iterator_close_records_evidence_and_retains_contexts(
    tmp_path: Path, stage: str
) -> None:
    class BrokenClose(Iterator):
        def close(self) -> None:
            self.closed = True
            raise OSError("synthetic iterator close failure")

    class BrokenBackend(FakeBackend):
        async def stdout_chunks(self, sandbox: Sandbox) -> Iterator:
            result = await super().stdout_chunks(sandbox)
            if stage == "stdout":
                return BrokenClose([("READY:" + DIGEST + "\n").encode()])
            return result

        async def volume_chunks(self, volume: Any) -> Iterator:
            result = await super().volume_chunks(volume)
            if stage == "volume":
                return BrokenClose([DATA])
            return result

    async def exercise() -> None:
        backend = BrokenBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "OSError"
        assert "OSError" in result["cleanup_errors"]
        assert result["provider_removal"] == "UNKNOWN"
        assert result["app_context_closed"] and result["volume_context_closed"]
        assert len(result["sandbox_terminal"]) == (1 if stage == "stdout" else 2)

    asyncio.run(exercise())


def test_stalled_iterator_close_retains_owner_through_decision_deadline(tmp_path: Path) -> None:
    class StallIterator:
        def __init__(self, release: asyncio.Event, entered: asyncio.Event) -> None:
            self._chunks = iter([("READY:" + DIGEST + "\n").encode()])
            self.release = release
            self.entered = entered

        def __aiter__(self) -> StallIterator:
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def aclose(self) -> None:
            self.entered.set()
            await self.release.wait()

    class StallBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.close_entered = asyncio.Event()
            self.close_release = asyncio.Event()

        async def stdout_chunks(self, sandbox: Sandbox) -> Any:
            if sandbox.object_id.endswith("1"):
                return StallIterator(self.close_release, self.close_entered)
            return await super().stdout_chunks(sandbox)

    async def exercise() -> None:
        backend = StallBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        await backend.close_entered.wait()
        result = await owner.observe()
        assert result["status"] == "UNRESOLVED"
        assert owner.task is not None and not owner.task.done()
        backend.close_release.set()
        final = await owner.terminal()
        assert final["primary_error"] == "RuntimeError"
        assert len(backend.sandboxes) == 1
        assert "poll_sb-synthetic-1" in backend.events
        assert backend.events[-2:] == ["volume_exit", "app_exit"]
        assert any(x["operation"] == "UNRESOLVED" for x in _events(tmp_path))

    asyncio.run(exercise())


@pytest.mark.parametrize("bad_line", [b"BAD\n", b"a" * 129 + b"\n", b"no-eof"])
def test_owner_rejects_bad_first_line_without_second_sandbox(
    tmp_path: Path, bad_line: bytes
) -> None:
    class BadBackend(FakeBackend):
        async def stdout_chunks(self, sandbox: Sandbox) -> Iterator:
            if sandbox.object_id.endswith("1"):
                iterator = Iterator([bad_line])
                self.iterators.append(iterator)
                return iterator
            return await super().stdout_chunks(sandbox)

    async def exercise() -> None:
        backend = BadBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert len(backend.sandboxes) == 1
        assert "poll_sb-synthetic-1" in backend.events
        assert backend.iterators[0].closed
        assert result["exact_bytes"] is False

    asyncio.run(exercise())


def test_unobserved_terminal_is_not_cleanup_success(tmp_path: Path) -> None:
    class NoPoll(FakeBackend):
        async def poll(self, sandbox: Sandbox) -> Any:
            self.events.append("poll_unknown")
            return None

    async def exercise() -> None:
        backend = NoPoll()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert result["sandbox_terminal"] == []
        assert len(backend.sandboxes) == 1
        assert result["provider_removal"] == "UNKNOWN"
        assert result["app_context_closed"] and result["volume_context_closed"]

    asyncio.run(exercise())


def test_context_exit_failure_never_implies_provider_removal(tmp_path: Path) -> None:
    class BadContext(Context):
        async def __aexit__(self, *_args: object) -> None:
            self.backend.events.append(self.name + "_exit_failed")
            raise OSError("synthetic context exit failure")

    class ContextFailure(FakeBackend):
        def volume_context(self) -> Context:
            return BadContext(self, "volume")

    async def exercise() -> None:
        backend = ContextFailure()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "OSError"
        assert result["volume_context_closed"] is False
        assert result["app_context_closed"] is True
        assert result["provider_removal"] == "UNKNOWN"
        assert result["exact_bytes"] is True
        assert len(result["sandbox_terminal"]) == 2

    asyncio.run(exercise())


def test_durable_intent_survives_synthetic_process_interruption(tmp_path: Path) -> None:
    journal_path = tmp_path / "private" / "ledger.jsonl"
    snippet = (
        "import os,pathlib,sys\n"
        "from clipper.modal_infra_qualification import Ledger\n"
        "ledger=Ledger(pathlib.Path(sys.argv[1]))\n"
        "ledger.record(operation='INTENT',step='sandbox_create',run_id='synthetic')\n"
        "os._exit(7)\n"
    )
    interrupted = subprocess.run(
        [sys.executable, "-c", snippet, str(journal_path)],
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert interrupted.returncode == 7
    evidence = json.loads(journal_path.read_text())
    assert evidence == {
        "operation": "INTENT",
        "run_id": "synthetic",
        "step": "sandbox_create",
    }


@pytest.mark.parametrize("workspace_name", [None, "", "wrong-workspace"])
def test_sdk_workspace_name_property_fails_before_any_resource_entry(
    tmp_path: Path, workspace_name: str | None
) -> None:
    """Match the SDK's Optional[str] property, not a callable test double."""
    from types import SimpleNamespace

    async def exercise() -> None:
        client = object()
        observed: list[str] = []

        class Workspace:
            @property
            def name(self) -> str | None:
                return workspace_name

            def hydrate(self) -> None:
                observed.append("hydrate")

        class WorkspaceAPI:
            @staticmethod
            def from_context(*, client: object) -> Workspace:
                assert client is explicit_client
                observed.append("workspace")
                return Workspace()

        explicit_client = client
        sdk = SimpleNamespace(Workspace=WorkspaceAPI)

        class NoResourceBackend(SDKBackend):
            def app_context(self) -> Any:
                observed.append("app_context")
                pytest.fail("Invalid workspace must not acquire an App")

            def volume_context(self) -> Any:
                observed.append("volume_context")
                pytest.fail("Invalid workspace must not acquire a Volume")

        owner = QualificationOwner(
            NoResourceBackend(sdk, explicit_client, "qualification"),
            Intent(
                "expected-workspace",
                "qualification",
                "a" * 40,
                "modal-cpu-synthetic-persistence",
            ),
            Ledger(tmp_path / "private" / "ledger.jsonl"),
        )
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert observed == ["workspace", "hydrate"]
        assert result["provider_removal"] == "UNKNOWN"
        assert not any(
            e["operation"] == "INTENT"
            for e in _events(tmp_path)
            if e.get("step") in {"app_context_enter", "volume_context_enter"}
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("app_id", [None, ""])
def test_missing_app_id_is_not_logged_as_success_and_context_still_exits(
    tmp_path: Path, app_id: str | None
) -> None:
    class MissingAppIdBackend(FakeBackend):
        def app_context(self) -> Context:
            ctx = Context(self, "app")
            ctx.app_id = app_id
            return ctx

    async def exercise() -> None:
        backend = MissingAppIdBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert backend.events == ["verify", "app_enter", "app_exit"]
        assert result["app_context_closed"] is True
        assert result["volume_context_closed"] is False
        assert result["provider_removal"] == "UNKNOWN"
        events = _events(tmp_path)
        assert any(
            e["operation"] == "UNKNOWN_APP_ID" and e.get("step") == "app_context_enter"
            for e in events
        )
        assert not any(
            e["operation"] == "RETURN" and e.get("step") == "app_context_enter" for e in events
        )

    asyncio.run(exercise())


def test_sdk_combined_async_contexts_use_single_explicit_client() -> None:
    """SDK-shaped App.run/Volume.ephemeral context managers, no Modal import."""
    from types import SimpleNamespace

    async def exercise() -> None:
        client = object()
        events: list[str] = []

        class AsyncContext:
            def __init__(self, kind: str, value: object) -> None:
                self.kind, self.value = kind, value

            async def __aenter__(self) -> object:
                events.append(self.kind + "_enter")
                return self.value

            async def __aexit__(self, *_args: object) -> None:
                events.append(self.kind + "_exit")

        class App:
            app_id = "ap-actual-shaped"

            def run(self, *, client: object, environment_name: str, detach: bool) -> AsyncContext:
                assert client is explicit_client
                assert environment_name == "qualification"
                assert detach is False
                return AsyncContext("app", self)

        class Volume:
            object_id = "vo-actual-shaped"

            @classmethod
            def ephemeral(cls, *, client: object, environment_name: str) -> AsyncContext:
                assert client is explicit_client
                assert environment_name == "qualification"
                return AsyncContext("volume", cls())

        explicit_client = client
        backend = SDKBackend(
            SimpleNamespace(App=App, Volume=Volume), explicit_client, "qualification"
        )
        app_context = backend.app_context()
        async with app_context as app:
            assert app.app_id == "ap-actual-shaped"
            async with backend.volume_context() as volume:
                assert volume.object_id == "vo-actual-shaped"
        assert events == ["app_enter", "volume_enter", "volume_exit", "app_exit"]

    asyncio.run(exercise())


def test_sdk_cpu_transport_helpers_use_supplied_handles_only() -> None:
    from types import SimpleNamespace

    async def exercise() -> None:
        calls: list[object] = []

        class Volume:
            def read_file(self, path: str) -> list[bytes]:
                calls.append(("volume_path", path))
                return [DATA[:4], DATA[4:]]

        class Sandbox:
            def __init__(self) -> None:
                self.stdout = ["READY:ok\n"]

            def terminate(self, *, wait: bool) -> None:
                calls.append(("terminate_wait", wait))

            def poll(self) -> int:
                calls.append("poll")
                return 137

        backend = SDKBackend(SimpleNamespace(), object(), "qualification")
        output = await backend.stdout_chunks(Sandbox())
        assert list(output) == ["READY:ok\n"]
        volume = await backend.volume_chunks(Volume())
        assert b"".join(volume) == DATA
        sandbox = Sandbox()
        await backend.terminate(sandbox)
        assert await backend.poll(sandbox) == 137
        assert calls == [
            ("volume_path", "infra-check.bin"),
            ("terminate_wait", True),
            "poll",
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ["sdk", "client", "environment"])
def test_sdk_backend_requires_explicit_injected_dependencies(missing: str) -> None:
    sdk: object | None = None if missing == "sdk" else object()
    client: object | None = None if missing == "client" else object()
    env = "" if missing == "environment" else "qualification"
    with pytest.raises(ValueError, match="Explicit SDK/client/environment"):
        SDKBackend(sdk, client, env)


def test_ledger_rejects_symlink_world_readable_and_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    root = tmp_path / "private"
    root.mkdir()
    target = root / "target.jsonl"
    target.write_text("{}\n")
    target.chmod(0o600)
    (root / "alias.jsonl").symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        Ledger(root / "alias.jsonl")
    target.chmod(0o644)
    with pytest.raises(ValueError, match="owner-private"):
        Ledger(target)
    target.chmod(0o600)
    ledger = Ledger(target)
    try:
        original_write = os.write
        with monkeypatch.context() as patch:
            patch.setattr(os, "write", lambda fd, data: len(data) - 1)
            with pytest.raises(OSError, match="Incomplete journal record"):
                ledger.record(operation="INTENT", step="synthetic")
        assert callable(original_write)
    finally:
        ledger.close()


def test_async_iterator_eof_and_invalid_readiness_type() -> None:
    async def exercise() -> None:
        async def chunks() -> Any:
            yield b"ABC"
            yield b"DEF"

        assert await read_exact(chunks()) == b"ABCDEF"
        with pytest.raises(ValueError, match="text or bytes"):
            await read_first_line(Iterator([object()]))  # type: ignore[list-item]

    asyncio.run(exercise())


def test_owner_prestart_double_start_and_postterminal_observe(tmp_path: Path) -> None:
    async def exercise() -> None:
        backend = FakeBackend()
        owner = _owner(tmp_path, backend)
        with pytest.raises(ValueError, match="not started"):
            await owner.observe()
        with pytest.raises(ValueError, match="not started"):
            await owner.terminal()
        owner.start()
        with pytest.raises(ValueError, match="start twice"):
            owner.start()
        result = await owner.terminal()
        assert result["exact_bytes"]
        assert await owner.observe() == result

    asyncio.run(exercise())


@pytest.mark.parametrize("context_kind", ["app", "volume"])
def test_context_enter_unknown_id_fail_closed_without_create(
    tmp_path: Path, context_kind: str
) -> None:
    class FailedEntry(Context):
        async def __aenter__(self) -> Context:
            self.backend.events.append(self.name + "_enter")
            raise OSError("Synthetic context entered without returned handle")

    class BrokenBackend(FakeBackend):
        def app_context(self) -> Context:
            return FailedEntry(self, "app") if context_kind == "app" else super().app_context()

        def volume_context(self) -> Context:
            return (
                FailedEntry(self, "volume")
                if context_kind == "volume"
                else super().volume_context()
            )

    async def exercise() -> None:
        backend = BrokenBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "OSError"
        assert "create" not in backend.events
        if context_kind == "app":
            assert backend.events == ["verify", "app_enter"]
        else:
            # An unsuccessful __aenter__ must NOT receive __aexit__.
            assert backend.events == ["verify", "app_enter", "volume_enter", "app_exit"]
        assert any(
            entry["operation"] == "UNKNOWN_ID_OUTCOME"
            and entry.get("step") == context_kind + "_context_enter"
            for entry in _events(tmp_path)
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("missing_kind", ["volume", "sandbox"])
def test_missing_resource_id_latches_stop_and_preserves_unknown_evidence(
    tmp_path: Path, missing_kind: str
) -> None:
    class MissingVolume(Context):
        def __init__(self, backend: FakeBackend) -> None:
            super().__init__(backend, "volume")
            self.object_id = ""

    class MissingBackend(FakeBackend):
        def volume_context(self) -> Context:
            return MissingVolume(self) if missing_kind == "volume" else super().volume_context()

        async def create(self, app: Any, volume: Any, script: str) -> Sandbox:
            sb = await super().create(app, volume, script)
            if missing_kind == "sandbox":
                sb.object_id = ""
            return sb

    async def exercise() -> None:
        backend = MissingBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["status"] == "BLOCKED"
        assert result["primary_error"] == "ValueError"
        assert result["exact_bytes"] is False
        assert result["provider_removal"] == "UNKNOWN"
        expected_kind = missing_kind
        assert any(item["kind"] == expected_kind for item in result["unknown_outcomes"])
        assert any(
            item["kind"] == expected_kind
            for item in _events(tmp_path)
            if item["operation"] == "UNKNOWN_ID_OUTCOME"
        )
        assert len(backend.sandboxes) == (1 if missing_kind == "sandbox" else 0)
        if backend.sandboxes:
            assert "terminate_" in " ".join(backend.events)
            assert "poll_" in " ".join(backend.events)
        assert backend.events[-2:] == ["volume_exit", "app_exit"]

    asyncio.run(exercise())


@pytest.mark.parametrize("journal_event", ["INTENT", "RETURN"])
def test_termination_evidence_failure_stops_second_sandbox_and_still_cleans(
    tmp_path: Path, journal_event: str
) -> None:
    class BrokenJournal(Ledger):
        failed = False

        def record(self, **fields: object) -> None:
            if (
                not self.failed
                and fields.get("operation") == journal_event
                and fields.get("step") == "sandbox_1_terminate"
            ):
                self.failed = True
                raise OSError("synthetic teardown journal unavailable")
            super().record(**fields)

    async def exercise() -> None:
        backend = FakeBackend()
        owner = QualificationOwner(
            backend,
            Intent(
                "explicit-workspace", "qualification", "a" * 40, "modal-cpu-synthetic-persistence"
            ),
            BrokenJournal(tmp_path / "private" / "ledger.jsonl"),
        )
        owner.start()
        result = await owner.terminal()
        assert result["status"] == "BLOCKED"
        assert result["evidence_errors"]
        assert result["exact_bytes"] is False
        assert len(backend.sandboxes) == 1
        assert "terminate_sb-synthetic-1" in backend.events
        assert "poll_sb-synthetic-1" in backend.events
        assert backend.events[-2:] == ["volume_exit", "app_exit"]
        assert not any(x == "create" for x in backend.events[backend.events.index("create") + 1 :])

    asyncio.run(exercise())


def test_primary_error_survives_failed_volume_and_app_exit(tmp_path: Path) -> None:
    class BrokenContext(Context):
        async def __aexit__(self, *_args: object) -> None:
            self.backend.events.append(self.name + "_exit_failed")
            raise OSError("synthetic cleanup failed")

    class BrokenBackend(FakeBackend):
        def app_context(self) -> Context:
            return BrokenContext(self, "app")

        def volume_context(self) -> Context:
            return BrokenContext(self, "volume")

        async def stdout_chunks(self, sandbox: Sandbox) -> Iterator:
            if sandbox.object_id.endswith("1"):
                iterator = Iterator([b"WRONG\n"])
                self.iterators.append(iterator)
                return iterator
            return await super().stdout_chunks(sandbox)

    async def exercise() -> None:
        backend = BrokenBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "ValueError"
        assert result["status"] == "BLOCKED"
        assert any(x.startswith("volume_context_exit") for x in result["cleanup_errors"])
        assert any(x.startswith("app_context_exit") for x in result["cleanup_errors"])
        assert result["app_context_closed"] is False
        assert result["volume_context_closed"] is False
        assert len(backend.sandboxes) == 1
        assert backend.events[-2:] == ["volume_exit_failed", "app_exit_failed"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "phase", ["workspace_verify", "sandbox_1_create", "volume_iterator_acquire"]
)
def test_observation_reports_current_await_separately_from_unknown_allocations(
    tmp_path: Path, phase: str
) -> None:
    class HeldBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.phase_entered = asyncio.Event()
            self.release = asyncio.Event()

        async def verify(self, expected_workspace: str) -> str:
            if phase == "workspace_verify":
                self.phase_entered.set()
                await self.release.wait()
            return await super().verify(expected_workspace)

        async def create(self, app: Any, volume: Any, script: str) -> Sandbox:
            if phase == "sandbox_1_create":
                self.phase_entered.set()
                await self.release.wait()
            return await super().create(app, volume, script)

        async def volume_chunks(self, volume: Any) -> Iterator:
            if phase == "volume_iterator_acquire":
                self.phase_entered.set()
                await self.release.wait()
            return await super().volume_chunks(volume)

    async def exercise() -> None:
        backend = HeldBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        await backend.phase_entered.wait()
        observed = await owner.observe()
        assert observed["status"] == "UNRESOLVED"
        assert phase in observed["active_operations"]
        assert observed["unknown_outcomes"] == []
        assert owner.task is not None and not owner.task.done()
        backend.release.set()
        final = await owner.terminal()
        assert final["status"] == "BLOCKED"
        assert final["active_operations"] == []
        if phase == "workspace_verify":
            assert backend.events == ["verify"]
        elif phase == "sandbox_1_create":
            assert len(backend.sandboxes) == 1
            assert "poll_sb-synthetic-1" in backend.events
        else:
            assert len(backend.sandboxes) == 2
            assert backend.events[-2:] == ["volume_exit", "app_exit"]

    asyncio.run(exercise())


def test_failed_entry_never_exits_unentered_context(tmp_path: Path) -> None:
    class RefuseExit(Context):
        async def __aenter__(self) -> Context:
            self.backend.events.append(self.name + "_enter")
            raise OSError("synthetic entry raised")

        async def __aexit__(self, *_args: object) -> None:
            pytest.fail("must not exit unentered async context")

    class BrokenBackend(FakeBackend):
        def volume_context(self) -> Context:
            return RefuseExit(self, "volume")

    async def exercise() -> None:
        backend = BrokenBackend()
        owner = _owner(tmp_path, backend)
        owner.start()
        result = await owner.terminal()
        assert result["primary_error"] == "OSError"
        assert result["status"] == "BLOCKED"
        assert backend.events == ["verify", "app_enter", "volume_enter", "app_exit"]
        assert result["volume_context_closed"] is False
        assert result["app_context_closed"] is True
        assert any(x["kind"] == "volume" for x in result["unknown_outcomes"])

    asyncio.run(exercise())


def test_terminal_journal_failure_invalidates_otherwise_good_result(tmp_path: Path) -> None:
    class BrokenJournal(Ledger):
        def record(self, **fields: object) -> None:
            if fields.get("operation") == "TERMINAL":
                raise OSError("synthetic terminal fsync failed")
            super().record(**fields)

    async def exercise() -> None:
        backend = FakeBackend()
        owner = QualificationOwner(
            backend,
            Intent(
                "explicit-workspace", "qualification", "a" * 40, "modal-cpu-synthetic-persistence"
            ),
            BrokenJournal(tmp_path / "private" / "ledger.jsonl"),
        )
        owner.start()
        result = await owner.terminal()
        assert result["exact_bytes"] is True
        assert result["status"] == "BLOCKED"
        assert any(x.startswith("TERMINAL") for x in result["evidence_errors"])

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "target", ["app_context_enter", "volume_context_enter", "sandbox_1_create"]
)
def test_pre_entry_operation_begin_journal_failure_blocks_allocation(
    tmp_path: Path, target: str
) -> None:
    class Journal(Ledger):
        def record(self, **fields: object) -> None:
            if fields.get("operation") == "OPERATION_BEGIN" and fields.get("step") == target:
                raise OSError("synthetic write failed before provider entry")
            super().record(**fields)

    async def exercise() -> None:
        backend = FakeBackend()
        owner = QualificationOwner(
            backend,
            Intent(
                "explicit-workspace", "qualification", "a" * 40, "modal-cpu-synthetic-persistence"
            ),
            Journal(tmp_path / "private" / "ledger.jsonl"),
        )
        owner.start()
        result = await owner.terminal()
        assert result["status"] == "BLOCKED"
        assert result["evidence_errors"]
        assert result["primary_error"] == "OSError"
        if target == "app_context_enter":
            assert backend.events == ["verify"]
        elif target == "volume_context_enter":
            assert backend.events == ["verify", "app_enter", "app_exit"]
        else:
            assert "create" not in backend.events
            assert backend.events[-2:] == ["volume_exit", "app_exit"]

    asyncio.run(exercise())
