"""Offline Modal qualification: fake backend only, no SDK import or provider calls."""
from __future__ import annotations

import asyncio
import hashlib
import json
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
