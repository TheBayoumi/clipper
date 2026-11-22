"""Synthetic owner-fenced terminal publication: no Modal execution or private data."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from clipper.dag import DagLeaseCoordinator, DagStore, StageClaim, StageRecord
from clipper.stage_contracts import StageContract, stage_identity


def _identity():
    return stage_identity(StageContract("shared.fenced", {"version": 1}), source_hash="synthetic")


class FakeAuthority:
    """In-memory serialized lease authority; worker mount is not its durable view."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.owner = ""
        self.attempt = 0
        self.expires = 0.0
        self.now = 10.0
        self.durable: dict[str, Any] | None = None
        self.claim_calls = 0
        self.publish_calls = 0
        self.operations = 0
        self.ack_failure: BaseException | None = None
        self.persist_on_failure = False
        self.publish_entered: threading.Event | None = None
        self.publish_continue: threading.Event | None = None
        self.owner_override = ""
        self.renew_failure = False

    def claim(self, _identity, owner: str, ttl: float) -> StageClaim:
        with self.lock:
            self.claim_calls += 1
            if self.durable and self.durable["record"]["status"] == "PASS":
                return StageClaim(False, self.attempt)
            if self.owner and self.now < self.expires and self.owner != owner:
                return StageClaim(False, self.attempt)
            self.owner = owner
            self.attempt += 1
            self.expires = self.now + ttl
            return StageClaim(True, self.attempt)

    def renew(self, _identity, owner: str, ttl: float) -> bool:
        with self.lock:
            if self.renew_failure or owner != self.owner or self.now >= self.expires:
                return False
            self.expires = self.now + ttl
            return True

    def release(self, _identity, owner: str) -> bool:
        with self.lock:
            if owner != self.owner:
                return False
            self.owner = ""
            return True

    def read(self, _identity) -> dict[str, Any] | None:
        with self.lock:
            return json.loads(json.dumps(self.durable)) if self.durable is not None else None

    def publish(self, _identity, owner: str, record: StageRecord, output: object) -> bool:
        if self.publish_entered is not None:
            self.publish_entered.set()
            assert self.publish_continue is not None
            assert self.publish_continue.wait(3)
        with self.lock:
            self.publish_calls += 1
            if self.owner_override:
                self.owner = self.owner_override
            if owner != self.owner or self.now >= self.expires:
                return False
            if record.attempt_count != self.attempt:
                return False
            if self.durable and self.durable["record"]["status"] == "PASS":
                return False
            if self.ack_failure is None or self.persist_on_failure:
                self.durable = {"record": record.to_dict(), "output": output}
            if self.ack_failure is not None:
                raise self.ack_failure
            return True

    def store(self, root: Path) -> DagStore:
        return DagStore(
            root,
            coordinator=DagLeaseCoordinator(
                claim=self.claim,
                renew=self.renew,
                release=self.release,
                commit=lambda: None,
                reload=lambda: None,
                read_committed=self.read,
                publish_terminal=self.publish,
            ),
            execution_lease_seconds=120.0,
            follower_poll_seconds=0.001,
        )


@pytest.mark.parametrize("coordinated", [False, True])
@pytest.mark.parametrize("value", [None, False, 0, {}, [], ""])
def test_null_and_falsey_output_is_cached_as_a_hit(
    tmp_path: Path, coordinated: bool, value: object
) -> None:
    authority = FakeAuthority()
    store = authority.store(tmp_path) if coordinated else DagStore(tmp_path)
    identity = _identity()
    operations: list[int] = []

    def operation() -> object:
        operations.append(1)
        return value

    assert store.execute(identity, operation) == (value, False)
    assert store.execute(identity, operation) == (value, True)
    assert operations == [1]
    if coordinated:
        assert authority.store(tmp_path).execute(identity, operation) == (value, True)
        assert operations == [1]


@pytest.mark.parametrize("termination_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("persisted", [False, True])
def test_lost_authority_acknowledgement_preserves_durable_only(
    tmp_path: Path, termination_type: type[BaseException], persisted: bool
) -> None:
    authority = FakeAuthority()
    store = authority.store(tmp_path)
    identity = _identity()
    primary = termination_type("synthetic lost acknowledgement")
    authority.ack_failure = primary
    authority.persist_on_failure = persisted
    operations: list[int] = []

    def operation() -> object:
        operations.append(1)
        return {"complete": True}

    with pytest.raises(termination_type) as error:
        store.execute(identity, operation)
    assert error.value is primary
    assert authority.publish_calls == 1
    assert authority.owner == ""
    assert store.cached_output(identity) == ({"complete": True} if persisted else None)
    authority.ack_failure = None
    assert authority.store(tmp_path).execute(identity, operation) == (
        {"complete": True},
        persisted,
    )
    assert operations == ([1] if persisted else [1, 1])


def test_owner_fence_blocks_reassigned_or_expired_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = FakeAuthority()
    identity = _identity()
    store = authority.store(tmp_path)
    old_owner = ""
    published: list[int] = []

    def operation() -> object:
        nonlocal old_owner
        old_owner = authority.owner
        authority.owner_override = "new-owner"
        published.append(1)
        return {"stale": True}

    with (
        patch("clipper.dag.time.sleep"),
        pytest.raises(RuntimeError, match="owner-fenced PASS publication rejected"),
    ):
        store.execute(identity, operation)
    assert published == [1]
    assert old_owner and authority.owner == "new-owner"
    assert authority.durable is None
    authority.owner_override = ""
    authority.owner = ""
    assert store.execute(identity, lambda: {"fresh": True}) == ({"fresh": True}, False)
    assert store.execute(identity, lambda: 0) == ({"fresh": True}, True)


def test_follower_waits_for_authority_not_dirty_worker_mount(tmp_path: Path) -> None:
    identity = _identity()
    authority = FakeAuthority()
    entered = threading.Event()
    proceed = threading.Event()
    authority.publish_entered = entered
    authority.publish_continue = proceed
    owner = authority.store(tmp_path)
    follower = authority.store(tmp_path)
    calls = {"owner": 0, "follower": 0}

    def run_owner() -> object:
        calls["owner"] += 1
        return {"published": True}

    def run_follower() -> object:
        calls["follower"] += 1
        raise AssertionError("follower must reuse")

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(owner.execute, identity, run_owner)
        assert entered.wait(3)
        waiting = pool.submit(follower.execute, identity, run_follower)
        # The owner is still publishing, so the committed read must miss.
        assert authority.read(identity) is None
        proceed.set()
        assert pending.result(timeout=5) == ({"published": True}, False)
        assert waiting.result(timeout=5) == ({"published": True}, True)
    assert calls == {"owner": 1, "follower": 0}


def test_authority_unavailable_read_stops_without_duplicate_operation(tmp_path: Path) -> None:
    authority = FakeAuthority()
    store = authority.store(tmp_path)
    original = authority.read

    def unavailable(_identity):
        raise OSError("committed storage unavailable")

    authority.read = unavailable
    with pytest.raises(OSError, match="committed storage unavailable"):
        authority.store(tmp_path).execute(_identity(), lambda: pytest.fail("operation ran"))
    authority.read = original
    assert store.execute(_identity(), lambda: 5) == (5, False)


class FakeVolume:
    """Separate dirty mount and durable bytes; reload can implicitly commit."""

    def __init__(self, root: Path, *, implicit: bool) -> None:
        self.root = root
        self.implicit = implicit
        self.durable: dict[str, bytes] = {}
        self.commit_calls = 0
        self.reload_calls = 0

    def _files(self) -> dict[str, bytes]:
        return {
            p.relative_to(self.root).as_posix(): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file()
        }

    def commit(self) -> None:
        self.commit_calls += 1
        self.durable.update(self._files())

    def reload(self) -> None:
        self.reload_calls += 1
        if self.implicit:
            self.commit()
        for p in tuple(self.root.rglob("*")):
            if p.is_file():
                p.unlink()
        for name, data in self.durable.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def read_file(self, path: str):
        if path not in self.durable:
            raise FileNotFoundError(path)
        yield self.durable[path]


def _load_endpoint(root: Path, *, implicit: bool):
    """Compile only actual script functions, without importing or calling Modal."""
    source = (Path(__file__).resolve().parents[1] / "scripts/modal_pipeline.py").read_text()
    module = ast.parse(source)
    names = {
        "_atomic_write_json",
        "_durable_json",
        "_dag_terminal_path",
        "_dag_legacy_terminal",
        "_dag_committed_terminal",
        "dag_execution_lease",
    }
    functions = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    assert {node.name for node in functions} == names
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *functions,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(tree)
    volume = FakeVolume(root, implicit=implicit)
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "math": __import__("math"),
        "os": os,
        "uuid": __import__("uuid"),
        "Path": Path,
        "time": time,
        "modal": SimpleNamespace(exception=SimpleNamespace(NotFoundError=FileNotFoundError)),
        "ARTIFACT_ROOT": str(root),
        "artifact_volume": volume,
        "_TERMINAL_SCHEMA": "clipper-dag-authoritative-terminal-v1",
        "_assert_expected_git_sha": lambda payload: None,
    }
    exec(  # noqa: S102 - only static, selected repo AST functions; no user-supplied code
        compile(tree, "scripts/modal_pipeline.py", "exec"), namespace
    )
    return namespace["dag_execution_lease"], volume


def _request(identity, action: str, owner: str = "", **extras):
    return {
        "action": action,
        "stage_name": identity.stage_name,
        "cache_key": identity.cache_key,
        "owner_id": owner,
        "ttl_seconds": 120.0,
        "dag_root": "legacy",
        **extras,
    }


@pytest.mark.parametrize("implicit", [False, True])
def test_endpoint_committed_read_does_not_flush_dirty_mount(
    tmp_path: Path, implicit: bool
) -> None:
    call, volume = _load_endpoint(tmp_path, implicit=implicit)
    identity = _identity()
    dirty = tmp_path / "_dag-authority" / (
        hashlib.sha256(f"{identity.stage_name}\0{identity.cache_key}".encode()).hexdigest()
        + ".json"
    )
    dirty.parent.mkdir()
    dirty.write_text('{"schema":"clipper-dag-authoritative-terminal-v1","record":{},"output":5}')
    assert call(_request(identity, "read")) == {"terminal": None}
    assert volume.commit_calls == 0
    assert volume.reload_calls == 0
    # Reload is *not* a passive read on an implicit-persistence backend.
    volume.reload()
    assert volume.commit_calls == (1 if implicit else 0)


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("terminal_type", ["PASS", "FAILED"])
def test_endpoint_fences_terminal_publication_against_owner_and_expiry(
    tmp_path: Path, implicit: bool, terminal_type: str
) -> None:
    call, volume = _load_endpoint(tmp_path, implicit=implicit)
    identity = _identity()
    owner = "a" * 32
    claim = call(_request(identity, "claim", owner))
    assert claim["acquired"] is True
    attempt = claim["attempt_count"]
    output = None if terminal_type == "FAILED" else {"complete": True}
    record = StageRecord(
        identity=identity,
        status=terminal_type,  # type: ignore[arg-type]
        attempt_count=attempt,
        started_at="now",
        completed_at="done",
        output_fingerprint=hashlib.sha256(
            json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() if terminal_type == "PASS" else None,
        usage={},
        cost_usd=0.0,
        error_type="TestFailure" if terminal_type == "FAILED" else None,
        error="synthetic failure" if terminal_type == "FAILED" else None,
    )
    assert call(_request(identity, "publish", "b" * 32, record=record.to_dict(), output=output))[
        "published"
    ] is False
    assert call(_request(identity, "publish", owner, record=record.to_dict(), output=output))[
        "published"
    ] is True
    committed = call(_request(identity, "read"))["terminal"]
    assert committed["record"] == record.to_dict()
    assert committed["output"] == output
    assert call(_request(identity, "release", "b" * 32))["released"] is False
    assert call(_request(identity, "release", owner))["released"] is True
    if terminal_type == "PASS":
        assert call(_request(identity, "claim", "c" * 32))["acquired"] is False
    else:
        assert call(_request(identity, "claim", "c" * 32))["attempt_count"] == attempt + 1
    assert volume.commit_calls >= 3


@pytest.mark.parametrize("implicit", [False, True])
def test_endpoint_legacy_committed_pass_reuses_without_mutation(
    tmp_path: Path, implicit: bool
) -> None:
    call, volume = _load_endpoint(tmp_path, implicit=implicit)
    identity = _identity()
    store = DagStore(tmp_path / "legacy")
    assert store.execute(identity, lambda: None) == (None, False)
    original_files = volume._files()
    volume.commit()
    snapshot = call(_request(identity, "read"))["terminal"]
    assert snapshot is not None and snapshot["output"] is None
    assert snapshot["record"]["status"] == "PASS"
    assert call(_request(identity, "claim", "a" * 32))["acquired"] is False
    assert all(volume.durable[k] == data for k, data in original_files.items())
