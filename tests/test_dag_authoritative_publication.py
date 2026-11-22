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
from clipper.stage_contracts import StageContract, content_fingerprint, stage_identity


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
        "_dag_canonical_identity_matches",
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
def test_endpoint_committed_read_does_not_flush_dirty_mount(tmp_path: Path, implicit: bool) -> None:
    call, volume = _load_endpoint(tmp_path, implicit=implicit)
    identity = _identity()
    dirty = (
        tmp_path
        / "_dag-authority"
        / (
            hashlib.sha256(f"{identity.stage_name}\0{identity.cache_key}".encode()).hexdigest()
            + ".json"
        )
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
        ).hexdigest()
        if terminal_type == "PASS"
        else None,
        usage={},
        cost_usd=0.0,
        error_type="TestFailure" if terminal_type == "FAILED" else None,
        error="synthetic failure" if terminal_type == "FAILED" else None,
    )
    assert (
        call(_request(identity, "publish", "b" * 32, record=record.to_dict(), output=output))[
            "published"
        ]
        is False
    )
    assert (
        call(_request(identity, "publish", owner, record=record.to_dict(), output=output))[
            "published"
        ]
        is True
    )
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


def test_authoritative_primary_operation_failure_records_failed_then_retries(
    tmp_path: Path,
) -> None:
    authority = FakeAuthority()
    identity = _identity()
    store = authority.store(tmp_path)
    primary = ValueError("synthetic primary operation failure")
    operations: list[int] = []

    def fail() -> object:
        operations.append(1)
        raise primary

    with pytest.raises(ValueError) as caught:
        store.execute(identity, fail)
    assert caught.value is primary
    assert authority.durable is not None
    assert authority.durable["record"]["status"] == "FAILED"
    assert authority.durable["record"]["error_type"] == "ValueError"
    assert authority.publish_calls == 1
    assert authority.owner == ""
    assert store.cached_output(identity) is None
    assert store.execute(identity, lambda: operations.append(2) or {"ready": True}) == (
        {"ready": True},
        False,
    )
    assert operations == [1, 2]
    assert authority.durable["record"]["attempt_count"] == 2


@pytest.mark.parametrize("mode", ["ack-lost", "invalid-receipt", "zero-attempt", "bad-read"])
def test_authoritative_claim_boundary_cleanup_preserves_errors(tmp_path: Path, mode: str) -> None:
    authority = FakeAuthority()
    identity = _identity()
    primary = KeyboardInterrupt("synthetic claim acknowledgement lost")
    original_claim = authority.claim
    original_read = authority.read
    operations: list[int] = []

    def claim(_identity, owner: str, ttl: float):
        receipt = original_claim(_identity, owner, ttl)
        if mode == "ack-lost":
            raise primary
        if mode == "invalid-receipt":
            return True
        if mode == "zero-attempt":
            return StageClaim(True, 0)
        return receipt

    def read(_identity):
        if mode == "bad-read" and authority.claim_calls > 0:
            raise OSError("synthetic post-claim read failure")
        return original_read(_identity)

    authority.claim = claim
    authority.read = read
    store = authority.store(tmp_path)
    if mode == "ack-lost":
        with pytest.raises(KeyboardInterrupt) as error:
            store.execute(identity, lambda: operations.append(1))
        assert error.value is primary
    else:
        with pytest.raises((RuntimeError, OSError)):
            store.execute(identity, lambda: operations.append(1))
    assert operations == []
    assert authority.owner == ""
    authority.claim = original_claim
    authority.read = original_read
    assert authority.store(tmp_path).execute(identity, lambda: {"success": True}) == (
        {"success": True},
        False,
    )


def test_authoritative_recheck_catches_pass_published_during_claim(tmp_path: Path) -> None:
    authority = FakeAuthority()
    identity = _identity()
    output = {"winner": True}
    original_claim = authority.claim
    operations: list[int] = []

    def claim(_identity, owner: str, ttl: float) -> StageClaim:
        receipt = original_claim(_identity, owner, ttl)
        record = StageRecord(
            identity=identity,
            status="PASS",
            attempt_count=receipt.attempt_count,
            started_at="before",
            completed_at="after",
            output_fingerprint=content_fingerprint(output),
            usage={},
            cost_usd=0.0,
        )
        authority.durable = {"record": record.to_dict(), "output": output}
        return receipt

    authority.claim = claim
    store = authority.store(tmp_path)
    assert store.execute(identity, lambda: operations.append(1)) == (output, True)
    assert not operations
    assert authority.owner == ""


@pytest.mark.parametrize("malformed", ["envelope", "record", "identity", "fingerprint"])
def test_authoritative_bad_committed_evidence_stops_without_operation(
    tmp_path: Path, malformed: str
) -> None:
    authority = FakeAuthority()
    identity = _identity()
    output = {"value": 1}
    record = StageRecord(
        identity=identity,
        status="PASS",
        attempt_count=1,
        started_at="start",
        completed_at="end",
        output_fingerprint=content_fingerprint(output),
        usage={},
        cost_usd=0.0,
    ).to_dict()
    if malformed == "envelope":
        snapshot: dict[str, Any] = {"record": record}
    elif malformed == "record":
        snapshot = {"record": None, "output": output}
    else:
        if malformed == "identity":
            record["identity"] = {"source_hash": "foreign"}
        else:
            record["output_fingerprint"] = "0" * 64
        snapshot = {"record": record, "output": output}
    authority.durable = snapshot
    with pytest.raises(RuntimeError, match="DAG committed"):
        authority.store(tmp_path).execute(identity, lambda: pytest.fail("operation must not run"))


@pytest.mark.parametrize("termination", ["false", "error", "cancel"])
def test_authoritative_background_lease_loss_or_termination_is_not_success(
    tmp_path: Path, termination: str
) -> None:
    authority = FakeAuthority()
    original_renew = authority.renew
    executing = threading.current_thread()
    worker_called = threading.Event()
    primary = KeyboardInterrupt("synthetic renewal termination")

    def renew(_identity, owner: str, ttl: float) -> bool:
        if threading.current_thread() is not executing:
            worker_called.set()
            if termination == "error":
                raise OSError("synthetic transport error")
            if termination == "cancel":
                raise primary
            return False
        return original_renew(_identity, owner, ttl)

    authority.renew = renew
    store = authority.store(tmp_path)
    store.execution_lease_seconds = 1.5
    identity = _identity()

    def operation() -> object:
        assert worker_called.wait(3), "background renewal was never observed"
        return {"not_committed": True}

    expected = KeyboardInterrupt if termination == "cancel" else RuntimeError
    with pytest.raises(expected) as caught:
        store.execute(identity, operation)
    if termination == "cancel":
        assert caught.value is primary
    assert authority.owner == ""
    if termination == "cancel":
        assert authority.durable is not None
        assert authority.durable["record"]["status"] == "FAILED"
        assert authority.publish_calls == 1
    else:
        assert authority.durable is None
        assert authority.publish_calls == 0


# Exact JSON identity semantics: ordinary dict equality aliases bool/int/float.
_IDENTITY_TYPE_CASES = [
    pytest.param(("flag",), 0, id="false-vs-zero"),
    pytest.param(("enabled",), 1, id="true-vs-one"),
    pytest.param(("temperature",), 0.0, id="integer-vs-float"),
    pytest.param(("nested", "flag"), 0, id="nested-false-vs-zero"),
    pytest.param(("nested", "sequence", 0), 1, id="nested-list-true-vs-one"),
    pytest.param(("nested", "sequence", 1), 1.0, id="nested-list-integer-vs-float"),
]


def _typed_identity():
    return stage_identity(
        StageContract("identity-type-probe", {"v": 1}),
        source_hash="synthetic",
        decoding_parameters={
            "flag": False,
            "enabled": True,
            "temperature": 0,
            "nested": {"flag": False, "sequence": [True, 1]},
        },
    )


def _mutate_record_identity(
    record: dict[str, Any], path: tuple[str | int, ...], replacement: object
) -> None:
    current: Any = record["identity"]["decoding_parameters"]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


@pytest.mark.parametrize(("path", "replacement"), _IDENTITY_TYPE_CASES)
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("output", [None, {"complete": True}], ids=["null", "object"])
def test_local_type_altered_identity_never_reuses_valid_pass(
    tmp_path: Path,
    private: bool,
    output: object,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    identity = _typed_identity()
    store = DagStore(
        tmp_path,
        private_stages=frozenset({identity.stage_name}) if private else frozenset(),
    )
    assert store.execute(identity, lambda: output) == (output, False)
    record_path = store._record_path(identity)
    output_path = store._output_path(identity)
    original_record = record_path.read_bytes()
    original_output = output_path.read_bytes()
    modified: dict[str, Any] = json.loads(original_record)
    _mutate_record_identity(modified, path, replacement)
    # Python dict equality is exactly the dangerous false-positive.
    assert modified["identity"] == identity.to_dict()
    assert content_fingerprint(modified["identity"]) != content_fingerprint(identity.to_dict())
    record_path.write_text(json.dumps(modified), encoding="utf-8")

    assert store._cache_hit(identity) == (False, None)
    assert store.cached_output(identity) is None
    assert output_path.read_bytes() == original_output
    if private:
        assert store._directory(identity).stat().st_mode & 0o777 == 0o700
        assert record_path.stat().st_mode & 0o777 == 0o600

    # Restore the original synthetic bytes: valid complete records still reuse
    # both ordinary and JSON-null outputs without invoking the operation.
    record_path.write_bytes(original_record)
    assert record_path.read_bytes() == original_record
    assert store.execute(identity, lambda: pytest.fail("unexpected operation rerun")) == (
        output,
        True,
    )
    assert output_path.read_bytes() == original_output


@pytest.mark.parametrize(("path", "replacement"), _IDENTITY_TYPE_CASES)
@pytest.mark.parametrize("output", [None, {"complete": True}], ids=["null", "object"])
def test_authoritative_type_altered_committed_identity_fails_closed(
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: object,
    output: object,
) -> None:
    authority = FakeAuthority()
    identity = _typed_identity()
    store = authority.store(tmp_path)
    assert store.execute(identity, lambda: output) == (output, False)
    assert authority.durable is not None
    original_bytes = json.dumps(authority.durable, sort_keys=True).encode("utf-8")
    _mutate_record_identity(authority.durable["record"], path, replacement)
    assert authority.durable["record"]["identity"] == identity.to_dict()
    assert content_fingerprint(authority.durable["record"]["identity"]) != content_fingerprint(
        identity.to_dict()
    )
    claims_before = authority.claim_calls
    with pytest.raises(RuntimeError, match="committed PASS identity or output is invalid"):
        store.cached_output(identity)
    with pytest.raises(RuntimeError, match="committed PASS identity or output is invalid"):
        authority.store(tmp_path).execute(
            identity, lambda: pytest.fail("invalid committed evidence was reused")
        )
    assert authority.claim_calls == claims_before

    authority.durable = json.loads(original_bytes)
    assert json.dumps(authority.durable, sort_keys=True).encode("utf-8") == original_bytes
    assert authority.store(tmp_path).execute(
        identity, lambda: pytest.fail("valid committed output was rerun")
    ) == (output, True)


@pytest.mark.parametrize(("path", "replacement"), _IDENTITY_TYPE_CASES)
@pytest.mark.parametrize("implicit", [False, True])
def test_legacy_committed_type_alteration_does_not_block_recovery(
    tmp_path: Path,
    implicit: bool,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    call, volume = _load_endpoint(tmp_path, implicit=implicit)
    identity = _typed_identity()
    store = DagStore(tmp_path / "legacy")
    assert store.execute(identity, lambda: None) == (None, False)
    volume.commit()
    record_path = store._record_path(identity)
    relative_record = record_path.relative_to(tmp_path).as_posix()
    relative_output = store._output_path(identity).relative_to(tmp_path).as_posix()
    original_record = volume.durable[relative_record]
    original_output = volume.durable[relative_output]
    previous_commits = volume.commit_calls
    valid = call(_request(identity, "read"))["terminal"]
    assert valid is not None and valid["output"] is None
    assert volume.commit_calls == previous_commits

    changed: dict[str, Any] = json.loads(original_record)
    _mutate_record_identity(changed, path, replacement)
    assert changed["identity"] == identity.to_dict()
    assert content_fingerprint(changed["identity"]) != content_fingerprint(identity.to_dict())
    changed_bytes = json.dumps(changed, sort_keys=True).encode("utf-8")
    volume.durable[relative_record] = changed_bytes
    record_path.write_bytes(changed_bytes)
    assert call(_request(identity, "read")) == {"terminal": None}
    # A type-altered legacy PASS must not falsely prevent the owner from claiming.
    acquired = call(_request(identity, "claim", "a" * 32))
    assert acquired["acquired"] is True
    assert volume.durable[relative_record] == changed_bytes
    assert volume.durable[relative_output] == original_output

    # No migration or rewriting of the original complete evidence is necessary.
    volume.durable[relative_record] = original_record
    record_path.write_bytes(original_record)
    assert call(_request(identity, "read"))["terminal"]["output"] is None
    assert volume.durable[relative_record] == original_record
    assert volume.durable[relative_output] == original_output


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("status", ["PASS", "FAILED"])
@pytest.mark.parametrize("output", [None, 7, False], ids=["null", "scalar", "false"])
@pytest.mark.parametrize(
    "tamper",
    [
        "source",
        "bool-int",
        "int-float",
        "nested",
        "missing",
        "extra",
        "wrong-cache-key",
        "bad-dependencies",
        "bad-decoding",
        "bad-model-revision",
    ],
)
def test_endpoint_rejects_unbound_canonical_identity_before_terminal_publication(
    tmp_path: Path,
    implicit: bool,
    status: str,
    output: object,
    tamper: str,
) -> None:
    """Exercise the actual endpoint AST; only fabricated JSON reaches fake storage."""
    call, volume = _load_endpoint(tmp_path, implicit=implicit)
    identity = _typed_identity()
    owner = "a" * 32
    receipt = call(_request(identity, "claim", owner))
    assert receipt["acquired"] is True
    record = StageRecord(
        identity=identity,
        status=status,  # type: ignore[arg-type]
        attempt_count=receipt["attempt_count"],
        started_at="synthetic-start",
        completed_at="synthetic-end",
        output_fingerprint=content_fingerprint(output) if status == "PASS" else None,
        usage={},
        cost_usd=0.0,
        error_type="SyntheticFailure" if status == "FAILED" else None,
        error="synthetic failure" if status == "FAILED" else None,
    ).to_dict()
    mutated = json.loads(json.dumps(record))
    material = mutated["identity"]
    if tamper == "source":
        material["source_hash"] = "different-source"
    elif tamper == "bool-int":
        material["decoding_parameters"]["flag"] = 0
    elif tamper == "int-float":
        material["decoding_parameters"]["temperature"] = 0.0
    elif tamper == "nested":
        material["decoding_parameters"]["nested"]["sequence"][0] = 1
    elif tamper == "missing":
        del material["model_revision"]
    elif tamper == "extra":
        material["unrecognized"] = "not-canonical"
    elif tamper == "wrong-cache-key":
        material["cache_key"] = "0" * 64
    elif tamper == "bad-dependencies":
        material["dependency_output_hashes"] = [False]
    elif tamper == "bad-decoding":
        material["decoding_parameters"] = []
    elif tamper == "bad-model-revision":
        material["model_revision"] = False
    assert mutated["identity"]["stage_name"] == identity.stage_name
    if tamper != "wrong-cache-key":
        assert mutated["identity"]["cache_key"] == identity.cache_key
    assert content_fingerprint(mutated["identity"]) != content_fingerprint(identity.to_dict())

    with pytest.raises(ValueError, match="DAG terminal publication identity mismatch"):
        call(_request(identity, "publish", owner, record=mutated, output=output))
    assert call(_request(identity, "read")) == {"terminal": None}
    assert not list((tmp_path / "_dag-authority").glob("*.json"))
    assert not any(key.startswith("_dag-authority/") for key in volume.durable)

    # The same owned attempt remains eligible: rejected bytes did not strand it.
    success = call(_request(identity, "publish", owner, record=record, output=output))
    assert success["published"] is True
    committed = call(_request(identity, "read"))["terminal"]
    assert committed == {"record": record, "output": output}
    assert call(_request(identity, "release", owner))["released"] is True
    if status == "PASS":
        assert call(_request(identity, "claim", "b" * 32))["acquired"] is False
    else:
        assert call(_request(identity, "claim", "b" * 32))["acquired"] is True
