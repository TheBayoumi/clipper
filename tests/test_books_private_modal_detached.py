"""Offline detached-launcher tests; never authenticate or allocate a resource."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from test_books_private_phase import _brief

from clipper.books import private_modal, private_phase
from clipper.books import private_modal_detached as det
from clipper.stage_contracts import content_fingerprint

APP = "ap-" + "a" * 22
CALL = "fc-" + "B" * 24


class FakeVolume:
    def __init__(self, store: dict[str, dict[str, bytes]], name: str) -> None:
        self.name = name
        self.files = store[name]

    @contextmanager
    def batch_upload(self):
        files = self.files

        class Up:
            def put_file(self, handle, path):
                files[path.lstrip("/")] = handle.read()

        yield Up()

    def read_file(self, path):
        data = self.files[path]
        yield data[:5]
        yield data[5:]


class FakeObjects:
    def __init__(self, store: dict[str, dict[str, bytes]]) -> None:
        self.store = store
        self.deleted: list[str] = []
        self.fail_delete = False

    def create(self, name, *, allow_existing, environment_name, client):
        assert allow_existing is False and environment_name == "main"
        if name in self.store:
            raise RuntimeError("exists")
        self.store[name] = {}

    def delete(self, name, *, allow_missing, environment_name, client):
        self.deleted.append(name)
        if not self.fail_delete:
            self.store.pop(name, None)

    def list(self, *, environment_name, client):
        return [SimpleNamespace(name=n) for n in self.store]


class FakeModal:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, bytes]] = {}
        self.Volume = Mock()
        self.objects = FakeObjects(self.store)
        self.Volume.objects = self.objects
        self.Volume.from_name.side_effect = lambda n, **_: FakeVolume(self.store, n)
        self.FunctionCall = Mock()


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, spawn_ok: bool = True):
    tmp_path.chmod(0o700)
    manifest, packet = _brief()
    identity = content_fingerprint(manifest)
    client = MagicMock()
    modal = FakeModal()
    app = Mock(app_id=APP)
    function = Mock()
    function.spawn.return_value = Mock(object_id=CALL if spawn_ok else "bad")
    factory_calls: list[dict] = []

    @contextmanager
    def scope(**kw):
        scope.kw = kw  # type: ignore[attr-defined]
        yield

    app.run.side_effect = scope
    monkeypatch.setattr(private_modal, "identify", lambda: (modal, client))
    monkeypatch.setattr(private_modal, "verify_checkout", lambda *_: None)

    def factory(_modal, _run, **kw):
        factory_calls.append(kw)
        return app, function

    monkeypatch.setattr(private_modal, "factory", factory)
    cleanup = Mock(return_value=True)
    monkeypatch.setattr(private_modal, "cleanup", cleanup)
    return {
        "manifest": manifest,
        "packet": packet,
        "identity": identity,
        "client": client,
        "modal": modal,
        "function": function,
        "factory_calls": factory_calls,
        "cleanup": cleanup,
        "scope": scope,
    }


def _start(tmp_path, ctx, **kw):
    return det.start(
        ctx["manifest"],
        ctx["packet"],
        None,
        approved_manifest_sha256=ctx["identity"],
        execute=True,
        checkout=tmp_path,
        journal_root=tmp_path,
        **kw,
    )


def test_start_uploads_packet_spawns_detached_and_journals_ids(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    result = _start(tmp_path, ctx, now=lambda: 1000.0)
    name = private_modal.volume_name(ctx["manifest"])
    assert name.startswith("clipper-private-") and len(name) <= 64
    assert result["status"] == "DETACHED_RUNNING"
    assert result["call_id"] == CALL and result["app_id"] == APP
    assert ctx["modal"].store[name] == {"packet.bin": ctx["packet"]}
    assert ctx["factory_calls"][0]["restrict_access"] is det.RESTRICT_ACCESS
    assert ctx["factory_calls"][0]["volume"] is not None
    assert ctx["scope"].kw["detach"] is True
    ctx["function"].spawn.assert_called_once_with(ctx["manifest"], name)
    ctx["cleanup"].assert_not_called()
    assert ctx["modal"].objects.deleted == []
    attach = json.loads(next(tmp_path.glob("*/ATTACH.json")).read_bytes())
    assert attach["call_id"] == CALL and attach["volume"] == name
    ctx["client"].__exit__.assert_called_once_with(None, None, None)
    with pytest.raises(FileExistsError):
        _start(tmp_path, ctx)


def test_start_bad_call_id_cleans_app_and_deletes_volume(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch, spawn_ok=False)
    result = _start(tmp_path, ctx)
    assert result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    assert result["call_id"] is None
    ctx["cleanup"].assert_called_once()
    assert ctx["modal"].store == {}
    assert ctx["modal"].objects.deleted == [private_modal.volume_name(ctx["manifest"])]


def test_preflight_only_does_not_authenticate(tmp_path, monkeypatch):
    manifest, packet = _brief()
    monkeypatch.setattr(private_modal, "identify", Mock(side_effect=AssertionError))
    result = det.start(
        manifest,
        packet,
        None,
        approved_manifest_sha256=content_fingerprint(manifest),
        execute=False,
        checkout=tmp_path,
        journal_root=tmp_path,
    )
    assert result["status"] == "PREFLIGHT_ONLY"


def _poll(tmp_path, ctx, monkeypatch, get, now=2000.0, identity=None):
    ctx["modal"].FunctionCall.from_id.return_value = Mock(get=get)
    out = tmp_path / "out" / "result.bin"
    out.parent.mkdir(mode=0o700, exist_ok=True)
    result = det.poll(
        ctx["manifest"],
        approved_manifest_sha256=identity or ctx["identity"],
        journal_root=tmp_path,
        output=out,
        now=lambda: now,
    )
    return result, out


def test_poll_pending_keeps_app_and_volume(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    result, out = _poll(tmp_path, ctx, monkeypatch, Mock(side_effect=TimeoutError()))
    assert result["status"] == "DETACHED_PENDING"
    ctx["cleanup"].assert_not_called()
    assert ctx["modal"].objects.deleted == []
    assert len(ctx["modal"].store) == 1
    assert not out.exists()
    assert ctx["modal"].FunctionCall.from_id.call_args.args[0] == CALL


def test_poll_past_deadline_cleans_app_and_deletes_volume(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    late = 1000.0 + ctx["manifest"]["timeout_seconds"] + det.DEADLINE_MARGIN_SECONDS + 1
    result, out = _poll(tmp_path, ctx, monkeypatch, Mock(side_effect=TimeoutError()), now=late)
    assert result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    ctx["cleanup"].assert_called_once()
    assert ctx["modal"].store == {}
    assert not out.exists()


def _report(ctx, raw):
    return {
        "status": "BRIEF_READY_UNREVIEWED",
        "manifest_sha256": ctx["identity"],
        "owner": ctx["manifest"]["owner"],
        "checkpoint_bytes": len(raw),
        "checkpoint_sha256": private_phase.sha(raw),
        "retention": "COMPLETE",
        "brief_identity": "a" * 64,
        "configured_execution_receipt_identity": "b" * 64,
        "publish_eligible": False,
    }


def _raw():
    return private_phase.pack(
        {"book-candidates/hybrid_argument/" + "a" * 64 + "/raw-1.json": b"fixture"}
    )


def test_poll_success_reads_volume_persists_and_deletes(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    raw = _raw()
    name = private_modal.volume_name(ctx["manifest"])
    ctx["modal"].store[name]["checkpoint.bin"] = private_modal.compress_checkpoint(raw)
    get = Mock(return_value=(_report(ctx, raw), None))
    result, out = _poll(tmp_path, ctx, monkeypatch, get)
    assert result["status"] == "BRIEF_READY_UNREVIEWED"
    assert result["checkpoint_saved"] is True and out.read_bytes() == raw
    ctx["cleanup"].assert_called_once()
    assert ctx["modal"].store == {}
    events = [json.loads(p.read_bytes())["event"] for p in tmp_path.glob("*/0*.json")]
    assert "VOLUME_DELETE_VERIFIED" in events and "DETACHED_FINISHED" in events
    again, _ = _poll(tmp_path, ctx, monkeypatch, get)
    assert again["status"] == "ALREADY_FINISHED"


def test_poll_corrupt_volume_checkpoint_rejected_and_volume_deleted(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    raw = _raw()
    name = private_modal.volume_name(ctx["manifest"])
    other = private_modal.compress_checkpoint(
        private_phase.pack(
            {"book-candidates/hybrid_argument/" + "c" * 64 + "/raw-1.json": b"different"}
        )
    )
    ctx["modal"].store[name]["checkpoint.bin"] = other
    result, out = _poll(tmp_path, ctx, monkeypatch, Mock(return_value=(_report(ctx, raw), None)))
    assert result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    assert not out.exists()
    assert ctx["modal"].store == {}


def test_unverified_volume_deletion_blocks_and_stays_retryable(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    raw = _raw()
    name = private_modal.volume_name(ctx["manifest"])
    ctx["modal"].store[name]["checkpoint.bin"] = private_modal.compress_checkpoint(raw)
    ctx["modal"].objects.fail_delete = True
    get = Mock(return_value=(_report(ctx, raw), None))
    result, _ = _poll(tmp_path, ctx, monkeypatch, get)
    assert result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    assert result["cleanup_verified"] is False
    events = [json.loads(p.read_bytes())["event"] for p in tmp_path.glob("*/0*.json")]
    assert "DETACHED_FINISHED" not in events


def test_poll_owner_mismatch_rejected(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    report = {
        "status": "BRIEF_READY_UNREVIEWED",
        "manifest_sha256": ctx["identity"],
        "owner": "c" * 64,
        "publish_eligible": False,
    }
    result, out = _poll(tmp_path, ctx, monkeypatch, Mock(return_value=(report, None)))
    assert result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    assert not out.exists()
    assert ctx["modal"].store == {}


def test_poll_requires_matching_attach(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx)
    with pytest.raises(private_phase.PrivatePhaseError):
        _poll(tmp_path, ctx, monkeypatch, Mock(), identity="d" * 64)


def test_remote_volume_entry_commits_checkpoint(tmp_path, monkeypatch):
    manifest, packet = _brief()
    raw = _raw()
    monkeypatch.setattr(private_modal, "VOLUME_MOUNT", str(tmp_path))
    (tmp_path / "packet.bin").write_bytes(packet)
    monkeypatch.setattr(
        private_modal, "_execute_in_container", lambda m, p, c: ({"status": "x"}, raw)
    )
    commit = Mock()
    fake = Mock()
    fake.Volume.from_name.return_value = Mock(commit=commit)
    monkeypatch.setitem(__import__("sys").modules, "modal", fake)
    report, none = private_modal._remote_phase_volume(manifest, "clipper-private-x")
    assert none is None and report == {"status": "x"}
    assert private_modal.decompress_checkpoint((tmp_path / "checkpoint.bin").read_bytes()) == raw
    commit.assert_called_once()
    assert not (tmp_path / "checkpoint.bin.tmp").exists()


def test_factory_mounts_volume_with_network_blocked_and_no_secrets():
    modal = Mock()
    app = modal.App.return_value
    private_modal.factory(modal, _brief()[0], restrict_access=True, volume="V")
    kw = app.function.call_args.kwargs
    assert kw["volumes"] == {"/vol": "V"}
    assert kw["block_network"] is True and kw["restrict_modal_access"] is True
    assert "secrets" not in kw and kw["retries"] == 0


def test_remote_volume_entry_survives_blocked_commit(tmp_path, monkeypatch):
    manifest, packet = _brief()
    raw = _raw()
    monkeypatch.setattr(private_modal, "VOLUME_MOUNT", str(tmp_path))
    (tmp_path / "packet.bin").write_bytes(packet)
    monkeypatch.setattr(
        private_modal, "_execute_in_container", lambda m, p, c: ({"status": "x"}, raw)
    )
    fake = Mock()
    fake.Volume.from_name.return_value = Mock(commit=Mock(side_effect=RuntimeError("401")))
    monkeypatch.setitem(__import__("sys").modules, "modal", fake)
    report, _ = private_modal._remote_phase_volume(manifest, "clipper-private-x")
    assert report == {"status": "x"} and (tmp_path / "checkpoint.bin").is_file()


def _l4_manifest():
    manifest, _ = _brief()
    manifest["accelerator"] = "L4"
    manifest["resource_approval"] = "EXPLICIT_PRIVATE_L4_PHASE"
    return manifest


def test_manifest_accelerator_and_approval_must_agree():
    assert private_phase.validate_manifest(_l4_manifest())["accelerator"] == "L4"
    for accelerator, approval in (
        ("L4", "EXPLICIT_PRIVATE_CPU_PHASE"),
        ("none", "EXPLICIT_PRIVATE_L4_PHASE"),
        ("A100", "EXPLICIT_PRIVATE_L4_PHASE"),
    ):
        bad = _l4_manifest()
        bad["accelerator"], bad["resource_approval"] = accelerator, approval
        with pytest.raises(private_phase.PrivatePhaseError):
            private_phase.validate_manifest(bad)
    missing = _l4_manifest()
    del missing["accelerator"]
    with pytest.raises(private_phase.PrivatePhaseError):
        private_phase.validate_manifest(missing)


@pytest.mark.parametrize("l4", [True, False])
def test_factory_selects_l4_and_torch_wheel(l4):
    modal = Mock()
    chain = modal.Image.debian_slim.return_value
    for name in ("apt_install", "run_commands", "pip_install", "env"):
        getattr(chain, name).return_value = chain
    manifest = _l4_manifest() if l4 else _brief()[0]
    private_modal.factory(modal, manifest, volume="V")
    kw = modal.App.return_value.function.call_args.kwargs
    assert kw["gpu"] == ("L4" if l4 else None)
    assert kw["block_network"] is True and kw["retries"] == 0
    joined = " ".join(str(c) for c in chain.run_commands.call_args_list)
    assert ("torch==2.6.0+cpu" in joined) is not l4


@pytest.mark.parametrize("l4", [True, False])
def test_container_sets_qwen_device_from_manifest(l4, monkeypatch):
    import os
    import subprocess

    manifest = _l4_manifest() if l4 else _brief()[0]
    manifest["commit"] = "f" * 40
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: Mock(stdout="f" * 40, returncode=0),
    )
    from clipper.books import private_phase as phase

    monkeypatch.setattr(phase, "execute_private_phase", lambda *a, **k: ({"s": 1}, None))
    monkeypatch.setenv("CLIPPER_QWEN_DEVICE", "unset")
    private_modal._execute_in_container(manifest, b"", None)
    assert os.environ["CLIPPER_QWEN_DEVICE"] == ("cuda" if l4 else "cpu")
