"""Offline SDK boundaries, private CLI and image declarations, never real credentials."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_books_private_phase import _brief

from clipper.books import private_modal as host
from clipper.books import private_phase as phase
from clipper.stage_contracts import content_fingerprint


@pytest.mark.parametrize("fault", ["none", "workspace", "hydrate", "client"])
def test_identity_uses_injected_client_and_closes_failed_identity(monkeypatch, fault):
    monkeypatch.setenv("MODAL_TOKEN_ID", "offline-fixture-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "offline-fixture-secret")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    client = Mock()
    workspace = Mock(name="workspace")
    workspace.name = phase.WORKSPACE if fault != "workspace" else "wrong-workspace"
    sdk = SimpleNamespace(Client=Mock(), Workspace=Mock())
    sdk.Client.from_env.return_value = client
    sdk.Workspace.from_context.return_value = workspace
    if fault == "hydrate":
        workspace.hydrate.side_effect = RuntimeError("offline failure")
    if fault == "client":
        sdk.Client.from_env.side_effect = RuntimeError("offline failure")
    monkeypatch.setattr(host, "version", lambda _: "1.5.5")
    monkeypatch.setattr(host.importlib, "import_module", lambda _: sdk)
    if fault == "none":
        assert host.identify() == (sdk, client)
        sdk.Workspace.from_context.assert_called_once_with(client=client)
        workspace.hydrate.assert_called_once_with(client=client)
        client.close.assert_not_called()
    else:
        with pytest.raises(phase.PrivatePhaseError, match="identity unavailable"):
            host.identify()
        assert client.close.call_count == (0 if fault == "client" else 1)


def test_close_client_failure_is_explicit():
    client = Mock()
    client.close.side_effect = OSError("offline failure")
    assert host.close_client(client) is False


def test_factory_has_no_local_mount_and_declares_cpu_only_offline_phase():
    sdk = Mock()
    image = sdk.Image.debian_slim.return_value
    for method in ("apt_install", "run_commands", "pip_install", "env"):
        getattr(image, method).return_value = image
    app = sdk.App.return_value
    manifest, _ = _brief()
    result_app, function = host.factory(sdk, manifest)
    assert result_app is app
    assert function is app.function.return_value.return_value
    sdk.App.assert_called_once_with(include_source=False)
    options = app.function.call_args.kwargs
    assert options["include_source"] is False
    assert options["cpu"] == (4, 4)
    assert options["memory"] == (16384, 16384)
    assert options["gpu"] is None
    assert options["retries"] == 0
    assert options["max_containers"] == 1
    assert options["min_containers"] == options["buffer_containers"] == 0
    assert options["block_network"] is options["restrict_modal_access"] is True
    assert options["single_use_containers"] is True
    assert options["timeout"] == manifest["timeout_seconds"]
    assert "volumes" not in options and "secrets" not in options
    declarations = repr(image.method_calls)
    assert "torch==2.6.0+cpu" in declarations
    assert "huggingface_hub==0.36.2" in declarations
    assert "transformers==4.57.3" in declarations
    assert "_pinned_qwen_snapshot" in declarations
    assert "add_local" not in declarations
    app.run.assert_not_called()


@pytest.mark.parametrize("dirty", [False, True])
def test_remote_worker_requires_exact_image_commit_before_consumer(monkeypatch, dirty):
    manifest, packet = _brief()
    run = Mock(side_effect=[Mock(stdout=manifest["commit"] + "\n"), Mock(returncode=int(dirty))])
    monkeypatch.setattr(host.subprocess, "run", run)
    consumer = Mock(return_value=({"status": "offline"}, b"checkpoint"))
    monkeypatch.setattr(phase, "execute_private_phase", consumer)
    if dirty:
        with pytest.raises(RuntimeError, match="source commit mismatch"):
            host._remote_phase(manifest, packet, None)
        consumer.assert_not_called()
    else:
        report, blob = host._remote_phase(manifest, packet, None)
        assert report == consumer.return_value[0]
        assert host.decompress_checkpoint(blob) == b"checkpoint"
        assert consumer.call_args.kwargs == {
            "verified_commit": manifest["commit"],
            "verified_workspace": phase.WORKSPACE,
        }
    assert run.call_count == 2


def test_checkout_guard_checks_own_commit_and_clean_inventory(monkeypatch):
    root = Path(host.__file__).resolve().parents[3]
    run = Mock(side_effect=["a" * 40, ""])
    monkeypatch.setattr(host, "git", run)
    host.verify_checkout(root, "a" * 40)
    assert run.call_args_list[1].args[1:] == ("status", "--porcelain=v1", "--untracked-files=all")


@pytest.mark.parametrize("execute", [False, True])
def test_private_cli_only_preflights_and_never_opens_provider(
    tmp_path, monkeypatch, capsys, execute
):
    manifest, packet = _brief()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    packet_path = tmp_path / "packet.bin"
    packet_path.write_bytes(packet)
    forbidden = Mock(side_effect=AssertionError("provider forbidden"))
    monkeypatch.setattr(host, "identify", forbidden)
    argv = [
        "--manifest",
        str(manifest_path),
        "--packet",
        str(packet_path),
        "--checkout",
        str(tmp_path),
        "--journal-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "output.bin"),
        "--approved-manifest-sha256",
        content_fingerprint(manifest),
    ]
    if execute:
        argv.append("--execute")
    assert host.main(argv) == (2 if execute else 0)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("PRIVATE_PHASE_BLOCKED_NO_RETRY" if execute else "PREFLIGHT_ONLY")
    forbidden.assert_not_called()
    assert not (tmp_path / "output.bin").exists()


def test_remote_result_fails_closed_when_compressed_checkpoint_exceeds_inline_limit(monkeypatch):
    manifest, packet = _brief()
    run = Mock(side_effect=[Mock(stdout=manifest["commit"] + "\n"), Mock(returncode=0)])
    monkeypatch.setattr(host.subprocess, "run", run)
    monkeypatch.setattr(host, "INLINE_RESULT_LIMIT", 8)
    report = {"status": "BRIEF_READY_UNREVIEWED", "manifest_sha256": "m", "owner": "o"}
    monkeypatch.setattr(phase, "execute_private_phase", Mock(return_value=(report, b"x" * 1000)))
    result, blob = host._remote_phase(manifest, packet, None)
    assert blob is None
    assert result["status"] == "PRIVATE_PHASE_FAILED_NO_RETRY"
    assert result["retention"] == "OUTPUT_NOT_RETAINED"
    assert result["checkpoint_bytes"] == 0


def test_remote_result_without_checkpoint_passes_through(monkeypatch):
    manifest, packet = _brief()
    run = Mock(side_effect=[Mock(stdout=manifest["commit"] + "\n"), Mock(returncode=0)])
    monkeypatch.setattr(host.subprocess, "run", run)
    report = {"status": "PRIVATE_PHASE_FAILED_NO_RETRY"}
    monkeypatch.setattr(phase, "execute_private_phase", Mock(return_value=(report, None)))
    assert host._remote_phase(manifest, packet, None) == (report, None)


@pytest.mark.parametrize(
    "blob",
    [b"", b"CXZ1", b"CXZ1not-xz", host.compress_checkpoint(b"a") + b"junk", b"XXXX" + b"\0" * 8],
)
def test_decompress_checkpoint_rejects_malformed_blobs(blob):
    with pytest.raises(phase.PrivatePhaseError, match="invalid compressed checkpoint"):
        host.decompress_checkpoint(blob)


def test_decompress_checkpoint_rejects_oversize_expansion(monkeypatch):
    monkeypatch.setattr(phase, "MAX_PACKET", 100)
    with pytest.raises(phase.PrivatePhaseError, match="invalid compressed checkpoint"):
        host.decompress_checkpoint(host.compress_checkpoint(b"\0" * 1000))
