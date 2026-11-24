"""Synthetic external-host verification; no Codespace, model, or network access."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import chapter_lifecycle


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], Path, Path]:
    root = tmp_path / "book-candidates"
    folder = root / "private-host"
    folder.mkdir(parents=True)
    secret = tmp_path / "owner-private-key-only.pem"
    key = folder / "external-supervisor-public.pem"
    subprocess.run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            str(secret),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(secret), "-pubout", "-out", str(key)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    monkeypatch.setenv("CODESPACES", "true")
    monkeypatch.setenv("PRIVATE_CHAPTER_APPROVED_SHA", "a" * 40)
    monkeypatch.setenv(
        "PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256",
        hashlib.sha256(key.read_bytes()).hexdigest(),
    )
    created = datetime.now(UTC) - timedelta(minutes=5)
    record: dict[str, object] = {
        "id": 12,
        "name": "one-synthetic-codespace",
        "state": "Available",
        "repository": {"full_name": "TheBayoumi/clipper"},
        "machine": {
            "cpus": 4,
            "memory_in_bytes": 17179869184,
            "storage_in_bytes": 34359738368,
        },
        "created_at": created.isoformat(),
    }
    signed = {
        "schema": chapter_lifecycle.SUPERVISOR_SCHEMA,
        "codespace_id": 12,
        "codespace_name": record["name"],
        "created_at": record["created_at"],
        "approved_sha": "a" * 40,
        "observed_at": datetime.now(UTC).isoformat(),
        "hard_deadline_at": (created + timedelta(seconds=7200)).isoformat(),
        "external_watchdog_required": True,
    }
    heartbeat = folder / "external-supervisor-heartbeat.json"
    signature = folder / "external-supervisor-heartbeat.sig"

    def sign() -> None:
        heartbeat.write_bytes((json.dumps(signed, sort_keys=True) + "\n").encode())
        subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-sign",
                str(secret),
                "-out",
                str(signature),
                str(heartbeat),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    sign()
    original = chapter_lifecycle._command

    def simulated_commands(argv: list[str], *, input_bytes: bytes | None = None) -> bytes:
        if argv[:2] == ["git", "rev-parse"]:
            return b"review/unified-tts-cleanup" if "--abbrev-ref" in argv else b"a" * 40
        if argv[:2] == ["git", "status"]:
            return b""
        if argv[:2] == ["gh", "api"]:
            return json.dumps(record).encode()
        return original(argv, input_bytes=input_bytes)

    monkeypatch.setattr(chapter_lifecycle, "_command", simulated_commands)
    # A private key is intentionally NOT copied to the host artifact root.
    assert secret.is_file()
    assert not list(root.rglob("owner-private-key-only.pem"))
    return root, record, heartbeat, signature


def _check(root: Path, record: dict[str, object]) -> chapter_lifecycle.VerifiedLifecycle:
    return chapter_lifecycle.verify_lifecycle(
        root,
        caller_created_at=str(record["created_at"]),
        codespace_name="one-synthetic-codespace",
    )


def test_signed_external_supervisor_and_authoritative_rest_all_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, record, heartbeat, _ = _fixture(tmp_path, monkeypatch)
    verified = _check(root, record)
    assert verified.approved_sha == "a" * 40
    assert verified.codespace_id == 12
    assert verified.codespace_name == "one-synthetic-codespace"
    assert verified.expires_at == verified.created_at + timedelta(seconds=7200)
    heartbeat.write_bytes(heartbeat.read_bytes() + b" ")
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_signature",
        "stale_signature",
        "missing_public_pin",
        "wrong_public_pin",
        "missing_approved_sha",
        "wrong_shape",
        "wrong_codespace",
        "wrong_id",
        "wrong_repo",
        "wrong_state",
        "wrong_created",
        "wrong_client_created",
        "not_a_codespace",
    ],
)
def test_live_external_identity_and_signing_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    root, record, heartbeat, signature = _fixture(tmp_path, monkeypatch)
    if fault == "missing_signature":
        signature.unlink()
    elif fault == "stale_signature":
        signature.write_bytes(b"fake")
    elif fault == "missing_public_pin":
        monkeypatch.delenv("PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256")
    elif fault == "wrong_public_pin":
        monkeypatch.setenv("PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256", "f" * 64)
    elif fault == "missing_approved_sha":
        monkeypatch.delenv("PRIVATE_CHAPTER_APPROVED_SHA")
    elif fault == "wrong_shape":
        record["machine"]["memory_in_bytes"] = 1234
    elif fault == "wrong_codespace":
        record["name"] = "different-one"
    elif fault == "wrong_id":
        record["id"] = 13
    elif fault == "wrong_repo":
        record["repository"]["full_name"] = "other/private"
    elif fault == "wrong_state":
        record["state"] = "Shutdown"
    elif fault == "wrong_created":
        record["created_at"] = datetime.now(UTC).isoformat()
    elif fault == "wrong_client_created":
        record["created_at"] = "not-a-timestamp"
    else:
        monkeypatch.setenv("CODESPACES", "false")
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)
    assert heartbeat.is_file()


def test_approval_sha_and_rest_timestamp_cannot_be_self_asserted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, record, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        chapter_lifecycle.verify_lifecycle(
            root,
            caller_created_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            codespace_name="one-synthetic-codespace",
        )
    monkeypatch.setenv("PRIVATE_CHAPTER_APPROVED_SHA", "b" * 40)
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)


def test_cli_without_external_watchdog_never_launches_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called = Mock(side_effect=chapter_lifecycle.LifecycleStop("missing owner supervisor"))
    monkeypatch.setattr(chapter_lifecycle, "verify_lifecycle", called)
    monkeypatch.setenv("CODESPACE_NAME", "synthetic")
    result = chapter_lifecycle.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--codespace-created-at",
            datetime.now(UTC).isoformat(),
        ]
    )
    assert result == 2
    assert "PRIVATE_CHAPTER_EXTERNAL_SUPERVISOR_UNVERIFIED" in capsys.readouterr().out
    called.assert_called_once()


@pytest.mark.parametrize("value", [None, 123, "2026-01-01T10:00:00", "invalid-date"])
def test_host_clock_requires_exact_offset_aware_iso_timestamp(value: object) -> None:
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        chapter_lifecycle._utc(value)


def test_rest_and_supervisor_evidence_are_finite_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    with pytest.raises(chapter_lifecycle.LifecycleStop, match="bounded evidence"):
        chapter_lifecycle._command([sys.executable, "-c", "print('x' * 40000)"])
    root, record, heartbeat, _signature = _fixture(tmp_path, monkeypatch)
    with pytest.raises(chapter_lifecycle.LifecycleStop, match="finite limit"):
        chapter_lifecycle._trusted_file(root, heartbeat.name, 1)
    valid_command = chapter_lifecycle._command
    for payload in (b"not-json", b"[]"):

        def poisoned(
            argv: list[str], *, input_bytes: bytes | None = None, bad: bytes = payload
        ) -> bytes:
            if argv[:2] == ["gh", "api"]:
                return bad
            return valid_command(argv, input_bytes=input_bytes)

        monkeypatch.setattr(chapter_lifecycle, "_command", poisoned)
        with pytest.raises(chapter_lifecycle.LifecycleStop, match="REST identity"):
            _check(root, record)
    monkeypatch.setattr(chapter_lifecycle, "_command", valid_command)
    record["created_at"] = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    with pytest.raises(chapter_lifecycle.LifecycleStop, match="does not attest"):
        _check(root, record)


@pytest.mark.parametrize("damage", ["non_json", "stale", "future"])
def test_signed_malformed_or_expired_external_heartbeat_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    root, record, heartbeat, signature = _fixture(tmp_path, monkeypatch)
    payload = json.loads(heartbeat.read_bytes())
    if damage == "non_json":
        raw = b"not-a-json-document"
    else:
        offset = timedelta(minutes=3) if damage == "future" else -timedelta(minutes=3)
        payload["observed_at"] = (datetime.now(UTC) + offset).isoformat()
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    heartbeat.write_bytes(raw)
    secret = tmp_path / "owner-private-key-only.pem"
    subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-sign",
            str(secret),
            "-out",
            str(signature),
            str(heartbeat),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)


def test_lifecycle_cli_success_and_invalid_root_never_approves_host_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    verified = Mock(return_value=None)
    monkeypatch.setattr(chapter_lifecycle, "verify_lifecycle", verified)
    monkeypatch.setenv("CODESPACE_NAME", "synthetic")
    timestamp = datetime.now(UTC).isoformat()
    assert (
        chapter_lifecycle.main(
            [
                "--artifact-root",
                str(tmp_path),
                "--codespace-created-at",
                timestamp,
            ]
        )
        == 0
    )
    assert "VERIFIED_CHECKPOINT_ONLY" in capsys.readouterr().out
    verified.assert_called_once()
    assert (
        chapter_lifecycle.main(
            [
                "--artifact-root",
                "relative-only",
                "--codespace-created-at",
                timestamp,
            ]
        )
        == 2
    )
    assert "UNVERIFIED" in capsys.readouterr().out
    verified.assert_called_once()


def test_atomic_signed_envelope_contains_one_exact_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, record, heartbeat, signature = _fixture(tmp_path, monkeypatch)
    folder = heartbeat.parent
    envelope = folder / "external-supervisor-heartbeat-envelope.json"
    envelope.write_bytes(
        json.dumps(
            {
                "schema": "clipper-external-supervisor-envelope-v1",
                "payload_b64": base64.b64encode(heartbeat.read_bytes()).decode("ascii"),
                "signature_b64": base64.b64encode(signature.read_bytes()).decode("ascii"),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    heartbeat.unlink()
    signature.unlink()
    assert _check(root, record).codespace_id == 12
    tampered = json.loads(envelope.read_bytes())
    tampered["payload_b64"] = base64.b64encode(b"not signed").decode("ascii")
    envelope.write_bytes(json.dumps(tampered).encode("utf-8"))
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)


def test_replaced_legacy_files_cannot_change_verified_payload_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, record, heartbeat, signature = _fixture(tmp_path, monkeypatch)
    original = chapter_lifecycle._command
    did_swap = False

    def replace_between_snapshot_and_openssl(
        argv: list[str], *, input_bytes: bytes | None = None
    ) -> bytes:
        nonlocal did_swap
        if argv[:2] == ["openssl", "dgst"] and not did_swap:
            assert input_bytes is not None
            did_swap = True
            heartbeat.write_bytes(b'{"unverified":true}')
            signature.write_bytes(b"forged")
        return original(argv, input_bytes=input_bytes)

    monkeypatch.setattr(chapter_lifecycle, "_command", replace_between_snapshot_and_openssl)
    assert _check(root, record).codespace_id == 12
    assert did_swap
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)


@pytest.mark.parametrize(
    "fault", ["bad_base64", "oversized_payload", "wrong_envelope_schema", "invalid_signature"]
)
def test_atomic_signed_envelope_rejects_malformed_or_untrusted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    root, record, heartbeat, signature = _fixture(tmp_path, monkeypatch)
    payload = heartbeat.read_bytes()
    signed = signature.read_bytes()
    schema = "clipper-external-supervisor-envelope-v1"
    if fault == "bad_base64":
        encoded_payload = "not-base64!"
    elif fault == "oversized_payload":
        encoded_payload = base64.b64encode(b"x" * 8193).decode("ascii")
    else:
        encoded_payload = base64.b64encode(payload).decode("ascii")
    if fault == "wrong_envelope_schema":
        schema = "untrusted-envelope"
    if fault == "invalid_signature":
        signed = b"not an OpenSSL signature"
    envelope = {
        "schema": schema,
        "payload_b64": encoded_payload,
        "signature_b64": base64.b64encode(signed).decode("ascii"),
    }
    (heartbeat.parent / "external-supervisor-heartbeat-envelope.json").write_bytes(
        json.dumps(envelope, sort_keys=True).encode("utf-8")
    )
    with pytest.raises(chapter_lifecycle.LifecycleStop):
        _check(root, record)


def test_bounded_heartbeat_snapshot_rejects_device_race_and_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a pathname replaced between the trusted stat and the actual
    # open; even a successful open of a directory cannot count as a snapshot.
    monkeypatch.setattr(chapter_lifecycle, "_trusted_file", lambda *_args: tmp_path)
    with pytest.raises(chapter_lifecycle.LifecycleStop, match="regular file"):
        chapter_lifecycle._snapshot(tmp_path, "synthetic", 8)
    monkeypatch.setattr(chapter_lifecycle, "_trusted_file", lambda *_args: tmp_path / "missing")
    with pytest.raises(chapter_lifecycle.LifecycleStop, match="cannot be snapshotted"):
        chapter_lifecycle._snapshot(tmp_path, "synthetic", 8)
    oversized = tmp_path / "changed-after-stat.bin"
    oversized.write_bytes(b"x" * 32)
    monkeypatch.setattr(chapter_lifecycle, "_trusted_file", lambda *_args: oversized)
    with pytest.raises(chapter_lifecycle.LifecycleStop, match="finite limit"):
        chapter_lifecycle._snapshot(tmp_path, "synthetic", 8)
