"""Fail-closed Codespace and *external* supervisor attestation.

A local Python process cannot enforce host deletion. The owner's independent
supervisor must keep signing short-lived heartbeats while retaining its private
key OUTSIDE the Codespace and force stop/deletion at the creation deadline.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SUPERVISOR_SCHEMA = "clipper-external-codespace-supervisor-v1"
MAX_HEARTBEAT_AGE_SECONDS = 45
EXECUTION_SECONDS = 7200 - 300
TOTAL_SECONDS = 7200
CLEANUP_RESERVE_SECONDS = 300
# Independent Codespaces running-session ceiling is NOT a free allowance grant.
MAX_GRANT_SECONDS = 12 * 3600 - 300
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class LifecycleStop(ValueError):
    """The external hard-lifecycle requirement has not been proven."""


@dataclass(frozen=True, slots=True)
class VerifiedLifecycle:
    codespace_id: int
    codespace_name: str
    created_at: datetime
    approved_sha: str
    expires_at: datetime
    execution_deadline: datetime | None = None


def _utc(text: object) -> datetime:
    if not isinstance(text, str):
        raise LifecycleStop("authoritative Codespace timestamp unavailable")
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise LifecycleStop("authoritative Codespace timestamp unavailable") from None
    if value.tzinfo is None:
        raise LifecycleStop("Codespace timestamp has no UTC offset")
    return value.astimezone(UTC)


def _command(argv: list[str], *, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            argv,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=12,
            check=True,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        raise LifecycleStop(
            "authoritative host or external supervisor verification unavailable"
        ) from None
    if len(result.stdout) > 32768:
        raise LifecycleStop("authoritative host response exceeds bounded evidence")
    return result.stdout.strip()


def _trusted_file(root: Path, name: str, limit: int) -> Path:
    folder = root / "private-host"
    path = folder / name
    if folder.is_symlink() or path.is_symlink() or not path.is_file():
        raise LifecycleStop("external supervisor proof is missing or redirected")
    if path.stat().st_size > limit:
        raise LifecycleStop("external supervisor evidence exceeds finite limit")
    return path


def _snapshot(root: Path, name: str, limit: int) -> bytes:
    """Open once, refuse symlinks, and capture only bounded immutable evidence."""
    path = _trusted_file(root, name, limit)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            import stat

            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise LifecycleStop("external supervisor evidence is not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(limit + 1)
        finally:
            os.close(fd)
    except OSError:
        raise LifecycleStop("external supervisor evidence cannot be snapshotted") from None
    if len(data) > limit:
        raise LifecycleStop("external supervisor evidence exceeds finite limit")
    return data


def _signed_snapshot(private_root: Path) -> tuple[bytes, bytes]:
    """Prefer the one-file atomically published payload/signature envelope."""
    envelope = private_root / "private-host" / "external-supervisor-heartbeat-envelope.json"
    if envelope.exists() or envelope.is_symlink():
        try:
            container = json.loads(_snapshot(private_root, envelope.name, 16384))
            if (
                not isinstance(container, dict)
                or set(container) != {"schema", "payload_b64", "signature_b64"}
                or container["schema"] != "clipper-external-supervisor-envelope-v1"
            ):
                raise ValueError("invalid envelope")
            payload = base64.b64decode(container["payload_b64"], validate=True)
            signature = base64.b64decode(container["signature_b64"], validate=True)
            if not payload or len(payload) > 8192 or not signature or len(signature) > 8192:
                raise ValueError("oversized envelope")
            return payload, signature
        except (binascii.Error, ValueError, TypeError, KeyError):
            raise LifecycleStop("external supervisor signed envelope malformed") from None
    # Read legacy pair once each; verification MUST use these exact snapshots,
    # not re-open pathnames after the OpenSSL check.
    return (
        _snapshot(private_root, "external-supervisor-heartbeat.json", 8192),
        _snapshot(private_root, "external-supervisor-heartbeat.sig", 8192),
    )


def _verified_free_allowance(
    value: object, *, code_id: int, name: str, created: datetime, sha: str
) -> int:
    """Validate the finite owner attestation INSIDE the verified signed payload."""
    if not isinstance(value, dict) or set(value) != {
        "schema", "codespace_id", "codespace_name", "repository",
        "created_at", "approved_sha", "machine", "verified_at",
        "remaining_included_core_seconds", "total_seconds",
        "storage_included_verified", "paid_overage_block_verified",
    }:
        raise LifecycleStop("signed free allowance grant missing or malformed")
    machine = value.get("machine")
    total = value.get("total_seconds")
    remaining = value.get("remaining_included_core_seconds")
    verified_at = _utc(value.get("verified_at"))
    now = datetime.now(UTC)
    if (
        value["schema"] != "clipper-owner-free-allowance-v1"
        or type(value["codespace_id"]) is not int or value["codespace_id"] != code_id
        or value["codespace_name"] != name
        or value["repository"] != "TheBayoumi/clipper"
        or _utc(value["created_at"]) != created
        or value["approved_sha"] != sha
        or not isinstance(machine, dict)
        or set(machine) != {"cpus", "memory_in_bytes", "storage_in_bytes"}
        or (machine["cpus"], machine["memory_in_bytes"], machine["storage_in_bytes"])
            != (4, 17179869184, 34359738368)
        or type(total) is not int or not (TOTAL_SECONDS < total <= MAX_GRANT_SECONDS)
        or type(remaining) is not int or remaining < total * 4
        or value["storage_included_verified"] is not True
        or value["paid_overage_block_verified"] is not True
        or verified_at > now + timedelta(seconds=5)
        or verified_at < created - timedelta(minutes=30)
        or verified_at > created + timedelta(minutes=5)
    ):
        raise LifecycleStop("signed free allowance grant does not authorize this host")
    return total


def verify_lifecycle(
    private_root: Path,
    *,
    caller_created_at: str,
    codespace_name: str,
) -> VerifiedLifecycle:
    """Check REST identity, exact checkout and signed independent live heartbeat.

    A signed recent heartbeat demonstrates a key outside this Codespace was
    active recently; only the external owner-controlled watchdog can stop and
    verify deletion when this Python process is hung or offline.
    """
    expected_sha = os.environ.get("PRIVATE_CHAPTER_APPROVED_SHA", "")
    public_key_sha = os.environ.get("PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256", "")
    if (
        os.environ.get("CODESPACES") != "true"
        or not codespace_name
        or not SHA.fullmatch(expected_sha)
        or not DIGEST.fullmatch(public_key_sha)
    ):
        raise LifecycleStop("approved SHA and external supervisor must be pinned")
    checked = _command(["git", "rev-parse", "--verify", "HEAD"]).decode("ascii", "strict")
    branch = _command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii", "strict")
    dirty = _command(["git", "status", "--porcelain", "--untracked-files=no"])
    if checked != expected_sha or branch != "review/unified-tts-cleanup" or dirty:
        raise LifecycleStop("Codespace checkout does not match authorized exact SHA")
    try:
        record: Any = json.loads(_command(["gh", "api", f"user/codespaces/{codespace_name}"]))
    except (ValueError, TypeError):
        raise LifecycleStop("Codespace REST identity could not be verified") from None
    if not isinstance(record, dict):
        raise LifecycleStop("Codespace REST identity unavailable")
    machine = record.get("machine")
    repository = record.get("repository")
    name = record.get("name")
    code_id = record.get("id")
    if (
        name != codespace_name
        or type(code_id) is not int
        or code_id < 1
        or not isinstance(repository, dict)
        or repository.get("full_name") != "TheBayoumi/clipper"
        or not isinstance(machine, dict)
        or (
            machine.get("cpus"),
            machine.get("memory_in_bytes"),
            machine.get("storage_in_bytes"),
        )
        != (4, 17179869184, 34359738368)
        or record.get("state") not in {"Available", "Starting"}
    ):
        raise LifecycleStop("Codespace REST shape, identity or state is not approved")
    created = _utc(record.get("created_at"))
    if created != _utc(caller_created_at):
        raise LifecycleStop("caller Codespace timestamp contradicts REST creation time")
    now = datetime.now(UTC)
    if created > now:
        raise LifecycleStop("shared creation-time execution deadline invalid")
    key_bytes = _snapshot(private_root, "external-supervisor-public.pem", 8192)
    raw, signature_bytes = _signed_snapshot(private_root)
    if hashlib.sha256(key_bytes).hexdigest() != public_key_sha:
        raise LifecycleStop("independent supervisor public key pin changed")
    # OpenSSL sees only the snapshotted public key, signature and exact payload.
    # A writer replacing any source path during verification cannot alter it.
    with tempfile.TemporaryDirectory(prefix="clipper-public-verify-") as temp:
        key_path = Path(temp) / "public.pem"
        signature_path = Path(temp) / "signature.bin"
        key_path.write_bytes(key_bytes)
        signature_path.write_bytes(signature_bytes)
        _command(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-verify",
                str(key_path),
                "-signature",
                str(signature_path),
            ],
            input_bytes=raw,
        )
    try:
        evidence: Any = json.loads(raw)
    except ValueError:
        raise LifecycleStop("signed supervisor heartbeat is malformed") from None
    basic_fields = {
        "schema", "codespace_id", "codespace_name", "created_at", "approved_sha",
        "observed_at", "hard_deadline_at", "external_watchdog_required",
    }
    if (
        not isinstance(evidence, dict)
        or set(evidence) not in (basic_fields, basic_fields | {"free_allowance"})
        or evidence["schema"] != SUPERVISOR_SCHEMA
        or evidence["codespace_id"] != code_id
        or evidence["codespace_name"] != codespace_name
        or evidence["approved_sha"] != expected_sha
        or evidence["external_watchdog_required"] is not True
        or _utc(evidence["created_at"]) != created
    ):
        raise LifecycleStop("external supervisor does not attest this exact Codespace")
    total_seconds = (
        _verified_free_allowance(
            evidence["free_allowance"],
            code_id=code_id, name=codespace_name, created=created, sha=expected_sha,
        )
        if "free_allowance" in evidence else TOTAL_SECONDS
    )
    hard_deadline = created + timedelta(seconds=total_seconds)
    if _utc(evidence["hard_deadline_at"]) != hard_deadline:
        raise LifecycleStop("signed hard deadline contradicts free allowance grant")
    recent = _utc(evidence["observed_at"])
    # Freshness must be checked AFTER cryptographic verification, not before a
    # potentially slow external process.
    now = datetime.now(UTC)
    execution_deadline = hard_deadline - timedelta(seconds=CLEANUP_RESERVE_SECONDS)
    if now >= execution_deadline:
        raise LifecycleStop("shared signed execution deadline expired")
    age = (now - recent).total_seconds()
    if age < -5 or age > MAX_HEARTBEAT_AGE_SECONDS:
        raise LifecycleStop("external supervisor heartbeat is stale or future-dated")
    return VerifiedLifecycle(
        code_id,
        codespace_name,
        created,
        expected_sha,
        hard_deadline,
        execution_deadline,
    )


def main(argv: list[str] | None = None) -> int:
    """Preflight BEFORE setup: no model dependencies, no private source logging."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify external lifecycle before chapter setup")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--codespace-created-at", required=True)
    parser.add_argument("--remaining-seconds", action="store_true")
    args = parser.parse_args(argv)
    try:
        name = os.environ.get("CODESPACE_NAME", "")
        if not args.artifact_root.is_absolute() or args.artifact_root.is_symlink():
            raise LifecycleStop("private artifact root is not absolute or is redirected")
        verified = verify_lifecycle(
            args.artifact_root / "book-candidates",
            caller_created_at=args.codespace_created_at,
            codespace_name=name,
        )
    except (LifecycleStop, ValueError, OSError):
        print("PRIVATE_CHAPTER_EXTERNAL_SUPERVISOR_UNVERIFIED")
        return 2
    if args.remaining_seconds:
        assert verified.execution_deadline is not None
        remaining = int((verified.execution_deadline - datetime.now(UTC)).total_seconds())
        if remaining <= 0:
            print("PRIVATE_CHAPTER_EXTERNAL_SUPERVISOR_UNVERIFIED")
            return 2
        print(remaining)
    else:
        print("PRIVATE_CHAPTER_EXTERNAL_SUPERVISOR_VERIFIED_CHECKPOINT_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
