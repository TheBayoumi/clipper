"""Fail-closed Codespace and *external* supervisor attestation.

A local Python process cannot enforce host deletion. The owner's independent
supervisor must keep signing short-lived heartbeats while retaining its private
key OUTSIDE the Codespace and force stop/deletion at the creation deadline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SUPERVISOR_SCHEMA = "clipper-external-codespace-supervisor-v1"
MAX_HEARTBEAT_AGE_SECONDS = 45
EXECUTION_SECONDS = 7200 - 300
TOTAL_SECONDS = 7200
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
    if created > now or now >= created + timedelta(seconds=EXECUTION_SECONDS):
        raise LifecycleStop("shared creation-time execution deadline expired")
    key = _trusted_file(private_root, "external-supervisor-public.pem", 8192)
    heartbeat = _trusted_file(private_root, "external-supervisor-heartbeat.json", 8192)
    signature = _trusted_file(private_root, "external-supervisor-heartbeat.sig", 8192)
    if hashlib.sha256(key.read_bytes()).hexdigest() != public_key_sha:
        raise LifecycleStop("independent supervisor public key pin changed")
    # Verification is by OpenSSL's public-key verifier; no secret may be on host.
    _command(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(key),
            "-signature",
            str(signature),
            str(heartbeat),
        ]
    )
    raw = heartbeat.read_bytes()
    try:
        evidence: Any = json.loads(raw)
    except ValueError:
        raise LifecycleStop("signed supervisor heartbeat is malformed") from None
    if (
        not isinstance(evidence, dict)
        or set(evidence)
        != {
            "schema",
            "codespace_id",
            "codespace_name",
            "created_at",
            "approved_sha",
            "observed_at",
            "hard_deadline_at",
            "external_watchdog_required",
        }
        or evidence["schema"] != SUPERVISOR_SCHEMA
        or evidence["codespace_id"] != code_id
        or evidence["codespace_name"] != codespace_name
        or evidence["approved_sha"] != expected_sha
        or evidence["external_watchdog_required"] is not True
        or _utc(evidence["created_at"]) != created
        or _utc(evidence["hard_deadline_at"]) != created + timedelta(seconds=TOTAL_SECONDS)
    ):
        raise LifecycleStop("external supervisor does not attest this exact Codespace")
    recent = _utc(evidence["observed_at"])
    age = (now - recent).total_seconds()
    if age < -5 or age > MAX_HEARTBEAT_AGE_SECONDS:
        raise LifecycleStop("external supervisor heartbeat is stale or future-dated")
    return VerifiedLifecycle(
        code_id,
        codespace_name,
        created,
        expected_sha,
        created + timedelta(seconds=TOTAL_SECONDS),
    )


def main(argv: list[str] | None = None) -> int:
    """Preflight BEFORE setup: no model dependencies, no private source logging."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify external lifecycle before chapter setup")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--codespace-created-at", required=True)
    args = parser.parse_args(argv)
    try:
        name = os.environ.get("CODESPACE_NAME", "")
        if not args.artifact_root.is_absolute() or args.artifact_root.is_symlink():
            raise LifecycleStop("private artifact root is not absolute or is redirected")
        verify_lifecycle(
            args.artifact_root / "book-candidates",
            caller_created_at=args.codespace_created_at,
            codespace_name=name,
        )
    except (LifecycleStop, ValueError, OSError):
        print("PRIVATE_CHAPTER_EXTERNAL_SUPERVISOR_UNVERIFIED")
        return 2
    print("PRIVATE_CHAPTER_EXTERNAL_SUPERVISOR_VERIFIED_CHECKPOINT_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
