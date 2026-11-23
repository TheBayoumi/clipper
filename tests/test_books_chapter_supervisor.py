"""Run the owner-side watchdog against synthetic hung GitHub CLI commands only."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/private_chapter_external_supervisor.sh"


@pytest.mark.parametrize(
    "hang",
    [
        "ssh",
        "copy",
        "stop",
        "wrong_identity",
        "swap_before_stop",
        "swap_after_stop",
        "cleanup_116",
        "expired_121",
        "unsafe_remote",
        "shell_metacharacters",
        "near_deadline_hung_stop",
    ],
)
def test_external_owner_watchdog_bounds_hung_remote_calls_and_still_deletes(
    tmp_path: Path, hang: str
) -> None:
    fake_bin = tmp_path / "fake-commands"
    fake_bin.mkdir()
    log = tmp_path / "synthetic-gh-operations.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import datetime,json,os,sys,time\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['MOCK_GH_LOG'],'a') as f: f.write(str(time.time())+' '+ ' '.join(args)+'\\n')\n"
        "hang=os.environ['MOCK_HANG']\n"
        "if args[:2]==['api','-X']:\n"
        "  if args[2]=='POST' and hang in ('stop','near_deadline_hung_stop'): time.sleep(120)\n"
        "  if args[2]=='POST' and hang=='swap_after_stop':\n"
        "    open(os.environ['MOCK_SWAPPED'],'w').write('replaced')\n"
        "  if args[2]=='POST' and hang!='stop':\n"
        "    open(os.environ['MOCK_STOPPED'],'w').write('stopped')\n"
        "  if args[2]=='DELETE': sys.exit(0)\n"
        "  sys.exit(0)\n"
        "if args[:2]==['api','-i']:\n"
        "  print('HTTP/2 404 Not Found')\n"
        "  sys.exit(0)\n"
        "if args[:1]==['api']:\n"
        "  created=os.environ['MOCK_CREATED']\n"
        "  swapped=os.path.exists(os.environ['MOCK_SWAPPED'])\n"
        "  state='Stopped' if os.path.exists(os.environ['MOCK_STOPPED']) else 'Available'\n"
        "  print(json.dumps({'id':999 if swapped else 12,"
        "'name':os.environ['CODESPACE_NAME'],"
        "'state':state,'created_at':created,"
        "'repository':{'full_name':os.environ['MOCK_REPO']},"
        "'machine':{'cpus':4,'memory_in_bytes':17179869184,"
        "'storage_in_bytes':34359738368}}))\n"
        "  sys.exit(0)\n"
        "if args[:2]==['codespace','ssh']:\n"
        "  if hang=='ssh': time.sleep(120)\n"
        "  if hang=='stop':\n"
        "    open(os.environ['MOCK_STOPPED'],'w').write('stopped')\n"
        "    sys.exit(3)\n"
        "  if hang=='swap_before_stop':\n"
        "    open(os.environ['MOCK_SWAPPED'],'w').write('replaced')\n"
        "    sys.exit(3)\n"
        "  sys.exit(0)\n"
        "if args[:2]==['codespace','cp']:\n"
        "  if hang=='copy': time.sleep(120)\n"
        "  sys.exit(0)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    key = tmp_path / "synthetic-owner.key"
    public = tmp_path / "synthetic-owner.pub"
    key.write_text("TEST_ONLY", encoding="utf-8")
    public.write_text("TEST_ONLY", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CODESPACE_NAME": "synthetic-only-approved-name",
            "PRIVATE_CHAPTER_APPROVED_SHA": "a" * 40,
            "PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY": str(key),
            "PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY": str(public),
            "PRIVATE_CHAPTER_ROOT": (
                str(tmp_path / "synthetic host")
                if hang == "unsafe_remote"
                else str(tmp_path / "host'; touch DO_NOT_EXECUTE; echo '")
                if hang == "shell_metacharacters"
                else str(tmp_path / "synthetic-host")
            ),
            "PRIVATE_CHAPTER_RECOVERY_DIR": str(tmp_path / "owner-private-recovery"),
            "MOCK_CREATED": (
                datetime.now(UTC)
                - (
                    timedelta(seconds=7196)
                    if hang == "near_deadline_hung_stop"
                    else timedelta(
                        minutes=116
                        if hang == "cleanup_116"
                        else 121
                        if hang == "expired_121"
                        else 5
                    )
                )
            ).isoformat(),
            "MOCK_SWAPPED": str(tmp_path / "synthetic-swapped.txt"),
            "MOCK_HANG": hang,
            "MOCK_REPO": (
                "other/unauthorized" if hang == "wrong_identity" else "TheBayoumi/clipper"
            ),
            "MOCK_GH_LOG": str(log),
            "MOCK_STOPPED": str(tmp_path / "synthetic-stopped.txt"),
        }
    )
    env.pop("CODESPACES", None)
    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=35,
    )
    elapsed = time.monotonic() - started
    expected_status = (
        0
        if hang in {"cleanup_116", "near_deadline_hung_stop"}
        else 124
        if hang in {"ssh", "copy"}
        else 3
        if hang == "stop"
        else 2
    )
    assert result.returncode == expected_status
    assert elapsed < 30
    actions = log.read_text(encoding="utf-8")
    if hang == "near_deadline_hung_stop":
        records = [
            (float(line.split(" ", 1)[0]), line.split(" ", 1)[1])
            for line in actions.splitlines()
        ]
        approved_hard_deadline = datetime.fromisoformat(env["MOCK_CREATED"]).timestamp() + 7200
        deletes = [
            stamp for stamp, call in records if "api -X DELETE" in call
        ]
        assert len(deletes) == 1
        assert deletes[0] < approved_hard_deadline
        assert not any("api -X POST" in call for _, call in records)
        assert "EXTERNAL_SUPERVISOR_HOST_DELETION_REST_VERIFIED" in result.stdout
    if hang in {"wrong_identity", "swap_before_stop", "expired_121"}:
        assert "api -X POST" not in actions
        assert "api -X DELETE" not in actions
    elif hang == "swap_after_stop":
        assert "api -X POST" in actions
        assert "api -X DELETE" not in actions
    else:
        if hang != "near_deadline_hung_stop":
            assert "api -X POST" in actions
        assert "api -X DELETE" in actions
        assert "api -i" in actions
    if hang in {
        "cleanup_116",
        "near_deadline_hung_stop",
        "unsafe_remote",
        "shell_metacharacters",
        "expired_121",
    }:
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
    if hang == "expired_121":
        assert "TOTAL_DEADLINE_EXPIRED" in result.stderr
        assert "WRONG_HOST_IDENTITY" not in result.stderr
    if hang in {"unsafe_remote", "shell_metacharacters"}:
        assert "DO_NOT_EXECUTE" not in result.stdout
        assert not (tmp_path / "DO_NOT_EXECUTE").exists()
    # Stop returning or stalling never licenses running another host.
    assert "PRIVATE_CHAPTER" not in actions


@pytest.mark.parametrize("tamper_archive", [False, True])
def test_owner_recovery_hash_verification_is_independent_of_rest_deletion(
    tmp_path: Path, tamper_archive: bool
) -> None:
    """No Codespace: fake CLI transfers one synthetic TAR or a corrupted copy."""
    source = tmp_path / "synthetic-private-outputs.tar"
    relative = "book-candidates/brief/synthetic/validated.json"
    text_bytes = b'{"only":"fabricated fixture text"}'
    file_entry = {
        "chapter": 1,
        "kind": "text",
        "relative_path": relative,
        "sha256": hashlib.sha256(text_bytes).hexdigest(),
        "bytes": len(text_bytes),
    }
    safe = {
        "schema": "clipper-private-chapter-safe-recovery-v1",
        "status": "PRIVATE_CHAPTER_RECOVERY_CHECKPOINT_ONLY",
        "chapters": {
            "1": {
                "recovery_state": "IN_FLIGHT",
                "last_subphase": "NARRATION_IN_FLIGHT",
                "runtime_journal_status": "NARRATION_IN_FLIGHT",
                "retrieval_status": "NOT_ATTEMPTED",
                "last_verified_artifact_identities": {"brief_identity": "a" * 64},
            },
            "7": {
                "recovery_state": "NEVER_STARTED",
                "last_subphase": "NOT_STARTED",
                "runtime_journal_status": "NOT_STARTED",
                "retrieval_status": "NOT_ATTEMPTED",
                "last_verified_artifact_identities": {},
            },
        },
        "files_retrieved": False,
        "publish_eligible": False,
    }
    safe_bytes = json.dumps(safe, sort_keys=True).encode("utf-8")
    listing = {
        "schema": "clipper-chapter-private-recovery-archive-manifest-v1",
        "chapters": [],
        "files": [file_entry],
        "runtime_status_sha256": hashlib.sha256(safe_bytes).hexdigest(),
    }
    listing_bytes = json.dumps(listing, sort_keys=True).encode("utf-8")
    with tarfile.open(source, "w") as archive:
        for name, data in (
            (relative, text_bytes),
            ("private-recovery-manifest.json", listing_bytes),
            ("safe-recovery-status.json", safe_bytes),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o600
            archive.addfile(member, io.BytesIO(data))
    receipt = {
        "schema": "clipper-private-chapter-recovery-created-v1",
        "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(listing_bytes).hexdigest(),
    }
    response = tmp_path / "synthetic-remote-recovery-result.json"
    response.write_text(json.dumps(receipt), encoding="utf-8")
    fake_bin = tmp_path / "commands"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import datetime,json,os,pathlib,shutil,sys\n"
        "a=sys.argv[1:]\n"
        "with open(os.environ['MOCK_LOG'],'a') as f: f.write(' '.join(a)+'\\n')\n"
        "if a[:2]==['api','-X']:\n"
        "  if a[2]=='POST': pathlib.Path(os.environ['MOCK_STOP']).write_text('stopped')\n"
        "  sys.exit(0)\n"
        "if a[:2]==['api','-i']:\n"
        "  print('HTTP/2 404 Not Found'); sys.exit(0)\n"
        "if a[:1]==['api']:\n"
        "  state='Stopped' if pathlib.Path(os.environ['MOCK_STOP']).exists() else 'Available'\n"
        "  print(json.dumps({'id':12,'name':os.environ['CODESPACE_NAME'],"
        "'state':state,'created_at':os.environ['MOCK_CREATED'],"
        "'repository':{'full_name':'TheBayoumi/clipper'},"
        "'machine':{'cpus':4,'memory_in_bytes':17179869184,"
        "'storage_in_bytes':34359738368}})); sys.exit(0)\n"
        "if a[:2]==['codespace','ssh']:\n"
        "  if ' recover ' in (' '+ ' '.join(a)+' '):\n"
        "    print(pathlib.Path(os.environ['MOCK_RECEIPT']).read_text())\n"
        "  sys.exit(0)\n"
        "if a[:2]==['codespace','cp']:\n"
        "  if 'chapter-recovery.tar' in ' '.join(a):\n"
        "    shutil.copyfile(os.environ['MOCK_ARCHIVE'],a[-1])\n"
        "    if os.environ['MOCK_TAMPER']=='true':\n"
        "      with open(a[-1],'ab') as f: f.write(b'corrupted')\n"
        "  sys.exit(0)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    key = tmp_path / "synthetic-private-key"
    public = tmp_path / "synthetic-public-key"
    key.write_text("INVALID_SIGNING_KEY_FOR_TEST", encoding="utf-8")
    public.write_text("PUBLIC_ONLY", encoding="utf-8")
    log = tmp_path / "synthetic-gh-ops.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CODESPACE_NAME": "synthetic-private-host",
            "PRIVATE_CHAPTER_APPROVED_SHA": "f" * 40,
            "PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY": str(key),
            "PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY": str(public),
            "PRIVATE_CHAPTER_ROOT": str(tmp_path / "synthetic-host-root"),
            "PRIVATE_CHAPTER_RECOVERY_DIR": str(tmp_path / "owner-private-recovery"),
            "MOCK_ARCHIVE": str(source),
            "MOCK_RECEIPT": str(response),
            "MOCK_LOG": str(log),
            "MOCK_CREATED": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
            "MOCK_STOP": str(tmp_path / "synthetic-stopped"),
            "MOCK_TAMPER": "true" if tamper_archive else "false",
        }
    )
    env.pop("CODESPACES", None)
    result = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0  # Invalid synthetic signer prevents host setup.
    actions = log.read_text(encoding="utf-8")
    assert "api -X POST" in actions and "api -X DELETE" in actions
    folder = tmp_path / "owner-private-recovery/synthetic-private-host"
    verification = json.loads((folder / "retrieval-verification.json").read_bytes())
    cleanup = json.loads((folder / "host-cleanup-verification.json").read_bytes())
    assert cleanup["host_stop_verified"] is True
    assert cleanup["host_deletion_verified"] is True
    assert verification["retrieval_status"] == (
        "INCOMPLETE" if tamper_archive else "VERIFIED_PRIVATE_ARCHIVE"
    )
    assert (folder / "chapter-recovery.tar").exists() is not tamper_archive
    assert "fabricated fixture text" not in result.stdout + result.stderr
