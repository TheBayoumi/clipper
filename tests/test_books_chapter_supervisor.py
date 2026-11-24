"""Run the owner-side watchdog against synthetic hung GitHub CLI commands only."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from contextlib import suppress
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
        "heartbeat_stream_fail",
        "heartbeat_publish_fail",
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
        "  if hang=='heartbeat_publish_fail' and 'watchdog.pending' in ' '.join(args): sys.exit(3)\n"
        "  if hang=='heartbeat_stream_fail' and 'watchdog.pending' in ' '.join(args): sys.exit(3)\n"
        "  sys.exit(0)\n"
        "if args[:2]==['codespace','cp']:\n"
        "  if hang=='copy': time.sleep(120)\n"
        "  if hang=='heartbeat_copy_fail' and 'watchdog.pending' in ' '.join(args): sys.exit(3)\n"
        "  sys.exit(0)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    key = tmp_path / "synthetic-owner.key"
    public = tmp_path / "synthetic-owner.pub"
    key.write_text("TEST_ONLY", encoding="utf-8")
    public.write_text("TEST_ONLY", encoding="utf-8")
    if hang in {"heartbeat_stream_fail", "heartbeat_publish_fail"}:
        # These cases must reach the heartbeat transport. The legacy
        # invalid-key cases intentionally continue testing signing refusal.
        fake_openssl = fake_bin / "openssl"
        fake_openssl.write_text(
            f"#!{sys.executable}\n"
            "import pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "pathlib.Path(args[args.index('-out')+1]).write_bytes(b'SYNTHETIC_SIGNATURE')\n",
            encoding="utf-8",
        )
        fake_openssl.chmod(0o700)
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
        if hang in {"heartbeat_stream_fail", "heartbeat_publish_fail"}
        else 2
    )
    assert result.returncode == expected_status
    assert elapsed < 30
    actions = log.read_text(encoding="utf-8")
    if hang == "near_deadline_hung_stop":
        records = [
            (float(line.split(" ", 1)[0]), line.split(" ", 1)[1]) for line in actions.splitlines()
        ]
        approved_hard_deadline = datetime.fromisoformat(env["MOCK_CREATED"]).timestamp() + 7200
        deletes = [stamp for stamp, call in records if "api -X DELETE" in call]
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
    if hang in {"ssh", "copy", "heartbeat_copy_fail", "heartbeat_publish_fail"}:
        journal = tmp_path / "owner-private-recovery/synthetic-only-approved-name"
        failure = json.loads((journal / "owner-first-failure.json").read_text())
        expected_operation = {
            "ssh": "key_setup",
            "copy": "key_copy",
            "heartbeat_stream_fail": "heartbeat_publish",
            "heartbeat_publish_fail": "heartbeat_publish",
        }
        assert failure["operation"] == expected_operation[hang]
        assert failure["codespace_id"] == 12
        assert failure["created_at"] == env["MOCK_CREATED"]
        assert failure["exit_class"] == ("TIMEOUT" if hang in {"ssh", "copy"} else "FAILURE")
        reason = json.loads((journal / "owner-exit-reason.json").read_text())
        assert reason["original_exit_code"] == result.returncode
        assert reason["most_recent_recorded_failed_operation"] == failure["operation"]
        assert "TEST_ONLY" not in json.dumps(failure) + json.dumps(reason)
    # Stop returning or stalling never licenses running another host.
    assert "PRIVATE_CHAPTER" not in actions


@pytest.mark.parametrize(
    ("scenario", "tamper_archive", "stream_fault"),
    [
        ("brief_only", False, "none"),
        ("brief_only", False, "boundary_padding"),
        ("brief_only", True, "none"),
        ("inflight_empty", False, "none"),
        ("one_complete_one_failed", False, "none"),
        ("all_ready", False, "none"),
        ("all_ready", False, "truncated"),
        ("all_ready", False, "partial_exit"),
        ("all_ready", False, "disconnect"),
        ("all_ready", False, "wrong_receipt"),
        ("all_ready", False, "wrong_size"),
        ("all_ready", False, "extra"),
        ("all_ready", False, "matched_junk"),
        ("all_ready", False, "padding_flip"),
        ("all_ready", False, "oversize"),
        ("all_ready", False, "hung"),
        ("all_ready", False, "fork_ignores_term"),
        ("all_ready", False, "fork_signal"),
        ("all_ready", False, "startup_soft"),
        ("all_ready", False, "startup_hard"),
        ("all_ready", False, "signal"),
        ("all_ready", False, "foreign_identity"),
    ],
)
def test_owner_recovery_hash_verification_is_independent_of_rest_deletion(
    tmp_path: Path, scenario: str, tamper_archive: bool, stream_fault: str
) -> None:
    """No host: transfer TARs with incomplete and genuinely READY chapter checkpoints."""
    from clipper.stage_contracts import content_fingerprint

    source = tmp_path / "synthetic-private-outputs.tar"
    payloads: dict[str, bytes] = {}
    scoped: dict[int, list[dict[str, object]]] = {1: [], 7: []}
    safe_chapters: dict[str, dict[str, object]] = {}
    manifests: list[dict[str, object]] = []

    def add(number: int, kind: str, path: str, payload: bytes) -> None:
        if path in payloads:
            assert payloads[path] == payload
        payloads[path] = payload
        scoped[number].append(
            {
                "chapter": number,
                "kind": kind,
                "relative_path": path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )

    for number in (1, 7):
        if scenario == "inflight_empty":
            safe_chapters[str(number)] = {
                "recovery_state": "IN_FLIGHT",
                "last_subphase": "MINING_IN_FLIGHT",
                "runtime_journal_status": "MINING_IN_FLIGHT",
                "retrieval_status": "INVENTORY_VERIFIED_NOT_RETRIEVED",
                "last_verified_artifact_identities": {},
            }
            manifests.append({"chapter": number, "inventory_status": "VERIFIED", "files": []})
            continue
        if scenario == "brief_only" and number == 7:
            safe_chapters[str(number)] = {
                "recovery_state": "NEVER_STARTED",
                "last_subphase": "NOT_STARTED",
                "runtime_journal_status": "NOT_STARTED",
                "retrieval_status": "NOT_ATTEMPTED",
                "last_verified_artifact_identities": {},
            }
            manifests.append({"chapter": number, "inventory_status": "NOT_STARTED", "files": []})
            continue
        if scenario == "one_complete_one_failed" and number == 7:
            safe_chapters[str(number)] = {
                "recovery_state": "INCOMPLETE",
                "last_subphase": "UNVERIFIED",
                "runtime_journal_status": "UNVERIFIED",
                "retrieval_status": "INCOMPLETE",
                "last_verified_artifact_identities": {},
            }
            manifests.append({"chapter": number, "inventory_status": "INCOMPLETE", "files": []})
            continue
        if scenario == "brief_only":
            add(
                number,
                "text",
                "book-candidates/brief/synthetic/validated.json",
                b'{"only":"fabricated fixture text"}',
            )
            safe_chapters[str(number)] = {
                "recovery_state": "AWAITING_REVIEW",
                "last_subphase": "AWAITING_HUMAN_REVIEW",
                "runtime_journal_status": "AWAITING_HUMAN_REVIEW",
                "retrieval_status": "INVENTORY_VERIFIED_NOT_RETRIEVED",
                "last_verified_artifact_identities": {"brief_identity": "a" * 64},
            }
            manifests.append(
                {"chapter": number, "inventory_status": "VERIFIED", "files": scoped[number]}
            )
            continue

        assert scenario in {"one_complete_one_failed", "all_ready"}
        ids = {
            stage: hashlib.sha256(f"{number}-{stage}".encode()).hexdigest()
            for stage in (
                "brief_identity",
                "adaptation_identity",
                "script_identity",
                "direction_identity",
                "final_result_identity",
            )
        }
        audio_bytes = b"RIFF\x00\xff\x80PRIVATE_SYNTHETIC_CANARY\x00WAVE"
        wav_relative = "chapter-audio-injected_test_wav/shared.wav"
        wav_path = "book-candidates/" + wav_relative
        audio_report = {
            "schema": "clipper-chapter-narration-v2",
            "provenance": "INJECTED_TEST_WAV",
            "records": [
                {
                    "wav_relative_path": wav_relative,
                    "sha256": hashlib.sha256(audio_bytes).hexdigest(),
                }
            ],
        }
        ids["narration_report_identity"] = content_fingerprint(audio_report)
        for identity, stage in (
            ("brief_identity", "brief"),
            ("adaptation_identity", "adaptation"),
            ("script_identity", "chapter_script"),
            ("direction_identity", "chapter_direction"),
        ):
            add(
                number,
                "text",
                f"book-candidates/{stage}/{ids[identity]}/validated.json",
                json.dumps({"stage": stage, "chapter": number}, sort_keys=True).encode(),
            )
        add(number, "wav", wav_path, audio_bytes)
        add(
            number,
            "narration_report",
            f"book-candidates/narration-report-{ids['narration_report_identity']}.json",
            json.dumps(audio_report, sort_keys=True).encode(),
        )
        add(
            number,
            "final_result",
            f"book-candidates/chapter-evaluations/{number}/"
            f"chapter-result-{ids['final_result_identity']}.json",
            json.dumps({"chapter": number, "ready": True}, sort_keys=True).encode(),
        )
        add(
            number,
            "source_ledger",
            f"book-candidates/chapter-evaluations/{number}/source-ledger.json",
            json.dumps({"chapter": number, "source": "synthetic"}, sort_keys=True).encode(),
        )
        add(
            number,
            "human_review",
            f"book-candidates/chapter-evaluations/{number}/human-review.json",
            json.dumps({"chapter": number, "review": "synthetic"}, sort_keys=True).encode(),
        )
        manifest = {
            "schema": "clipper-chapter-private-output-manifest-v1",
            "chapter": number,
            "bundle_sha256": hashlib.sha256(f"bundle-{number}".encode()).hexdigest(),
            "scope_sha256": hashlib.sha256(f"scope-{number}".encode()).hexdigest(),
            "verified_artifact_identities": ids,
            "files": scoped[number],
            "publish_eligible": False,
        }
        raw_manifest = (json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n").encode()
        safe_chapters[str(number)] = {
            "recovery_state": "READY",
            "last_subphase": "PREPRODUCTION_READY",
            "runtime_journal_status": "PREPRODUCTION_READY",
            "retrieval_status": "INVENTORY_VERIFIED_NOT_RETRIEVED",
            "bundle_sha256": manifest["bundle_sha256"],
            "scope_sha256": manifest["scope_sha256"],
            "output_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "last_verified_artifact_identities": ids,
        }
        manifests.append({**manifest, "inventory_status": "VERIFIED"})

    aggregate: dict[str, dict[str, object]] = {}
    for number in (1, 7):
        for scoped_file in scoped[number]:
            path = str(scoped_file["relative_path"])
            if path not in aggregate:
                aggregate[path] = {
                    **{key: value for key, value in scoped_file.items() if key != "chapter"},
                    "chapters": [number],
                }
            else:
                assert aggregate[path]["sha256"] == scoped_file["sha256"]
                aggregate[path]["chapters"].append(number)

    safe = {
        "schema": "clipper-private-chapter-safe-recovery-v1",
        "status": "PRIVATE_CHAPTER_RECOVERY_CHECKPOINT_ONLY",
        "chapters": safe_chapters,
        "files_retrieved": False,
        "publish_eligible": False,
    }
    safe_bytes = json.dumps(safe, sort_keys=True).encode()
    listing = {
        "schema": "clipper-chapter-private-recovery-archive-manifest-v1",
        "chapters": manifests,
        "files": list(aggregate.values()),
        "runtime_status_sha256": hashlib.sha256(safe_bytes).hexdigest(),
    }
    listing_bytes = json.dumps(listing, sort_keys=True).encode()

    def write_fixture_tar() -> None:
        with tarfile.open(source, "w") as archive:
            for name, data in (
                *payloads.items(),
                ("private-recovery-manifest.json", listing_bytes),
                ("safe-recovery-status.json", safe_bytes),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o600
                archive.addfile(member, io.BytesIO(data))

    write_fixture_tar()
    if stream_fault == "boundary_padding":
        # Force the last member to end at offset 9728 modulo 10240.
        # Python tarfile writes two EOF blocks plus record padding: 10752 bytes.
        with tarfile.open(source, "r:") as archive:
            delta = (9728 - archive.offset % 10240) % 10240
        relative = "book-candidates/brief/synthetic/validated.json"
        payloads[relative] += b"Z" * delta
        new_hash = hashlib.sha256(payloads[relative]).hexdigest()
        for row in (scoped[1][0], aggregate[relative]):
            row["bytes"] = len(payloads[relative])
            row["sha256"] = new_hash
        listing_bytes = json.dumps(listing, sort_keys=True).encode()
        write_fixture_tar()
        with tarfile.open(source, "r:") as archive:
            assert source.stat().st_size - archive.offset == 10752
    receipt = {
        "schema": "clipper-private-chapter-recovery-created-v1",
        "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "archive_bytes": source.stat().st_size,
        "manifest_sha256": hashlib.sha256(listing_bytes).hexdigest(),
    }
    response = tmp_path / "synthetic-remote-recovery-result.json"
    response.write_text(json.dumps(receipt), encoding="utf-8")
    if scenario in {"one_complete_one_failed", "all_ready"}:
        # Exercise the owner's EXACT bounded Python TAR verifier in isolation;
        # the shell's fallback intentionally hides its private error details.
        content = _SCRIPT.read_text(encoding="utf-8")
        core = content.split("if critical archive_verification python3 -c '\n", 1)[1].split(
            '\n\' "$session/recovery-created.json"', 1
        )[0]
        direct = subprocess.run(
            [
                sys.executable,
                "-c",
                core,
                str(response),
                str(source),
                str(tmp_path / "direct-retrieval-report.json"),
                str(tmp_path / "direct-safe-status.json"),
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        assert direct.returncode == 0, (direct.stderr, direct.stdout)
    fake_bin = tmp_path / "commands"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    remote_archive = (
        tmp_path / "synthetic-host-root/book-candidates/private-host/chapter-recovery.tar"
    )
    remote_archive.parent.mkdir(parents=True)
    remote_archive.write_bytes(source.read_bytes() + (b"corrupted" if tamper_archive else b""))
    if stream_fault == "padding_flip":
        damaged = bytearray(source.read_bytes())
        assert damaged[-1] == 0
        damaged[-1] = 0x81
        remote_archive.write_bytes(damaged)
        receipt["archive_sha256"] = hashlib.sha256(damaged).hexdigest()
        response.write_text(json.dumps(receipt), encoding="utf-8")
    if stream_fault == "wrong_receipt":
        receipt["archive_sha256"] = "0" * 64
        response.write_text(json.dumps(receipt), encoding="utf-8")
    if stream_fault == "oversize":
        # No declared size: exercise the receiver's unconditional 800MB cap
        # through a scaled-down test-only threshold, not preflight rejection.
        receipt.pop("archive_bytes")
        response.write_text(json.dumps(receipt), encoding="utf-8")
    if stream_fault == "wrong_size":
        receipt["archive_bytes"] = source.stat().st_size - 100
        response.write_text(json.dumps(receipt), encoding="utf-8")
    if stream_fault == "matched_junk":
        receipt["archive_sha256"] = hashlib.sha256(
            source.read_bytes() + b"junk-after-tar"
        ).hexdigest()
        receipt["archive_bytes"] = source.stat().st_size + len(b"junk-after-tar")
        response.write_text(json.dumps(receipt), encoding="utf-8")
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import datetime,hashlib,json,os,pathlib,signal,shutil,subprocess,sys,time\n"
        "a=sys.argv[1:]\n"
        "with open(os.environ['MOCK_LOG'],'a') as f: f.write(' '.join(a)+'\\n')\n"
        "if a[:2]==['api','-X']:\n"
        "  if a[2]=='POST': pathlib.Path(os.environ['MOCK_STOP']).write_text('stopped')\n"
        "  sys.exit(0)\n"
        "if a[:2]==['api','-i']:\n"
        "  print('HTTP/2 404 Not Found'); sys.exit(0)\n"
        "if a[:1]==['api']:\n"
        "  state='Stopped' if pathlib.Path(os.environ['MOCK_STOP']).exists() else 'Available'\n"
        "  foreign=os.environ['MOCK_STREAM_FAULT']=='foreign_identity' and pathlib.Path(os.environ['MOCK_FOREIGN_MARK']).exists()\n"
        "  print(json.dumps({'id':999 if foreign else 12,'name':os.environ['CODESPACE_NAME'],"
        "'state':state,'created_at':os.environ['MOCK_CREATED'],"
        "'repository':{'full_name':'TheBayoumi/clipper'},"
        "'machine':{'cpus':4,'memory_in_bytes':17179869184,"
        "'storage_in_bytes':34359738368}})); sys.exit(0)\n"
        "if a[:2]==['codespace','ssh']:\n"
        "  if 'clipper.books.chapter_host status' in ' '.join(a):\n"
        "    print(pathlib.Path(os.environ['MOCK_SAFE_STATUS']).read_text()); sys.exit(0)\n"
        "  if 'exec cat -- ' in a[-1]:\n"
        "    assert '-T' in a, a\n"
        "    fault=os.environ['MOCK_STREAM_FAULT']\n"
        "    pathlib.Path(os.environ['MOCK_TRANSFER_PID']).write_text(str(os.getpid()))\n"
        "    if fault in ('fork_ignores_term','fork_signal'):\n"
        "      pid=os.fork()\n"
        "      if pid==0:\n"
        "        signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "        pathlib.Path(os.environ['MOCK_GRANDCHILD_PID']).write_text(str(os.getpid()))\n"
        "        while True: time.sleep(1)\n"
        "      pathlib.Path(os.environ['MOCK_STREAM_ENTERED']).touch()\n"
        "      time.sleep(120)\n"
        "    if fault in ('hung','signal'):\n"
        "      pathlib.Path(os.environ['MOCK_STREAM_ENTERED']).touch()\n"
        "      time.sleep(120)\n"
        "    if fault=='disconnect': sys.exit(3)\n"
        "    completed=subprocess.run(['bash','-c',a[-1]],capture_output=True,check=False)\n"
        "    data=completed.stdout\n"
        "    if os.environ['MOCK_TAMPER']=='true':\n"
        "      data+=b'CORRUPTED_SECOND_SSH_STREAM'\n"
        "      pathlib.Path(os.environ['MOCK_TAMPERED_STREAM_SHA']).write_text(hashlib.sha256(data).hexdigest())\n"
        "    if fault=='truncated': data=data[:-17]\n"
        "    if fault=='partial_exit': sys.stdout.buffer.write(data[:137]); sys.exit(7)\n"
        "    if fault in ('extra','matched_junk'): data+=b'junk-after-tar'\n"
        "    if fault=='oversize': data+=b'X'*10000\n"
        "    sys.stdout.buffer.write(data); sys.exit(completed.returncode)\n"
        "  if ' recover ' in (' '+ ' '.join(a)+' '):\n"
        "    if os.environ['MOCK_STREAM_FAULT']=='foreign_identity':\n"
        "      pathlib.Path(os.environ['MOCK_FOREIGN_MARK']).touch()\n"
        "    print(pathlib.Path(os.environ['MOCK_RECEIPT']).read_text())\n"
        "  sys.exit(0)\n"
        "if a[:2]==['codespace','cp']:\n"
        "  assert 'chapter-recovery.tar' not in ' '.join(a)\n"
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
    safe_fixture = tmp_path / "synthetic-safe-phase-only.json"
    safe_fixture.write_bytes(safe_bytes)
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
            "MOCK_TAMPERED_STREAM_SHA": str(tmp_path / "tampered-stream-sha"),
            "MOCK_STREAM_FAULT": stream_fault,
            "MOCK_FOREIGN_MARK": str(tmp_path / "foreign-host-id"),
            "MOCK_STREAM_ENTERED": str(tmp_path / "stream-entered"),
            "MOCK_MAX_STAGED_SIZE": str(tmp_path / "max-staged-size"),
            "MOCK_TRANSFER_PID": str(tmp_path / "transfer-pid"),
            "MOCK_SAFE_STATUS": str(safe_fixture),
            "MOCK_GRANDCHILD_PID": str(tmp_path / "grandchild-pid"),
        }
    )
    env.pop("CODESPACES", None)
    script = _SCRIPT
    if stream_fault == "oversize":
        # Smaller test-only cap and 1KiB blocks exercise *midstream* writes
        # without provisioning hundreds of MB of synthetic private material.
        script = tmp_path / "bounded-supervisor.sh"
        raw = _SCRIPT.read_text(encoding="utf-8")
        assert "limit = 800_000_000" in raw
        assert "1024 * 1024, limit - used + 1" in raw
        raw = raw.replace("limit = 800_000_000", "limit = 4096", 1)
        raw = raw.replace("1024 * 1024, limit - used + 1", "1024, limit - used + 1", 1)
        bound_probe = "if used + len(block) > limit:\n                sys.exit(2)"
        assert bound_probe in raw
        raw = raw.replace(
            bound_probe,
            "if used + len(block) > limit:\n"
            '                open(os.environ["MOCK_MAX_STAGED_SIZE"], "w").write(str(used))\n'
            "                sys.exit(2)",
            1,
        )
        script.write_text(raw)
    if stream_fault in {"startup_soft", "startup_hard"}:
        # The launcher OS process exists, but the receiver has not yet
        # obtained the Popen return value. Delay that exact registration
        # boundary, then simulate TERM or unconditional hard KILL.
        script = tmp_path / "startup-boundary-supervisor.sh"
        raw = _SCRIPT.read_text(encoding="utf-8")
        start = 'try:\n    proc = subprocess.Popen(\n        [sys.executable, "-c", launcher_code, name, remote],'
        assert start in raw
        wrapper = (
            "original_popen = subprocess.Popen\n"
            "def delayed_popen(*args, **kwargs):\n"
            "    launched = original_popen(*args, **kwargs)\n"
            "    if args[0][0] == sys.executable:\n"
            '        open(os.environ["MOCK_STARTUP_LAUNCHER_PID"], "w").write(str(launched.pid))\n'
            "        time.sleep(1.5)\n"
            "    return launched\n"
            "subprocess.Popen = delayed_popen\n"
        )
        raw = raw.replace(start, wrapper + start, 1)
        boundary = "critical recovery_copy python3 -c '"
        assert boundary in raw
        kill_type = "KILL" if stream_fault == "startup_hard" else "TERM"
        raw = raw.replace(
            boundary,
            f"critical recovery_copy timeout --signal={kill_type} --kill-after=2s 1s python3 -c '",
            1,
        )
        script.write_text(raw)
        env["MOCK_STARTUP_LAUNCHER_PID"] = str(tmp_path / "startup-launcher-pid")
    if stream_fault in {"signal", "fork_signal"}:
        import signal

        child = subprocess.Popen(
            ["bash", str(script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            marker = tmp_path / "stream-entered"
            until = time.monotonic() + 15
            while not marker.exists() and child.poll() is None and time.monotonic() < until:
                time.sleep(0.025)
            assert marker.exists() and child.poll() is None
            child.send_signal(signal.SIGTERM)
            stdout, stderr = child.communicate(timeout=30)
            result = subprocess.CompletedProcess([str(script)], child.returncode, stdout, stderr)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
    else:
        result = subprocess.run(
            ["bash", str(script)], env=env, capture_output=True, text=True, timeout=30
        )
    assert result.returncode != 0  # Invalid synthetic signer prevents host setup.
    if stream_fault in {"startup_soft", "startup_hard"}:
        launcher_pid = int((tmp_path / "startup-launcher-pid").read_text())
        assert "exec cat -- " not in log.read_text(encoding="utf-8")

        # No producer may launch before registration. Even SIGKILL during
        # startup must close the parent-death pipe and retire the launcher.
        def launcher_live() -> bool:
            stat = Path(f"/proc/{launcher_pid}/stat")
            return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"

        until = time.monotonic() + 2
        while launcher_live() and time.monotonic() < until:
            time.sleep(0.02)
        assert not launcher_live(), "unregistered startup launcher survived"
    if stream_fault in {"fork_ignores_term", "fork_signal"}:
        grandchild = tmp_path / "grandchild-pid"
        assert grandchild.is_file()
        pid = int(grandchild.read_text())

        # Linux runner may briefly retain a reparented zombie; that is not a
        # live descendant. A TERM-ignoring running process is a failure.
        def running() -> bool:
            stat = Path(f"/proc/{pid}/stat")
            return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"

        until = time.monotonic() + 2
        while running() and time.monotonic() < until:
            time.sleep(0.02)
        assert not running(), f"SSH descendant {pid} survived owner timeout/signal"
    actions = log.read_text(encoding="utf-8")
    if stream_fault == "foreign_identity":
        assert "exec cat -- " not in actions
        assert "api -X DELETE" not in actions and "api -X POST" not in actions
        assert not list((tmp_path / "owner-private-recovery").rglob("chapter-recovery.tar"))
        return
    assert "api -X POST" in actions and "api -X DELETE" in actions
    assert "codespace cp" not in "\n".join(
        call for call in actions.splitlines() if "chapter-recovery.tar" in call
    )
    if stream_fault not in {"startup_soft", "startup_hard"}:
        assert " -T " in actions and "exec cat -- " in actions
    folder = tmp_path / "owner-private-recovery/synthetic-private-host"
    verification = json.loads((folder / "retrieval-verification.json").read_bytes())
    cleanup = json.loads((folder / "host-cleanup-verification.json").read_bytes())
    assert cleanup["host_stop_verified"] is True
    assert cleanup["host_deletion_verified"] is True
    assert verification["retrieval_status"] == (
        "INCOMPLETE"
        if tamper_archive or stream_fault not in {"none", "boundary_padding"}
        else "VERIFIED_PRIVATE_ARCHIVE"
        if scenario == "all_ready"
        else "VERIFIED_PRIVATE_ARCHIVE_PARTIAL"
    ), (result.stderr, verification)
    if not tamper_archive and stream_fault in {"none", "boundary_padding"}:
        expected = {
            "brief_only": ({"1": "PARTIAL_OUTPUTS", "7": "NOT_STARTED"}, "PARTIAL"),
            "inflight_empty": ({"1": "CHECKPOINT_ONLY", "7": "CHECKPOINT_ONLY"}, "NONE"),
            "one_complete_one_failed": (
                {"1": "READY_OUTPUTS_VERIFIED", "7": "INCOMPLETE"},
                "PARTIAL",
            ),
            "all_ready": (
                {"1": "READY_OUTPUTS_VERIFIED", "7": "READY_OUTPUTS_VERIFIED"},
                "COMPLETE",
            ),
        }
        assert verification["archive_integrity_status"] == "VERIFIED"
        assert verification["chapters"] == expected[scenario][0]
        assert verification["output_scope"] == expected[scenario][1]
        assert verification["host_cleanup_status"] == "NOT_YET_REST_VERIFIED"
        assert verification["recovered_file_count"] == len(aggregate)
        assert verification["retrieval_status"] == (
            "VERIFIED_PRIVATE_ARCHIVE"
            if scenario == "all_ready"
            else "VERIFIED_PRIVATE_ARCHIVE_PARTIAL"
        )
        if scenario == "inflight_empty":
            assert verification["recovered_file_count"] == 0
        if scenario == "all_ready":
            with tarfile.open(folder / "chapter-recovery.tar", "r:") as copied:
                assert (
                    copied.getnames().count(
                        "book-candidates/chapter-audio-injected_test_wav/shared.wav"
                    )
                    == 1
                )
    assert (folder / "chapter-recovery.tar").exists() is (
        not tamper_archive and stream_fault in {"none", "boundary_padding"}
    )
    assert not list(folder.glob(".recovery-stage.*"))
    if stream_fault == "oversize":
        assert int((tmp_path / "max-staged-size").read_text()) == 4096
    if stream_fault in {"hung", "signal", "oversize", "partial_exit"}:
        transport_pid = tmp_path / "transfer-pid"
        if transport_pid.exists():
            try:
                os.kill(int(transport_pid.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                pytest.fail("recovery stream left a live gh subprocess")
    if stream_fault in {"truncated", "partial_exit", "wrong_receipt", "disconnect"}:
        assert "clipper.books.chapter_host status" in actions
        assert json.loads((folder / "safe-recovery-status.json").read_bytes()) == safe
        assert "api -X DELETE" in actions
    assert "fabricated fixture text" not in result.stdout + result.stderr
    assert "PRIVATE_SYNTHETIC_CANARY" not in result.stdout + result.stderr
    assert (folder.stat().st_mode & 0o077) == 0
    for artifact in (folder / "chapter-recovery.tar", folder / "retrieval-verification.json"):
        if artifact.exists():
            assert (artifact.stat().st_mode & 0o077) == 0
    if scenario == "brief_only" and not tamper_archive and stream_fault == "none":
        # A second *valid* prepared archive must never break an older
        # three-file snapshot if promotion fails after the bundle rename.
        baseline = {
            name: (folder / name).read_bytes()
            for name in (
                "chapter-recovery.tar",
                "retrieval-verification.json",
                "safe-recovery-status.json",
            )
        }
        previous_pointer = os.readlink(folder / "recovery-current")
        injected = tmp_path / "promotion-fails-supervisor.sh"
        original_supervisor = _SCRIPT.read_text(encoding="utf-8")
        injection_point = "os.replace(stage, root / bundle)\n# This is the sole commit point"
        assert injection_point in original_supervisor
        injected.write_text(
            original_supervisor.replace(
                injection_point,
                "os.replace(stage, root / bundle)\nraise SystemExit(91)\n# This is the sole commit point",
                1,
            ),
            encoding="utf-8",
        )
        stopped = Path(env["MOCK_STOP"])
        stopped.unlink(missing_ok=True)
        failed_promotion = subprocess.run(
            ["bash", str(injected)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert failed_promotion.returncode != 0
        assert "api -X DELETE" in log.read_text()
        assert os.readlink(folder / "recovery-current") == previous_pointer
        assert all((folder / name).read_bytes() == body for name, body in baseline.items())
        assert "PRIVATE_SYNTHETIC_CANARY" not in (failed_promotion.stdout + failed_promotion.stderr)
        # Interrupt the actual Bash supervisor with SIGTERM after the new
        # immutable directory is renamed but before the atomic pointer swap.
        promotion_signal = tmp_path / "promotion-signal-supervisor.sh"
        mark = tmp_path / "promotion-renamed"
        env["MOCK_PROMOTION_MARK"] = str(mark)
        promotion_signal.write_text(
            original_supervisor.replace(
                injection_point,
                "os.replace(stage, root / bundle)\n"
                'pathlib.Path(os.environ["MOCK_PROMOTION_MARK"]).touch()\n'
                "import time\ntime.sleep(120)\n"
                "# This is the sole commit point",
                1,
            ),
            encoding="utf-8",
        )
        stopped.unlink(missing_ok=True)
        import signal

        interrupted = subprocess.Popen(
            ["bash", str(promotion_signal)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            until = time.monotonic() + 15
            while not mark.exists() and interrupted.poll() is None and time.monotonic() < until:
                time.sleep(0.025)
            assert mark.exists() and interrupted.poll() is None
            interrupted.send_signal(signal.SIGTERM)
            signal_out, signal_err = interrupted.communicate(timeout=30)
            assert interrupted.returncode != 0
        finally:
            if interrupted.poll() is None:
                interrupted.kill()
                interrupted.communicate(timeout=5)
        assert os.readlink(folder / "recovery-current") == previous_pointer
        assert all((folder / name).read_bytes() == body for name, body in baseline.items())
        assert "api -X DELETE" in log.read_text()
        assert "PRIVATE_SYNTHETIC_CANARY" not in signal_out + signal_err
        # Drive the REAL supervisor loop through one successful heartbeat,
        # a verified owner-private partial archive, then a later failed
        # heartbeat. Accelerate only checkpoint schedule in a temp shell copy.
        fake_openssl = fake_bin / "openssl"
        fake_openssl.write_text(
            f"#!{sys.executable}\n"
            "import pathlib,sys\n"
            "a=sys.argv[1:]\n"
            "pathlib.Path(a[a.index('-out')+1]).write_bytes(b'SYNTHETIC_TEST_SIGNATURE')\n",
            encoding="utf-8",
        )
        fake_openssl.chmod(0o700)
        safe_path = tmp_path / "synthetic-phase-status.json"
        safe_path.write_text(json.dumps(safe), encoding="utf-8")
        gh_body = fake_gh.read_text(encoding="utf-8")
        gh_body = gh_body.replace(
            "if a[:2]==['codespace','ssh']:\n",
            "if a[:2]==['codespace','ssh']:\n"
            "  if 'clipper.books.chapter_host status' in ' '.join(a):\n"
            "    print(pathlib.Path(os.environ['MOCK_SAFE_STATUS']).read_text()); sys.exit(0)\n",
        )
        gh_body = gh_body.replace(
            "if a[:2]==['codespace','ssh']:\n",
            "if a[:2]==['codespace','ssh']:\n"
            "  if 'heartbeat-envelope.watchdog.pending' in ' '.join(a):\n"
            "    path=pathlib.Path(os.environ['MOCK_HEARTBEAT_COUNT'])\n"
            "    count=int(path.read_text())+1 if path.exists() else 1\n"
            "    path.write_text(str(count))\n"
            "    if count>=2: sys.exit(3)\n"
            "    import hashlib\n"
            "    print('ACK:'+hashlib.sha256(sys.stdin.buffer.read()).hexdigest()); sys.exit(0)\n",
        )
        fake_gh.write_text(gh_body, encoding="utf-8")
        env["MOCK_SAFE_STATUS"] = str(safe_path)
        counter = tmp_path / "synthetic-heartbeat-counter"
        env["MOCK_HEARTBEAT_COUNT"] = str(counter)
        controlled = tmp_path / "synthetic-checkpoint-supervisor.sh"
        original = _SCRIPT.read_text(encoding="utf-8")
        assert 'last_checkpoint_epoch="$last_phase_poll_epoch"' in original
        controlled.write_text(
            original.replace(
                'last_checkpoint_epoch="$last_phase_poll_epoch"', "last_checkpoint_epoch=0", 1
            ).replace('last_phase_poll_epoch="$(date -u +%s)"', "last_phase_poll_epoch=0", 1),
            encoding="utf-8",
        )
        stopped = Path(env["MOCK_STOP"])
        stopped.unlink(missing_ok=True)
        first = subprocess.run(
            ["bash", str(controlled)],
            env=env,
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        assert first.returncode != 0
        pointer_path = folder / "latest-verified-snapshot.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        assert pointer["retrieval_status"] == "VERIFIED_OWNER_PRIVATE_SNAPSHOT"
        assert pointer["archive_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        verified = folder / pointer["snapshot_dir"]
        assert (verified / "chapter-recovery.tar").read_bytes() == source.read_bytes()
        assert (verified / "retrieval-verification.json").is_file()
        safe_phase = json.loads((folder / "owner-safe-phase.json").read_text())
        assert safe_phase["chapters"] == {
            "1": "AWAITING_HUMAN_REVIEW",
            "7": "NOT_STARTED",
        }
        assert safe_phase["codespace_id"] == 12
        exit_reason = json.loads((folder / "owner-exit-reason.json").read_text())
        assert exit_reason["codespace_id"] == 12
        # The initial synthetic run deliberately used an invalid signer
        # before this run; its persistent last-failure record may be older.
        # Prove the exact heartbeat-publication failure in a fresh isolated attempt
        # below rather than attributing the previous failure to this run.
        assert "fabricated fixture text" not in first.stdout + first.stderr
        # A failed *next* checkpoint must not replace or damage the verified
        # owner-private first snapshot. REST DELETE remains independent.
        stopped.unlink(missing_ok=True)
        counter.unlink(missing_ok=True)
        env["MOCK_TAMPER"] = "true"
        second = subprocess.run(
            ["bash", str(controlled)],
            env=env,
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        assert second.returncode != 0
        tampered_hash = (tmp_path / "tampered-stream-sha").read_text().strip()
        assert tampered_hash != receipt["archive_sha256"]
        assert tampered_hash != hashlib.sha256(source.read_bytes()).hexdigest()
        assert json.loads(pointer_path.read_text()) == pointer
        assert (verified / "chapter-recovery.tar").read_bytes() == source.read_bytes()
        assert "api -X DELETE" in log.read_text()
        assert "fabricated fixture text" not in second.stdout + second.stderr

        # Distinguishing transport failure: start the GH heartbeat counter at
        # one, so the first attempted heartbeat CP in THIS invocation fails.
        # Remove only the previous synthetic attempt's last-failure record;
        # this makes the named operation impossible to inherit accidentally.
        counter.write_text("1", encoding="utf-8")
        stopped.unlink(missing_ok=True)
        (folder / "owner-last-failure.json").unlink(missing_ok=True)
        env["MOCK_TAMPER"] = "false"
        transport_before = log.read_text(encoding="utf-8").count(
            "heartbeat-envelope.watchdog.pending"
        )
        transport_failed = subprocess.run(
            ["bash", str(controlled)],
            env=env,
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        assert transport_failed.returncode != 0
        assert (
            log.read_text(encoding="utf-8").count("heartbeat-envelope.watchdog.pending")
            > transport_before
        )
        latest_failure = json.loads((folder / "owner-last-failure.json").read_text())
        assert latest_failure["operation"] == "heartbeat_publish", (
            transport_failed.stderr,
            latest_failure,
        )
        assert latest_failure["exit_code"] == 3
        copy_exit = json.loads((folder / "owner-exit-reason.json").read_text())
        assert copy_exit["most_recent_recorded_failed_operation"] == "heartbeat_publish"
        assert copy_exit["original_exit_code"] == transport_failed.returncode
        assert json.loads(pointer_path.read_text()) == pointer
        assert (verified / "chapter-recovery.tar").read_bytes() == source.read_bytes()
        assert "api -X DELETE" in log.read_text()
        assert "fabricated fixture text" not in transport_failed.stdout + transport_failed.stderr

        # A legitimate, integrity-verified status-only EMPTY TAR is not an
        # output checkpoint. It must not supersede the earlier real brief.
        empty_safe = {
            "schema": "clipper-private-chapter-safe-recovery-v1",
            "status": "PRIVATE_CHAPTER_RECOVERY_CHECKPOINT_ONLY",
            "chapters": {
                str(n): {
                    "recovery_state": "IN_FLIGHT",
                    "last_subphase": "MINING_IN_FLIGHT",
                    "runtime_journal_status": "MINING_IN_FLIGHT",
                    "retrieval_status": "INVENTORY_VERIFIED_NOT_RETRIEVED",
                    "last_verified_artifact_identities": {},
                }
                for n in (1, 7)
            },
            "files_retrieved": False,
            "publish_eligible": False,
        }
        empty_safe_bytes = json.dumps(empty_safe, sort_keys=True).encode()
        empty_listing = {
            "schema": "clipper-chapter-private-recovery-archive-manifest-v1",
            "chapters": [
                {"chapter": n, "inventory_status": "VERIFIED", "files": []} for n in (1, 7)
            ],
            "files": [],
            "runtime_status_sha256": hashlib.sha256(empty_safe_bytes).hexdigest(),
        }
        empty_listing_bytes = json.dumps(empty_listing, sort_keys=True).encode()
        empty_tar = tmp_path / "synthetic-empty-status-only.tar"
        with tarfile.open(empty_tar, "w") as archive:
            for name, raw in (
                ("private-recovery-manifest.json", empty_listing_bytes),
                ("safe-recovery-status.json", empty_safe_bytes),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                member.mode = 0o600
                archive.addfile(member, io.BytesIO(raw))
        empty_receipt = tmp_path / "synthetic-empty-receipt.json"
        empty_receipt.write_text(
            json.dumps(
                {
                    "schema": "clipper-private-chapter-recovery-created-v1",
                    "archive_sha256": hashlib.sha256(empty_tar.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(empty_listing_bytes).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        env["MOCK_ARCHIVE"] = str(empty_tar)
        env["MOCK_RECEIPT"] = str(empty_receipt)
        env["MOCK_TAMPER"] = "false"
        direct_before_empty = {
            name: (folder / name).read_bytes()
            for name in (
                "chapter-recovery.tar",
                "retrieval-verification.json",
                "safe-recovery-status.json",
            )
        }
        pointer_before_empty = os.readlink(folder / "recovery-current")
        stopped.unlink(missing_ok=True)
        counter.unlink(missing_ok=True)
        empty_run = subprocess.run(
            ["bash", str(controlled)],
            env=env,
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        assert empty_run.returncode != 0
        assert json.loads(pointer_path.read_text()) == pointer
        assert (verified / "chapter-recovery.tar").read_bytes() == source.read_bytes()
        assert os.readlink(folder / "recovery-current") == pointer_before_empty
        assert all(
            (folder / name).read_bytes() == body for name, body in direct_before_empty.items()
        )
        assert "api -X DELETE" in log.read_text()
        assert "fabricated fixture text" not in empty_run.stdout + empty_run.stderr

        # Two individually successful 7-second optional I/O calls previously
        # compounded with a fixed 10-second sleep. The real shell must renew
        # its signed heartbeat without allowing this work to consume freshness.
        env["MOCK_ARCHIVE"] = str(source)
        env["MOCK_RECEIPT"] = str(response)
        stopped.unlink(missing_ok=True)
        counter.unlink(missing_ok=True)
        timed = fake_gh.read_text(encoding="utf-8")
        timed = timed.replace(
            "import datetime,json,os,pathlib,shutil,sys\n",
            "import datetime,json,os,pathlib,shutil,sys,time\n",
        )
        timed = timed.replace(
            "if 'clipper.books.chapter_host status' in ' '.join(a):\n",
            "if 'clipper.books.chapter_host status' in ' '.join(a):\n    time.sleep(7)\n",
        )
        timed = timed.replace(
            "  if 'chapter-recovery.tar' in ' '.join(a):\n",
            "  if 'chapter-recovery.tar' in ' '.join(a):\n    time.sleep(7)\n",
        )
        timed = timed.replace(
            "    path.write_text(str(count))\n",
            "    path.write_text(str(count))\n"
            "    with open(os.environ['MOCK_HEARTBEAT_STAMPS'],'a') as stamp:\n"
            "      stamp.write(str(time.time())+'\\n')\n",
        )
        fake_gh.write_text(timed, encoding="utf-8")
        stamps = tmp_path / "heartbeat-copy-monotonic-stamps"
        env["MOCK_HEARTBEAT_STAMPS"] = str(stamps)
        slow_run = subprocess.run(
            ["bash", str(controlled)],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert slow_run.returncode != 0
        heartbeat_times = [float(v) for v in stamps.read_text().splitlines()]
        assert len(heartbeat_times) >= 2
        assert 0 < heartbeat_times[1] - heartbeat_times[0] < 20
        assert json.loads(pointer_path.read_text()) == pointer
        assert "api -X DELETE" in log.read_text()
        assert "fabricated fixture text" not in slow_run.stdout + slow_run.stderr


@pytest.mark.parametrize("mode", ["inspect", "brief", "resume", "hybrid-brief", "hybrid-resume"])
def test_actual_codespace_wrapper_dispatches_all_modes_through_shared_preflight(
    tmp_path: Path, mode: str
) -> None:
    """A subprocess executes the REAL wrapper; all GitHub/host commands are local stubs."""
    host_script = Path(__file__).resolve().parents[1] / "scripts/private_chapter_codespace.sh"
    fake_bin = tmp_path / "fake-codespace-bin"
    fake_bin.mkdir()
    operations = tmp_path / "codespace-subprocess-log.jsonl"
    for command, source in {
        "git": (
            "import os,sys\n"
            "args=sys.argv[1:]\n"
            "if args==['rev-parse','--abbrev-ref','HEAD']: print('review/unified-tts-cleanup')\n"
            "elif args==['rev-parse','HEAD']: print(os.environ['MOCK_CHECKOUT_SHA'])\n"
            "elif args==['rev-parse','--show-toplevel']: print(os.environ['MOCK_REPO_ROOT'])\n"
            "else: sys.exit(2)\n"
        ),
        "gh": (
            "import json,os,sys\n"
            "if sys.argv[1:]!=['api','user/codespaces/'+os.environ['CODESPACE_NAME']]: sys.exit(2)\n"
            "print(json.dumps({'name':os.environ['CODESPACE_NAME'],"
            "'created_at':os.environ['MOCK_CREATED'],"
            "'repository':{'full_name':'TheBayoumi/clipper'},"
            "'machine':{'cpus':4,'memory_in_bytes':17179869184,"
            "'storage_in_bytes':int(os.environ['MOCK_STORAGE'])}}))\n"
        ),
        "python": (
            "import json,os,sys\n"
            "args=sys.argv[1:]\n"
            "if args[:1]==['-c']: os.execv(sys.executable,[sys.executable,*args])\n"
            "with open(os.environ['MOCK_OPERATIONS'],'a',encoding='utf-8') as f:\n"
            "  f.write(json.dumps(args)+'\\n')\n"
            "if args[:3]==['-m','clipper.books.chapter_lifecycle','--artifact-root']:\n"
            "  if int(os.environ['MOCK_LIFECYCLE_STATUS']): sys.exit(2)\n"
            "  import datetime\n"
            "  created=datetime.datetime.fromisoformat(os.environ['MOCK_CREATED'])\n"
            "  age=(datetime.datetime.now(datetime.timezone.utc)-created).total_seconds()\n"
            "  if age>6900 and os.environ.get('MOCK_SIGNED_GRANT')!='true': sys.exit(2)\n"
            "  print('6800'); sys.exit(0)\n"
            "if args[:3]==['-m','clipper.books.chapter_host',os.environ['MOCK_MODE']]:\n"
            "  sys.exit(int(os.environ['MOCK_HOST_STATUS']))\n"
            "sys.exit(2)\n"
        ),
    }.items():
        executable = fake_bin / command
        executable.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
        executable.chmod(0o700)
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(fake_bin) + os.pathsep + env["PATH"],
            "CODESPACES": "true",
            "CODESPACE_NAME": "synthetic-approved-machine",
            "PRIVATE_CHAPTER_APPROVED_SHA": "a" * 40,
            "PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256": "b" * 64,
            "PRIVATE_CHAPTER_ROOT": str(tmp_path / "private-root"),
            "PRIVATE_CHAPTER_MANIFEST": str(tmp_path / "private-root/two-chapters.json"),
            "MOCK_REPO_ROOT": str(tmp_path / "fake-checkout"),
            "MOCK_CHECKOUT_SHA": "a" * 40,
            "MOCK_CREATED": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            "MOCK_STORAGE": "34359738368",
            "MOCK_OPERATIONS": str(operations),
            "MOCK_MODE": mode,
            "MOCK_LIFECYCLE_STATUS": "0",
            "MOCK_HOST_STATUS": "0",
        }
    )

    def invoke(*, selected: str = mode) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(host_script), selected],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    good = invoke()
    assert good.returncode == 0, (good.stderr, good.stdout)
    recorded = [json.loads(line) for line in operations.read_text().splitlines()]
    assert len(recorded) == 2
    assert recorded[0][:2] == ["-m", "clipper.books.chapter_lifecycle"]
    assert recorded[1][:3] == ["-m", "clipper.books.chapter_host", mode]
    assert "--codespace-created-at" in recorded[1]
    assert recorded[1][recorded[1].index("--artifact-root") + 1] == env["PRIVATE_CHAPTER_ROOT"]
    assert recorded[1][recorded[1].index("--manifest") + 1] == env["PRIVATE_CHAPTER_MANIFEST"]
    operations.unlink()

    # Unknown modes fail before touching REST, lifecycle or host commands.
    rejected = invoke(selected="not-a-mode")
    assert rejected.returncode == 2 and not operations.exists()
    # A mismatched SHA, external heartbeat, REST shape, or original deadline
    # cannot dispatch even the approved hybrid or legacy modes.
    for mutation, value in (
        ("MOCK_CHECKOUT_SHA", "c" * 40),
        ("MOCK_STORAGE", "1"),
        ("MOCK_CREATED", (datetime.now(UTC) - timedelta(minutes=116)).isoformat()),
        ("MOCK_LIFECYCLE_STATUS", "2"),
        ("CODESPACES", "false"),
    ):
        original = env[mutation]
        env[mutation] = value
        blocked = invoke()
        assert blocked.returncode != 0
        if mutation not in {"MOCK_LIFECYCLE_STATUS", "MOCK_CREATED"}:
            assert not operations.exists(), mutation
        else:
            # REST creation timestamps are validated inside the signed
            # lifecycle preflight; the wrapper must never dispatch host work.
            records = [json.loads(line) for line in operations.read_text().splitlines()]
            assert len(records) == 1
            assert records[0][:2] == ["-m", "clipper.books.chapter_lifecycle"]
            operations.unlink()
        env[mutation] = original
    # The wrapper has no independent legacy 2-hour cutoff; for an
    # over-two-hour host it must consume a positive result from the signed
    # lifecycle verifier. The verifier's REAL OpenSSL acceptance is tested
    # separately in test_real_signed_owner_free_allowance_host_lifecycle.
    env["MOCK_CREATED"] = (datetime.now(UTC) - timedelta(hours=2, minutes=10)).isoformat()
    env["MOCK_SIGNED_GRANT"] = "true"
    extended = invoke()
    assert extended.returncode == 0, extended.stderr
    extended_ops = [json.loads(line) for line in operations.read_text().splitlines()]
    assert len(extended_ops) == 2
    assert "--remaining-seconds" in extended_ops[0]
    operations.unlink()
    env["MOCK_SIGNED_GRANT"] = "false"
    blocked = invoke()
    assert blocked.returncode == 2
    assert len([json.loads(line) for line in operations.read_text().splitlines()]) == 1
    operations.unlink()
    env["MOCK_CREATED"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    env["MOCK_HOST_STATUS"] = "124"
    timed_out = invoke()
    assert timed_out.returncode == 2
    assert "PRIVATE_CHAPTER_SHARED_DEADLINE_STOP_NO_AUTO_RESTART" in timed_out.stderr
    env["MOCK_HOST_STATUS"] = "137"
    killed = invoke()
    assert killed.returncode == 2
    assert "PRIVATE_CHAPTER_SHARED_DEADLINE_STOP_NO_AUTO_RESTART" in killed.stderr


@pytest.mark.parametrize(
    "scenario",
    [
        "provisioning_then_ready",
        "never_ready_original_clock",
        "transient_ssh",
        "identity_swap",
        "lost_rest",
        "hung_rest",
        "rest_transient_then_valid",
        "rest_persistent_unavailable",
        "rest_persistent_timeout",
        "rest_persistent_malformed",
        "rest_foreign_confirmed",
        "rest_initial_unavailable",
        "rest_initial_malformed",
        "hung_ssh",
        "terminal_state",
        "transient_cp",
        "bootstrap_over_execution_reserve",
        "free_grant_extended_deadline",
        "free_grant_reserve_cleanup",
        "free_grant_wrong_id",
        "free_grant_paid_block_missing",
        "free_grant_invalid_past_legacy",
        "free_grant_unsigned_override",
        "foreign_available_id",
        "foreign_available_name",
        "foreign_available_repo",
        "foreign_available_shape",
        "foreign_available_created",
    ],
)
def test_external_supervisor_bootstrap_real_shell_consumer(tmp_path: Path, scenario: str) -> None:
    """Execute real owner script, fake only external commands and REST responses."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "operations.log"
    state_counter = tmp_path / "rest-counter"
    ssh_counter = tmp_path / "ssh-counter"
    key = tmp_path / "owner-private.key"
    pub = tmp_path / "owner-public.pem"
    key.write_text("SYNTHETIC_KEY", encoding="utf-8")
    pub.write_text("SYNTHETIC_PUBLIC", encoding="utf-8")
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys,time\n"
        "a=sys.argv[1:]\n"
        "p=pathlib.Path(os.environ['MOCK_LOG'])\n"
        "with p.open('a') as f: f.write(' '.join(a)+'\\n')\n"
        "mode=os.environ['MOCK_BOOTSTRAP']\n"
        "if a[:2]==['api','-X']:\n"
        "  if a[2]=='POST': pathlib.Path(os.environ['MOCK_STOP']).write_text('stopped')\n"
        "  sys.exit(0)\n"
        "if a[:2]==['api','-i']:\n"
        "  print('HTTP/2 404 Not Found'); sys.exit(0)\n"
        "if a[:1]==['api']:\n"
        "  c=pathlib.Path(os.environ['MOCK_REST_COUNTER'])\n"
        "  n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n))\n"
        "  if mode=='hung_rest' and n==4: time.sleep(120)\n"
        "  if mode=='lost_rest' and n==4: sys.exit(3)\n"
        "  if mode=='rest_transient_then_valid' and n==4: sys.exit(3)\n"
        "  if mode=='rest_persistent_unavailable' and n>=4: sys.exit(3)\n"
        "  if mode=='rest_persistent_timeout' and n>=4: time.sleep(120)\n"
        "  if mode=='rest_persistent_malformed' and n>=4: print('{malformed'); sys.exit(0)\n"
        "  if mode=='rest_initial_unavailable': sys.exit(3)\n"
        "  if mode=='rest_initial_malformed': print('{malformed'); sys.exit(0)\n"
        "  state='Available'\n"
        "  if mode in ('provisioning_then_ready','transient_ssh') and n<5: state='Created'\n"
        "  if mode=='never_ready_original_clock': state='Created'\n"
        "  if mode=='terminal_state': state='Shutdown'\n"
        "  host_id=999 if mode in ('identity_swap','rest_foreign_confirmed') and n>=4 else 12\n"
        "  name=os.environ['CODESPACE_NAME']\n"
        "  repo='TheBayoumi/clipper'\n"
        "  created=os.environ['MOCK_CREATED']\n"
        "  cpus=4\n"
        "  if mode.startswith('foreign_available_'):\n"
        # Both readiness reads (n=3 and n=6) must disagree with the
        # approved identity probes (n=1,2,4,5,7) to expose old parser.
        "    state='Available' if n in (3,6) else 'Created'\n"
        "    if n in (3,6):\n"
        "      if mode=='foreign_available_id': host_id=999\n"
        "      if mode=='foreign_available_name': name='foreign-host'\n"
        "      if mode=='foreign_available_repo': repo='foreign/repo'\n"
        "      if mode=='foreign_available_shape': cpus=8\n"
        "      if mode=='foreign_available_created': created='2020-01-01T00:00:00Z'\n"
        "  print(json.dumps({'id':host_id,'name':name,"
        "'created_at':created,'state':state,"
        "'repository':{'full_name':repo},"
        "'machine':{'cpus':cpus,'memory_in_bytes':17179869184,"
        "'storage_in_bytes':34359738368}})); sys.exit(0)\n"
        "if a[:2]==['codespace','ssh']:\n"
        "  c=pathlib.Path(os.environ['MOCK_SSH_COUNTER'])\n"
        "  n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n))\n"
        "  if mode=='hung_ssh': time.sleep(120)\n"
        "  if mode=='transient_ssh' and n<3: sys.exit(3)\n"
        "  sys.exit(0)\n"
        "if a[:2]==['codespace','cp']:\n"
        "  if mode=='transient_cp' and 'external-supervisor-public.pem' in ' '.join(a):\n"
        "    c=pathlib.Path(os.environ['MOCK_SSH_COUNTER']).with_suffix('.cp')\n"
        "    n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n))\n"
        "    if n<3: sys.exit(3)\n"
        "  if 'heartbeat-envelope.watchdog.pending' in ' '.join(a): sys.exit(3)\n"
        "  sys.exit(0)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    fake_openssl = fake_bin / "openssl"
    fake_openssl.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,sys\n"
        "a=sys.argv[1:]; pathlib.Path(a[a.index('-out')+1]).write_bytes(b'SYNTHETIC_SIGNATURE')\n",
        encoding="utf-8",
    )
    fake_openssl.chmod(0o700)
    age = 1
    if scenario == "never_ready_original_clock":
        age = 599
    elif scenario == "bootstrap_over_execution_reserve":
        age = 6890
    elif scenario == "free_grant_extended_deadline":
        age = 1
    elif scenario == "free_grant_reserve_cleanup":
        age = 4 * 3600 - 240
    elif scenario in {"free_grant_invalid_past_legacy", "free_grant_unsigned_override"}:
        age = 2 * 3600 + 600
    elif scenario.startswith("free_grant_"):
        # Invalid execution grants must still clean up a young verified host.
        age = 1
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(fake_bin) + os.pathsep + env["PATH"],
            "CODESPACE_NAME": "synthetic-approved-host",
            "PRIVATE_CHAPTER_APPROVED_SHA": "f" * 40,
            "PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY": str(key),
            "PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY": str(pub),
            "PRIVATE_CHAPTER_ROOT": str(tmp_path / "host-private"),
            "PRIVATE_CHAPTER_RECOVERY_DIR": str(tmp_path / "owner-private"),
            "MOCK_BOOTSTRAP": scenario,
            "MOCK_CREATED": (datetime.now(UTC) - timedelta(seconds=age)).isoformat(),
            "MOCK_REST_COUNTER": str(state_counter),
            "MOCK_SSH_COUNTER": str(ssh_counter),
            "MOCK_STOP": str(tmp_path / "stopped"),
            "MOCK_LOG": str(log),
        }
    )
    env.pop("CODESPACES", None)
    if scenario.startswith("free_grant_"):
        if scenario == "free_grant_unsigned_override":
            env["PRIVATE_CHAPTER_FREE_BUDGET_SECONDS"] = "14400"
        else:
            grant = {
                "schema": "clipper-owner-free-allowance-v1",
                "codespace_id": 999 if scenario == "free_grant_wrong_id" else 12,
                "codespace_name": env["CODESPACE_NAME"],
                "repository": "TheBayoumi/clipper",
                "created_at": env["MOCK_CREATED"],
                "approved_sha": env["PRIVATE_CHAPTER_APPROVED_SHA"],
                "machine": {
                    "cpus": 4,
                    "memory_in_bytes": 17179869184,
                    "storage_in_bytes": 34359738368,
                },
                "verified_at": env["MOCK_CREATED"],
                "remaining_included_core_seconds": 4 * 4 * 3600,
                "total_seconds": 4 * 3600,
                "storage_included_verified": True,
                "paid_overage_block_verified": scenario
                not in {"free_grant_paid_block_missing", "free_grant_invalid_past_legacy"},
            }
            grant_file = tmp_path / "owner-finite-free-grant.json"
            grant_file.write_text(json.dumps(grant), encoding="utf-8")
            grant_file.chmod(0o600)
            env["PRIVATE_CHAPTER_FREE_GRANT_FILE"] = str(grant_file)
    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    elapsed = time.monotonic() - started
    if scenario.startswith("rest_"):
        # No unbounded REST retry, even for persistent timeouts.
        assert elapsed < 25, (scenario, elapsed, result.stderr)
    actions = log.read_text(encoding="utf-8")
    if scenario == "free_grant_reserve_cleanup":
        assert result.returncode == 0
        assert "CLEANUP_RESERVE" in result.stderr
    else:
        assert result.returncode != 0
    preflight_rejected_grant = scenario in {
        "free_grant_wrong_id",
        "free_grant_paid_block_missing",
        "free_grant_invalid_past_legacy",
    }
    expired_invalid_grant = scenario in {
        "free_grant_invalid_past_legacy",
        "free_grant_unsigned_override",
    }
    never_verified = scenario in {
        "identity_swap",
        "rest_persistent_unavailable",
        "rest_persistent_timeout",
        "rest_persistent_malformed",
        "rest_foreign_confirmed",
        "rest_initial_unavailable",
        "rest_initial_malformed",
    }
    assert "api -X DELETE" in actions or never_verified or expired_invalid_grant, result.stderr
    if never_verified:
        assert "api -X POST" not in actions
        assert "api -X DELETE" not in actions
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
    if scenario == "free_grant_unsigned_override":
        assert "EXTERNAL_SUPERVISOR_TOTAL_DEADLINE_EXPIRED_NO_AUTOMATED_DELETE" in result.stderr
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
    if preflight_rejected_grant:
        assert "EXTERNAL_SUPERVISOR_FREE_GRANT_REJECTED_NO_EXECUTION" in result.stderr
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
        if expired_invalid_grant:
            # No invalid-grant extension beyond the independently safe clock.
            assert "api -X DELETE" not in actions
        else:
            assert "api -X DELETE" in actions
            assert "api -X POST" in actions
    if scenario == "identity_swap":
        assert "api -X DELETE" not in actions
        assert "api -X POST" not in actions
    if scenario in (
        "never_ready_original_clock",
        "bootstrap_over_execution_reserve",
        "free_grant_reserve_cleanup",
    ) or scenario.startswith("foreign_available_"):
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
        assert "external-supervisor-heartbeat-envelope" not in actions
    if scenario in (
        "identity_swap",
        "terminal_state",
        "rest_persistent_unavailable",
        "rest_persistent_timeout",
        "rest_persistent_malformed",
        "rest_foreign_confirmed",
        "rest_initial_unavailable",
        "rest_initial_malformed",
    ):
        assert "codespace cp" not in actions
    if scenario == "terminal_state":
        # A valid pinned ID in Shutdown must never be called a mismatch.
        # It refuses remote setup, but exact-ID cleanup remains permitted.
        assert "BOOTSTRAP_STATE_REST_DISALLOWED_STATE" in result.stderr
        assert "CONFIRMED_MISMATCH" not in result.stderr
        assert "api -X POST" in actions
        assert "api -X DELETE" in actions
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
    if scenario == "hung_ssh":
        assert "BOOTSTRAP_REMOTE_HUNG" in result.stderr
    expected_rest_class = {
        "rest_persistent_unavailable": "TRANSPORT",
        "rest_persistent_timeout": "TIMEOUT",
        "rest_persistent_malformed": "MALFORMED",
        "rest_foreign_confirmed": "CONFIRMED_MISMATCH",
        "rest_initial_unavailable": "TRANSPORT",
        "rest_initial_malformed": "MALFORMED",
        "terminal_state": "DISALLOWED_STATE",
    }
    if scenario in expected_rest_class:
        category = expected_rest_class[scenario]
        assert "REST_" + category in result.stderr, (scenario, result.stderr)
        assert "IDENTITY_CHANGED" not in result.stderr
        journal = tmp_path / "owner-private/synthetic-approved-host"
        evidence = json.loads((journal / "owner-last-rest-failure.json").read_text())
        assert evidence["schema"] == "clipper-owner-rest-failure-v1"
        assert evidence["failure_class"] == category
        assert evidence["operation"] == (
            "rest_metadata"
            if scenario.startswith("rest_initial_")
            else "rest_state"
            if scenario == "terminal_state"
            else "rest_identity"
        )
        assert evidence["identity_verified"] is (not scenario.startswith("rest_initial_"))
        assert evidence["codespace_id"] == (None if scenario.startswith("rest_initial_") else 12)
        assert evidence["created_at"] == (
            None if scenario.startswith("rest_initial_") else env["MOCK_CREATED"]
        )
        assert "foreign" not in json.dumps(evidence)
    if scenario in {"lost_rest", "hung_rest", "rest_transient_then_valid"}:
        reads = int(state_counter.read_text())
        assert reads >= 5, reads
        # Recovery from an early metadata failure requires a NEW exact REST
        # identity check; never infer success from a failed read.
        assert "external-supervisor-public.pem" in actions
    if scenario in (
        "provisioning_then_ready",
        "transient_ssh",
        "transient_cp",
        "lost_rest",
        "hung_rest",
        "rest_transient_then_valid",
        "free_grant_extended_deadline",
    ):
        assert "codespace ssh" in actions
        assert "external-supervisor-public.pem" in actions
        assert "external-supervisor-heartbeat-envelope.watchdog.pending" in actions
        assert actions.index("external-supervisor-public.pem") < actions.index(
            "external-supervisor-heartbeat-envelope.watchdog.pending"
        )
        if scenario == "transient_ssh":
            assert actions.count("codespace ssh") >= 3
        if scenario == "transient_cp":
            assert actions.count("external-supervisor-public.pem") >= 3
    else:
        assert "external-supervisor-heartbeat-envelope.watchdog.pending" not in actions
    # Cleanup may retrieve the scoped recovery archive and, if that fails,
    # read its safe status. Neither operation authorizes generation or setup.
    host_calls = [call for call in actions.splitlines() if "clipper.books.chapter_host" in call]
    allowed_recovery = "clipper.books.chapter_host recover --artifact-root "
    allowed_status = "clipper.books.chapter_host status --artifact-root "
    assert all(allowed_recovery in call or allowed_status in call for call in host_calls), (
        host_calls
    )
    if scenario in (
        "provisioning_then_ready",
        "transient_ssh",
        "transient_cp",
        "lost_rest",
        "hung_rest",
        "rest_transient_then_valid",
        "free_grant_extended_deadline",
    ):
        assert any(allowed_recovery in call for call in host_calls), host_calls
    else:
        assert not host_calls, "Unready or unverified bootstrap must not run host commands"
    if scenario.startswith("foreign_available_"):
        # The only substitution below is the exact vulnerable bootstrap_state
        # function from reviewed 5ccd857c. Keep the rest of the REAL shell
        # supervisor and fake gh REST sequence identical. This is an offline
        # differential consumer test, not a helper/parser-only assertion.
        old_readiness = r"""bootstrap_state() {
  bootstrap_identity || return 2
  local metadata state
  metadata="$(bounded gh api "user/codespaces/$CODESPACE_NAME")" || return 2
  state="$(printf '%s' "$metadata" | bounded python3 -c '
import json, sys
try:
    obj=json.load(sys.stdin)
    state=obj["state"]
    if state not in {"Available","Created","Starting","Provisioning","Queued","Rebuilding"}:
        sys.exit(2)
    print(state)
except (ValueError, TypeError, KeyError):
    sys.exit(2)
')" || return 2
  bootstrap_identity || return 2
  printf '%s\n' "$state"
}"""
        source = _SCRIPT.read_text(encoding="utf-8")
        start = source.index("bootstrap_state() {")
        end = source.index("\nremote=", start)
        assert "SUPERVISED_CODESPACE_ID" in source[start:end]
        old_script = tmp_path / "vulnerable-supervisor.sh"
        old_script.write_text(source[:start] + old_readiness + source[end:], encoding="utf-8")
        state_counter.unlink()
        log.unlink()
        stopped = Path(env["MOCK_STOP"])
        if stopped.exists():
            stopped.unlink()
        vulnerable = subprocess.run(
            ["bash", str(old_script)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        old_calls = log.read_text(encoding="utf-8")
        assert "codespace ssh" in old_calls, (scenario, vulnerable.stderr, old_calls)
        assert "mkdir -p" in old_calls
        assert "codespace cp" not in old_calls
        assert "api -X DELETE" in old_calls
    for forbidden in (
        "pip install",
        "chapter_host brief",
        "chapter_host resume",
        "chapter_host hybrid-brief",
        "chapter_host hybrid-resume",
        "chapter_host inspect",
        "model.generate",
    ):
        assert forbidden not in actions


@pytest.mark.parametrize(
    ("held_operation", "identity_swapped"),
    [
        ("grant_validation", False),
        ("grant_validation", True),
        ("journal_setup", False),
        ("journal_setup", True),
    ],
)
def test_signal_during_local_preflight_retains_exact_id_cleanup_clock(
    tmp_path: Path, held_operation: str, identity_swapped: bool
) -> None:
    """SIGTERM while a REAL owner shell is inside a bounded local subprocess.

    The REST host is initially ID 12 and one second old. The synthetic
    transport optionally replaces the ID during the held local operation.
    No network, actual hosting, book text or model execution is involved.
    """
    import signal

    fake_bin = tmp_path / "commands"
    fake_bin.mkdir()
    marker = tmp_path / "local-operation-entered"
    gh_log = tmp_path / "gh-operations.log"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['MOCK_LOG'],'a') as f: f.write(' '.join(args)+'\\n')\n"
        "if args[:2]==['api','-i']:\n"
        "  print('HTTP/2 404 Not Found'); sys.exit(0)\n"
        "if args[:2]==['api','-X']: sys.exit(0)\n"
        "if args[:1]==['api']:\n"
        "  changed=os.environ['MOCK_SWAP']=='1' and pathlib.Path(os.environ['MOCK_MARK']).exists()\n"
        "  print(json.dumps({'id':999 if changed else 12,"
        "'name':os.environ['CODESPACE_NAME'],'created_at':os.environ['MOCK_CREATED'],"
        "'state':'Available','repository':{'full_name':'TheBayoumi/clipper'},"
        "'machine':{'cpus':4,'memory_in_bytes':17179869184,"
        "'storage_in_bytes':34359738368}})); sys.exit(0)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    # Hold the local Python grant validator or a real mkdir-after-creation
    # without affecting the REST probes. The parent receives SIGTERM while
    # this subprocess is alive, not after grant validation returns.
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os,pathlib,sys,time\n"
        "args=sys.argv[1:]\n"
        "if os.environ['MOCK_HOLD']=='grant_validation' and len(args)>1 "
        "and 'PRIVATE_CHAPTER_FREE_GRANT_FILE' in args[1]:\n"
        "  pathlib.Path(os.environ['MOCK_MARK']).touch(); time.sleep(3)\n"
        "os.execv(sys.executable,[sys.executable,*args])\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    fake_mkdir = fake_bin / "mkdir"
    fake_mkdir.write_text(
        f"#!{sys.executable}\n"
        "import os,pathlib,subprocess,sys,time\n"
        "args=sys.argv[1:]\n"
        "code=subprocess.call(['/usr/bin/mkdir',*args])\n"
        "if code: sys.exit(code)\n"
        "if os.environ['MOCK_HOLD']=='journal_setup':\n"
        "  pathlib.Path(os.environ['MOCK_MARK']).touch(); time.sleep(3)\n",
        encoding="utf-8",
    )
    fake_mkdir.chmod(0o700)
    created = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    grant = {
        "schema": "clipper-owner-free-allowance-v1",
        "codespace_id": 12,
        "codespace_name": "synthetic-approved",
        "repository": "TheBayoumi/clipper",
        "created_at": created,
        "approved_sha": "a" * 40,
        "machine": {
            "cpus": 4,
            "memory_in_bytes": 17179869184,
            "storage_in_bytes": 34359738368,
        },
        "verified_at": created,
        "remaining_included_core_seconds": 4 * 4 * 3600,
        "total_seconds": 4 * 3600,
        "storage_included_verified": True,
        "paid_overage_block_verified": False,
    }
    grant_path = tmp_path / "invalid-grant.json"
    grant_path.write_text(json.dumps(grant), encoding="utf-8")
    grant_path.chmod(0o600)
    private = tmp_path / "owner-private"
    private.mkdir(mode=0o700)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "CODESPACE_NAME": "synthetic-approved",
            "PRIVATE_CHAPTER_APPROVED_SHA": "a" * 40,
            "PRIVATE_CHAPTER_ROOT": str(tmp_path / "host-private"),
            "PRIVATE_CHAPTER_RECOVERY_DIR": str(private),
            "PRIVATE_CHAPTER_FREE_GRANT_FILE": str(grant_path),
            "MOCK_CREATED": created,
            "MOCK_LOG": str(gh_log),
            "MOCK_MARK": str(marker),
            "MOCK_HOLD": held_operation,
            "MOCK_SWAP": "1" if identity_swapped else "0",
        }
    )
    env.pop("CODESPACES", None)
    child = subprocess.Popen(
        ["bash", str(_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        until = time.monotonic() + 12
        while time.monotonic() < until and child.poll() is None and not marker.exists():
            time.sleep(0.025)
        assert marker.exists(), "expected local preflight operation was never reached"
        assert child.poll() is None, "supervisor exited before injection"
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=22)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
    # The parent was signaled while the subprocess was held; Bash may defer
    # the TERM trap until its bounded local command returns. A subsequently
    # replaced REST identity must keep the existing fail-closed exit code 2.
    assert child.returncode == (2 if identity_swapped else 143), (stdout, stderr)
    assert "EXTERNAL_SUPERVISOR_SIGNAL_TERM_CLEANUP_REQUIRED" in stderr
    calls = gh_log.read_text(encoding="utf-8")
    assert "codespace ssh" not in calls and "codespace cp" not in calls
    if identity_swapped:
        assert "api -X POST" not in calls
        assert "api -X DELETE" not in calls
        assert "REJECTED_HOST_IDENTITY_NO_DESTRUCTIVE_CALL" in stderr
    else:
        assert "api -X POST" in calls
        assert "api -X DELETE" in calls
        assert "EXTERNAL_SUPERVISOR_HOST_" in stdout
        assert "TOTAL_DEADLINE_EXPIRED" not in stderr
    journal = private / "synthetic-approved"
    first = json.loads((journal / "owner-first-failure.json").read_text())
    latest = json.loads((journal / "owner-last-failure.json").read_text())
    reason = json.loads((journal / "owner-exit-reason.json").read_text())
    # Keep the actual first failure: a held grant validator may finish and
    # reject the grant before Bash runs its deferred SIGTERM handler.
    assert first["operation"] == (
        "grant_validation" if held_operation == "grant_validation" else "signal"
    )
    assert first["exit_code"] == (2 if held_operation == "grant_validation" else 143)
    assert first["codespace_id"] == 12
    if held_operation == "journal_setup" and identity_swapped:
        # The bounded local command can finish after the deferred TERM trap
        # and update the mutable last-operation record. Preserve its truth.
        assert latest["operation"] == "journal_setup"
        assert latest["exit_code"] != 0
    else:
        assert latest["operation"] == "signal"
        assert latest["exit_code"] == 143
        assert latest["exit_class"] == "SIGNAL"
    assert latest["codespace_id"] == 12
    assert reason["original_exit_code"] == 143
    assert reason["exit_class"] == "SIGNAL"
    assert reason["codespace_id"] == 12
    assert reason["most_recent_recorded_failed_operation"] == "signal"
    assert "invalid-grant" not in json.dumps(first) + json.dumps(latest) + json.dumps(reason)


@pytest.mark.parametrize(
    "mode",
    [
        "disconnect",
        "hung",
        "slow_then_disconnect",
        "healthy_then_disconnect",
        "truncated_input",
        "duplicate_input",
        "duplicate_ack",
        "out_of_order_ack",
        "foreign_host",
        "signal",
        "near_deadline",
    ],
)
def test_heartbeat_stream_shell_transport_atomic_ack_and_cleanup(tmp_path: Path, mode: str) -> None:
    """Real owner Bash + a subprocess-backed remote shell, never a Codespace."""
    import signal

    fake_bin = tmp_path / "mock-bin"
    fake_bin.mkdir()
    log = tmp_path / "mock-gh.log"
    publish_counter = tmp_path / "publish-count"
    remote_root = tmp_path / "remote-private"
    first_ack = tmp_path / "first-ack"
    entered = tmp_path / "publish-entered"
    transport_pid = tmp_path / "publish-pid"
    swapped = tmp_path / "id-swapped"
    mock_gh = fake_bin / "gh"
    mock_gh.write_text(
        f"#!{sys.executable}\n"
        "import hashlib,json,os,pathlib,subprocess,sys,time\n"
        "args=sys.argv[1:]\n"
        "root=pathlib.Path(os.environ['MOCK_REMOTE_ROOT'])\n"
        "with open(os.environ['MOCK_LOG'],'a',encoding='utf-8') as f:\n"
        "  f.write(' '.join(args)+'\\n')\n"
        "mode=os.environ['MOCK_STREAM_MODE']\n"
        "if args[:2]==['api','-X']:\n"
        "  if args[2]=='POST': pathlib.Path(os.environ['MOCK_STOPPED']).touch()\n"
        "  sys.exit(0)\n"
        "if args[:2]==['api','-i']:\n"
        "  print('HTTP/2 404 Not Found'); sys.exit(0)\n"
        "if args[:1]==['api']:\n"
        "  changed=mode=='foreign_host' and pathlib.Path(os.environ['MOCK_SWAPPED']).exists()\n"
        "  state='Stopped' if pathlib.Path(os.environ['MOCK_STOPPED']).exists() else 'Available'\n"
        "  print(json.dumps({'id':999 if changed else 12,'name':os.environ['CODESPACE_NAME'],"
        "'created_at':os.environ['MOCK_CREATED'],'state':state,"
        "'repository':{'full_name':'TheBayoumi/clipper'},"
        "'machine':{'cpus':4,'memory_in_bytes':17179869184,"
        "'storage_in_bytes':34359738368}})); sys.exit(0)\n"
        "if args[:2]==['codespace','cp']:\n"
        "  if mode=='foreign_host': pathlib.Path(os.environ['MOCK_SWAPPED']).touch()\n"
        "  sys.exit(0)\n"
        "if args[:2]==['codespace','ssh']:\n"
        "  command=args[-1]\n"
        "  if 'watchdog.pending' not in command:\n"
        "    completed=subprocess.run(['bash','-c',command],capture_output=True,check=False)\n"
        "    sys.exit(completed.returncode)\n"
        "  assert args[-2]=='-T', args\n"
        "  count_file=pathlib.Path(os.environ['MOCK_PUBLISH_COUNTER'])\n"
        "  n=int(count_file.read_text())+1 if count_file.exists() else 1\n"
        "  count_file.write_text(str(n))\n"
        "  pathlib.Path(os.environ['MOCK_PID']).write_text(str(os.getpid()))\n"
        "  pathlib.Path(os.environ['MOCK_ENTERED']).touch()\n"
        "  if mode in ('hung','signal'): time.sleep(120)\n"
        "  if mode in ('slow_then_disconnect','healthy_then_disconnect','out_of_order_ack') and n==2:\n"
        "    if mode=='out_of_order_ack':\n"
        "      print(pathlib.Path(os.environ['MOCK_FIRST_ACK']).read_text(),end='')\n"
        "      sys.exit(0)\n"
        "    sys.exit(3)\n"
        "  if mode=='disconnect': sys.exit(3)\n"
        "  if mode=='slow_then_disconnect': time.sleep(4)\n"
        "  data=sys.stdin.buffer.read()\n"
        "  if mode=='truncated_input': data=data[:-7]\n"
        "  if mode=='duplicate_input': data=data+data\n"
        "  completed=subprocess.run(['bash','-c',command],input=data,capture_output=True,check=False)\n"
        "  if completed.returncode:\n"
        "    sys.stderr.buffer.write(completed.stderr); sys.exit(completed.returncode)\n"
        "  ack=completed.stdout.decode('ascii')\n"
        "  if n==1: pathlib.Path(os.environ['MOCK_FIRST_ACK']).write_text(ack)\n"
        "  if mode=='duplicate_ack': ack=ack+ack\n"
        "  sys.stdout.write(ack); sys.exit(0)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    mock_gh.chmod(0o700)
    fake_openssl = fake_bin / "openssl"
    fake_openssl.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,sys\n"
        "a=sys.argv[1:]; pathlib.Path(a[a.index('-out')+1]).write_bytes(b'SYNTHETIC_SIGNATURE')\n",
        encoding="utf-8",
    )
    fake_openssl.chmod(0o700)
    key = tmp_path / "owner.key"
    pub = tmp_path / "owner.pub"
    key.write_text("SYNTHETIC_PRIVATE_ONLY", encoding="utf-8")
    pub.write_text("SYNTHETIC_PUBLIC_ONLY", encoding="utf-8")
    created = (
        datetime.now(UTC) - timedelta(seconds=7196 if mode == "near_deadline" else 1)
    ).isoformat()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "CODESPACE_NAME": "synthetic-stream-host",
            "PRIVATE_CHAPTER_APPROVED_SHA": "b" * 40,
            "PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY": str(key),
            "PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY": str(pub),
            "PRIVATE_CHAPTER_ROOT": str(remote_root),
            "PRIVATE_CHAPTER_RECOVERY_DIR": str(tmp_path / "owner-private"),
            "MOCK_REMOTE_ROOT": str(remote_root),
            "MOCK_CREATED": created,
            "MOCK_LOG": str(log),
            "MOCK_PUBLISH_COUNTER": str(publish_counter),
            "MOCK_ENTERED": str(entered),
            "MOCK_FIRST_ACK": str(first_ack),
            "MOCK_PID": str(transport_pid),
            "MOCK_SWAPPED": str(swapped),
            "MOCK_STOPPED": str(tmp_path / "stopped"),
            "MOCK_STREAM_MODE": mode,
        }
    )
    env.pop("CODESPACES", None)
    started = time.monotonic()
    if mode == "signal":
        child = subprocess.Popen(
            ["bash", str(_SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            until = time.monotonic() + 12
            while time.monotonic() < until and child.poll() is None and not entered.exists():
                time.sleep(0.025)
            assert entered.exists() and child.poll() is None
            child.send_signal(signal.SIGTERM)
            stdout, stderr = child.communicate(timeout=30)
            rc = child.returncode
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
    else:
        finished = subprocess.run(
            ["bash", str(_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=32,
        )
        rc, stdout, stderr = finished.returncode, finished.stdout, finished.stderr
    elapsed = time.monotonic() - started
    assert elapsed < 30, (mode, elapsed, stderr)
    actions = log.read_text(encoding="utf-8")
    assert "api -X DELETE" not in actions if mode == "foreign_host" else "api -X DELETE" in actions
    if mode == "foreign_host":
        assert "api -X POST" not in actions
        assert not entered.exists()
        assert rc == 2
        assert "BOOTSTRAP_REST_CONFIRMED_MISMATCH" in stderr
        return
    if mode == "near_deadline":
        assert "codespace ssh" not in actions and not entered.exists()
        assert rc == 0
        assert "CLEANUP_RESERVE" in stderr
        return
    assert "api -X POST" in actions
    assert entered.exists()
    assert "codespace cp" in actions  # Only bootstrap public-key transfer.
    assert "remote:" in actions
    assert "watchdog.pending" in actions
    assert "codespace cp" not in "\n".join(
        line for line in actions.splitlines() if "watchdog.pending" in line
    )
    assert "SYNTHETIC_PRIVATE_ONLY" not in actions
    if mode in {"hung", "signal"}:
        assert rc == (143 if mode == "signal" else 124), (rc, stderr)
        assert "EXTERNAL_SUPERVISOR_HOST_" in stdout
        assert "heartbeat-envelope.json" not in [
            p.name for p in remote_root.rglob("*") if p.is_file()
        ]
    elif mode in {"duplicate_ack", "out_of_order_ack"}:
        assert rc == 2, (rc, stderr)
        assert "HEARTBEAT_PUBLICATION_ACK_INVALID" in stderr
    elif mode in {"truncated_input", "duplicate_input"}:
        assert rc != 0
        assert "HEARTBEAT_PUBLICATION_ACK_INVALID" not in stderr
        published = list(remote_root.rglob("external-supervisor-heartbeat-envelope.json"))
        assert not published, published
    elif mode in {"disconnect", "healthy_then_disconnect", "slow_then_disconnect"}:
        assert rc == 3, (rc, stderr)
    if mode in {"healthy_then_disconnect", "slow_then_disconnect", "out_of_order_ack"}:
        assert int(publish_counter.read_text()) == 2
        assert len(first_ack.read_text().strip()) == 68  # ACK: plus 64 hex.
        published = list(remote_root.rglob("external-supervisor-heartbeat-envelope.json"))
        assert len(published) == 1
        body = published[0].read_bytes()
        assert hashlib.sha256(body).hexdigest() == first_ack.read_text().strip()[4:]
        assert b"SYNTHETIC_PRIVATE_ONLY" not in body
    if transport_pid.exists():
        pid = int(transport_pid.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            pytest.fail(f"orphaned heartbeat transport process {pid}")


@pytest.mark.parametrize("producer_exit", [0, 7])
def test_exact_recovery_receiver_success_and_nonzero_producer(
    tmp_path: Path, producer_exit: int
) -> None:
    """Run the actual bounded receiver and subprocess, not the archive verifier."""
    content = _SCRIPT.read_text(encoding="utf-8")
    marker = "critical recovery_copy python3 -c '\n"
    assert content.count(marker) == 1
    receiver = content.split(marker, 1)[1].split(
        '\' "$CODESPACE_NAME" "$remote_archive" "$target" "$session/recovery-created.json"',
        1,
    )[0]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "assert sys.argv[1:7]==['codespace','ssh','-c','synthetic-pinned','--','-T']\n"
        "sys.stdout.buffer.write(b'hello\\x00\\xff')\n"
        "sys.stdout.buffer.flush()\n"
        "sys.exit(int(os.environ['MOCK_PRODUCER_EXIT']))\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    target = tmp_path / "received.tar"
    receipt = tmp_path / "prepared.json"
    receipt.write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["MOCK_PRODUCER_EXIT"] = str(producer_exit)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            receiver,
            "synthetic-pinned",
            "/synthetic-private/chapter-recovery.tar",
            str(target),
            str(receipt),
        ],
        env=env,
        capture_output=True,
        check=False,
        timeout=8,
    )
    assert target.read_bytes() == b"hello\x00\xff"
    assert result.stdout == b""
    assert b"Fatal Python error" not in result.stderr
    assert result.returncode == (0 if producer_exit == 0 else 2), result.stderr
    assert target.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("interruption", ["soft", "hard"])
def test_exact_recovery_receiver_inner_popen_startup_interruption(
    tmp_path: Path, interruption: str
) -> None:
    """Interrupt *after* the inner OS spawn but *before* Popen returns.

    Both producer and non-detached descendant must leave the registered
    launcher process group. This runs the actual receiver and launcher
    source with only the inner Popen-return boundary widened.
    """
    import signal

    content = _SCRIPT.read_text(encoding="utf-8")
    marker = "critical recovery_copy python3 -c '\n"
    assert content.count(marker) == 1
    receiver = content.split(marker, 1)[1].split(
        '\' "$CODESPACE_NAME" "$remote_archive" "$target" "$session/recovery-created.json"',
        1,
    )[0]
    boundary = "threading.Thread(target=parent_gone, daemon=True).start()\n"
    assert receiver.count(boundary) == 1
    injection = (
        "import pathlib, time\n"
        "_original_popen = subprocess.Popen\n"
        "def _held_inner_popen(*args, **kwargs):\n"
        "    process = _original_popen(*args, **kwargs)\n"
        "    pathlib.Path(os.environ['MOCK_INNER_ENTERED']).touch()\n"
        "    time.sleep(3)\n"
        "    return process\n"
        "subprocess.Popen = _held_inner_popen\n"
    )
    receiver = receiver.replace(boundary, boundary + injection, 1)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    producer = fake_bin / "gh"
    producer.write_text(
        f"#!{sys.executable}\n"
        "import os,pathlib,subprocess,sys,time\n"
        "pathlib.Path(os.environ['MOCK_PRODUCER_PID']).write_text(str(os.getpid()))\n"
        "pathlib.Path(os.environ['MOCK_PRODUCER_PGID']).write_text(str(os.getpgrp()))\n"
        "descendant=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "pathlib.Path(os.environ['MOCK_DESCENDANT_PID']).write_text(str(descendant.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    producer.chmod(0o700)
    target = tmp_path / "archive.tar"
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    entered = tmp_path / "inner-entered"
    producer_pid = tmp_path / "producer.pid"
    descendant_pid = tmp_path / "descendant.pid"
    producer_pgid = tmp_path / "producer.pgid"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "MOCK_INNER_ENTERED": str(entered),
            "MOCK_PRODUCER_PID": str(producer_pid),
            "MOCK_PRODUCER_PGID": str(producer_pgid),
            "MOCK_DESCENDANT_PID": str(descendant_pid),
        }
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            receiver,
            "synthetic-pinned",
            "/synthetic-private/chapter-recovery.tar",
            str(target),
            str(receipt),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 7
        while time.monotonic() < deadline and child.poll() is None:
            if entered.exists() and producer_pid.exists() and descendant_pid.exists():
                break
            time.sleep(0.02)
        assert entered.exists() and producer_pid.exists() and descendant_pid.exists()
        assert child.poll() is None, "receiver exited before interruption"
        child.send_signal(signal.SIGTERM if interruption == "soft" else signal.SIGKILL)
        _, stderr = child.communicate(timeout=5)
        assert child.returncode == (143 if interruption == "soft" else -9), stderr

        def running(pid: int) -> bool:
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists():
                return False
            # Reparented zombies do not represent surviving processes.
            return stat.read_text().split(") ", 1)[1][:1] != "Z"

        deadline = time.monotonic() + 4
        pids = [int(producer_pid.read_text()), int(descendant_pid.read_text())]
        while time.monotonic() < deadline and any(running(pid) for pid in pids):
            time.sleep(0.02)
        assert all(not running(pid) for pid in pids), (
            interruption,
            pids,
            producer_pgid.read_text(),
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        # Cleanup only the synthetic process group established for this test.
        if producer_pgid.exists():
            with suppress(ProcessLookupError):
                os.killpg(int(producer_pgid.read_text()), signal.SIGKILL)
