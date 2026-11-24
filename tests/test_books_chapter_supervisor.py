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
    # Stop returning or stalling never licenses running another host.
    assert "PRIVATE_CHAPTER" not in actions


@pytest.mark.parametrize(
    ("scenario", "tamper_archive"),
    [
        ("brief_only", False),
        ("brief_only", True),
        ("inflight_empty", False),
        ("one_complete_one_failed", False),
        ("all_ready", False),
    ],
)
def test_owner_recovery_hash_verification_is_independent_of_rest_deletion(
    tmp_path: Path, scenario: str, tamper_archive: bool
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
        audio_bytes = b"synthetic WAV samples only"
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
    receipt = {
        "schema": "clipper-private-chapter-recovery-created-v1",
        "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(listing_bytes).hexdigest(),
    }
    response = tmp_path / "synthetic-remote-recovery-result.json"
    response.write_text(json.dumps(receipt), encoding="utf-8")
    if scenario in {"one_complete_one_failed", "all_ready"}:
        # Exercise the owner's EXACT bounded Python TAR verifier in isolation;
        # the shell's fallback intentionally hides its private error details.
        content = _SCRIPT.read_text(encoding="utf-8")
        core = content.split("if bounded python3 -c '\n", 1)[1].split(
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
        "INCOMPLETE"
        if tamper_archive
        else "VERIFIED_PRIVATE_ARCHIVE"
        if scenario == "all_ready"
        else "VERIFIED_PRIVATE_ARCHIVE_PARTIAL"
    ), (result.stderr, verification)
    if not tamper_archive:
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
    assert (folder / "chapter-recovery.tar").exists() is not tamper_archive
    assert "fabricated fixture text" not in result.stdout + result.stderr


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
            "  sys.exit(int(os.environ['MOCK_LIFECYCLE_STATUS']))\n"
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
        if mutation != "MOCK_LIFECYCLE_STATUS":
            assert not operations.exists(), mutation
        else:
            records = [json.loads(line) for line in operations.read_text().splitlines()]
            assert len(records) == 1
            assert records[0][:2] == ["-m", "clipper.books.chapter_lifecycle"]
            operations.unlink()
        env[mutation] = original
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
        "hung_ssh",
        "terminal_state",
        "transient_cp",
        "bootstrap_over_execution_reserve",
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
        "  state='Available'\n"
        "  if mode in ('provisioning_then_ready','transient_ssh') and n<5: state='Created'\n"
        "  if mode=='never_ready_original_clock': state='Created'\n"
        "  if mode=='terminal_state': state='Shutdown'\n"
        "  host_id=999 if mode=='identity_swap' and n>=4 else 12\n"
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
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    actions = log.read_text(encoding="utf-8")
    assert result.returncode != 0
    assert "api -X DELETE" in actions or scenario == "identity_swap", result.stderr
    if scenario == "identity_swap":
        assert "api -X DELETE" not in actions
        assert "api -X POST" not in actions
    if scenario in (
        "never_ready_original_clock",
        "bootstrap_over_execution_reserve",
    ) or scenario.startswith("foreign_available_"):
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
        assert "external-supervisor-heartbeat-envelope" not in actions
    if scenario in ("lost_rest", "hung_rest", "identity_swap", "terminal_state"):
        assert "codespace cp" not in actions
    if scenario == "hung_ssh":
        assert "BOOTSTRAP_REMOTE_HUNG" in result.stderr
    if scenario in ("provisioning_then_ready", "transient_ssh", "transient_cp"):
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
    allowed_recovery = "python3 -m clipper.books.chapter_host recover --artifact-root "
    allowed_status = "python3 -m clipper.books.chapter_host status --artifact-root "
    assert all(allowed_recovery in call or allowed_status in call for call in host_calls), (
        host_calls
    )
    if scenario in ("provisioning_then_ready", "transient_ssh", "transient_cp"):
        assert any(allowed_recovery in call for call in host_calls), host_calls
    else:
        assert not host_calls, "Unready or unverified bootstrap must not run host commands"
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
