"""Run the owner-side watchdog against synthetic hung GitHub CLI commands only."""

from __future__ import annotations

import os
import subprocess
import sys
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
        "with open(os.environ['MOCK_GH_LOG'],'a') as f: f.write(' '.join(args)+'\\n')\n"
        "hang=os.environ['MOCK_HANG']\n"
        "if args[:2]==['api','-X']:\n"
        "  if args[2]=='POST' and hang=='stop': time.sleep(120)\n"
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
                - timedelta(
                    minutes=116 if hang == "cleanup_116" else 121 if hang == "expired_121" else 5
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
    assert result.returncode == (0 if hang == "cleanup_116" else 2)
    assert elapsed < 30
    actions = log.read_text(encoding="utf-8")
    if hang in {"wrong_identity", "swap_before_stop", "expired_121"}:
        assert "api -X POST" not in actions
        assert "api -X DELETE" not in actions
    elif hang == "swap_after_stop":
        assert "api -X POST" in actions
        assert "api -X DELETE" not in actions
    else:
        assert "api -X POST" in actions
        assert "api -X DELETE" in actions
        assert "api -i" in actions
    if hang in {"cleanup_116", "unsafe_remote", "shell_metacharacters", "expired_121"}:
        assert "codespace ssh" not in actions
        assert "codespace cp" not in actions
    if hang in {"unsafe_remote", "shell_metacharacters"}:
        assert "DO_NOT_EXECUTE" not in result.stdout
        assert not (tmp_path / "DO_NOT_EXECUTE").exists()
    # Stop returning or stalling never licenses running another host.
    assert "PRIVATE_CHAPTER" not in actions
