"""GitHub-runner orchestration of one verified Modal original and shared TJR editing.

Source discovery may skip ineligible live/short/unavailable uploads before staging,
but once one original is verified and staged, editorial weakness never causes a
silent switch to an older video. Zero creator-grade clips is a valid no-op.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.tjr_quality import probe_original
from scripts.tjr_youtube_preview import render_youtube_previews

VOLUME = "clipper-tjr-source-transport"
_REMOTE_SOURCE = re.compile(r"runs/\d{4,20}-\d{1,4}/[A-Za-z0-9_-]{11}/original\.(?:mp4|mkv|webm)")


def _load_staged_original(root: Path) -> dict[str, Any]:
    stage = root / "staged-original.json"
    if not stage.is_file():
        raise RuntimeError("Modal completed without a verified staged-original manifest")
    data: Any = json.loads(stage.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED":
        raise RuntimeError("Modal source staging did not verify real original bytes")
    return data


def _acquire_original(excluded: set[str], root: Path) -> dict[str, Any]:
    if os.getenv("TJR_MODAL_USE_STAGED") == "1":
        return _load_staged_original(root)
    env = {
        **os.environ,
        "TJR_MODAL_STAGE_ORIGINAL": "1",
        "TJR_MODAL_EXCLUDE_VIDEO_IDS": ",".join(sorted(excluded)),
    }
    stage = root / "staged-original.json"
    stage.unlink(missing_ok=True)
    subprocess.run(["modal", "run", "-m", "scripts.tjr_modal_probe"], env=env, check=True)
    return _load_staged_original(root)


def _transfer_verified_original(staging: dict[str, Any], destination: Path) -> Path:
    remote = str(staging.get("source_remote_path") or "")
    video_id = str(staging.get("video_id") or "")
    channel_id = str(staging.get("channel_id") or "")
    source_url = str(staging.get("public_video_url") or "")
    expected = str(staging.get("source_sha256") or "")
    if (
        staging.get("source_scan_complete") is not True
        or float(staging.get("staged_duration_seconds") or 0) + 30
        < float(staging.get("duration") or 0)
        or float(staging.get("duration") or 0) > 3600
        or not _REMOTE_SOURCE.fullmatch(remote)
        or video_id != Path(remote).parent.name
        or source_url != f"https://www.youtube.com/watch?v={video_id}"
        or channel_id not in {"UCGHBUXjDCeiIXNdKR0HUZnA", "UCZen39LQJPx04GjPj7FOMcw"}
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise RuntimeError("Modal original manifest failed channel, source or hash validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = destination.with_suffix(Path(remote).suffix)
    try:
        subprocess.run(["modal", "volume", "get", VOLUME, remote, str(original)], check=True)
        if not original.is_file():
            raise RuntimeError("Modal volume transfer did not produce an original")
        with original.open("rb") as src:
            actual = hashlib.file_digest(src, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError("Modal original source transfer SHA-256 mismatch")
        probe_original(original)
        manifest = {
            "video_id": video_id,
            "channel_id": channel_id,
            "public_video_url": source_url,
            "duration": staging["duration"],
            "title": staging.get("title") or "",
            "source_path": str(original.resolve()),
            "source_sha256": actual,
            "source_transport": "modal_exact_original_youtube",
        }
        destination.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return original
    except Exception:
        original.unlink(missing_ok=True)
        destination.with_suffix(".json").unlink(missing_ok=True)
        raise


def _purge_remote(staging: dict[str, Any], diagnostic: Path | None = None) -> None:
    remote = str(staging.get("source_remote_path") or "")
    if not _REMOTE_SOURCE.fullmatch(remote):
        raise RuntimeError("invalid remote path for mandatory Modal cleanup")
    try:
        result = subprocess.run(
            ["modal", "volume", "rm", VOLUME, remote],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return
        detail = {
            "source_remote_path": remote,
            "exit_code": result.returncode,
            "stderr": result.stderr[-1200:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        detail = {
            "source_remote_path": remote,
            "exception": type(exc).__name__,
            "error": str(exc)[-500:],
        }
    if diagnostic is not None:
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError("MODAL_VOLUME_CLEANUP_FAILED: " + remote)


def run_modal_production(
    *,
    root: Path = Path("tjr-modal-artifacts"),
    brief: Path = Path("campaigns/reach-tjr-weekly.yaml"),
    probe_root: Path = Path("tjr-modal-probe"),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    attempt_root = root / "attempt-1"
    attempt_root.mkdir(parents=True, exist_ok=True)
    staging: dict[str, Any] | None = None
    original: Path | None = None
    try:
        staging = _acquire_original(set(), probe_root)
        egress_report = probe_root / "verified-original-egress.json"
        if egress_report.is_file():
            (attempt_root / "verified-original-egress.json").write_text(
                egress_report.read_text(encoding="utf-8"), encoding="utf-8"
            )
        (attempt_root / "source-roundtrip.json").write_text(
            json.dumps(
                {
                    "attempt": 1,
                    "video_id": staging["video_id"],
                    "channel_id": staging["channel_id"],
                    "source_sha256": staging["source_sha256"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        original = _transfer_verified_original(staging, attempt_root / "source")
        env_manifest = str((attempt_root / "source.json").resolve())
        # This exact source is authoritative after verification. Editorial weakness
        # must produce an audited zero-clip result, never a different source.
        os.environ["TJR_BROWSER_CAPTURE_FILE"] = env_manifest
        os.environ["TJR_REQUIRE_STAGED_ORIGINAL"] = "1"
        os.environ["TJR_SOURCE_VIDEO_ID"] = str(staging["video_id"])
        result = render_youtube_previews(attempt_root, brief)
        print("REAL_VERIFIED_TJR_RENDER_ARTIFACTS:", result, flush=True)
        return result
    finally:
        try:
            if staging is not None:
                _purge_remote(staging, attempt_root / "remote-cleanup-error.json")
        finally:
            if original is not None:
                original.unlink(missing_ok=True)
            (attempt_root / "source.json").unlink(missing_ok=True)
            os.environ.pop("TJR_BROWSER_CAPTURE_FILE", None)
            os.environ.pop("TJR_REQUIRE_STAGED_ORIGINAL", None)
            os.environ.pop("TJR_SOURCE_VIDEO_ID", None)


def main() -> int:
    try:
        run_modal_production()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"TJR_MODAL_PRODUCTION_FAILED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
