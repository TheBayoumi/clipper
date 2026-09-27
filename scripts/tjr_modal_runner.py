"""GitHub-runner orchestration of verified Modal originals and shared TJR editing.

Retries a different *approved* upload only when the previous video's own
transcript failed the editorial gate. Never substitutes outside creators, an
unverified mirror, or a synthetic clip. Each failed source retains diagnostics.
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
from scripts.tjr_youtube_preview import NoEditorialMoments, render_youtube_previews

VOLUME = "clipper-tjr-source-transport"
_REMOTE_SOURCE = re.compile(
    r"runs/\d{4,20}-\d{1,4}/[A-Za-z0-9_-]{11}/original\.(?:mp4|mkv|webm)"
)


def _acquire_original(excluded: set[str], root: Path) -> dict[str, Any]:
    env = {
        **os.environ,
        "TJR_MODAL_STAGE_ORIGINAL": "1",
        "TJR_MODAL_EXCLUDE_VIDEO_IDS": ",".join(sorted(excluded)),
    }
    stage = root / "staged-original.json"
    stage.unlink(missing_ok=True)
    subprocess.run(["modal", "run", "-m", "scripts.tjr_modal_probe"], env=env, check=True)
    if not stage.is_file():
        raise RuntimeError("Modal completed without a verified staged-original manifest")
    data: Any = json.loads(stage.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED":
        raise RuntimeError("Modal source staging did not verify real original bytes")
    return data


def _transfer_verified_original(staging: dict[str, Any], destination: Path) -> Path:
    remote = str(staging.get("source_remote_path") or "")
    video_id = str(staging.get("video_id") or "")
    channel_id = str(staging.get("channel_id") or "")
    source_url = str(staging.get("public_video_url") or "")
    expected = str(staging.get("source_sha256") or "")
    if (
        not _REMOTE_SOURCE.fullmatch(remote)
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


def _purge_remote(staging: dict[str, Any]) -> None:
    remote = str(staging.get("source_remote_path") or "")
    if _REMOTE_SOURCE.fullmatch(remote):
        subprocess.run(
            ["modal", "volume", "rm", VOLUME, remote],
            check=False,
            stdout=subprocess.DEVNULL,
        )


def run_modal_production(
    *,
    root: Path = Path("tjr-modal-artifacts"),
    brief: Path = Path("campaigns/reach-tjr-weekly.yaml"),
    probe_root: Path = Path("tjr-modal-probe"),
    max_sources: int = 2,
) -> Path:
    if not 1 <= max_sources <= 3:
        raise ValueError("max_sources must be between 1 and 3")
    explicitly_pinned = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip()
    excluded: set[str] = set()
    failures: list[dict[str, str]] = []
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, (1 if explicitly_pinned else max_sources) + 1):
        staging: dict[str, Any] | None = None
        original: Path | None = None
        attempt_root = root / f"attempt-{attempt}"
        attempt_root.mkdir(parents=True, exist_ok=True)
        try:
            staging = _acquire_original(excluded, probe_root)
            egress_report = probe_root / "verified-original-egress.json"
            if egress_report.is_file():
                (attempt_root / "verified-original-egress.json").write_text(
                    egress_report.read_text(encoding="utf-8"), encoding="utf-8"
                )
            (attempt_root / "source-roundtrip.json").write_text(
                json.dumps(
                    {
                        "attempt": attempt,
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
            # This exact source is required: the editor must never silently
            # acquire another video after verification or an editorial failure.
            os.environ["TJR_BROWSER_CAPTURE_FILE"] = env_manifest
            os.environ["TJR_REQUIRE_STAGED_ORIGINAL"] = "1"
            os.environ["TJR_SOURCE_VIDEO_ID"] = str(staging["video_id"])
            try:
                result = render_youtube_previews(attempt_root, brief)
            except NoEditorialMoments as exc:
                failed_id = str(staging["video_id"])
                failures.append({"video_id": failed_id, "error": str(exc)})
                (attempt_root / "editorial-failure.json").write_text(
                    json.dumps(failures[-1], indent=2) + "\n", encoding="utf-8"
                )
                if explicitly_pinned:
                    raise
                excluded.add(failed_id)
                os.environ.pop("TJR_SOURCE_VIDEO_ID", None)
                continue
            print("REAL_VERIFIED_TJR_RENDER_ARTIFACTS:", result, flush=True)
            return result
        finally:
            if staging is not None:
                _purge_remote(staging)
            if original is not None:
                original.unlink(missing_ok=True)
            (attempt_root / "source.json").unlink(missing_ok=True)
            os.environ.pop("TJR_BROWSER_CAPTURE_FILE", None)
            os.environ.pop("TJR_REQUIRE_STAGED_ORIGINAL", None)
    raise NoEditorialMoments(
        "all tested approved originals failed editorial screening: "
        + json.dumps(failures, ensure_ascii=False)
    )


def main() -> int:
    requested = os.getenv("TJR_MODAL_EDITORIAL_MAX_SOURCES", "2")
    try:
        run_modal_production(max_sources=int(requested))
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"TJR_MODAL_PRODUCTION_FAILED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
