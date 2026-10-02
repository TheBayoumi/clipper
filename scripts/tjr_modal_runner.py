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
import shutil
import subprocess
import sys
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


def _pipeline_identity(source_sha: str) -> str:
    from scripts.tjr_semantic_editor import _review_model_profile

    project = Path(__file__).resolve().parents[1]
    files = [
        "scripts/tjr_modal_runner.py",
        "scripts/tjr_youtube_preview.py",
        "scripts/tjr_semantic_editor.py",
        "scripts/tjr_editorial.py",
        "scripts/tjr_quality.py",
        "scripts/tjr_visual_analysis.py",
        "campaigns/reach-double-coverage-dedicated.yaml",
        "pyproject.toml",
        *[
            str(path.relative_to(project))
            for path in sorted((project / "src/clipper").glob("*.py"))
        ],
    ]
    code = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in files}
    return hashlib.sha256(
        json.dumps(
            {
                "version": "integrated-pipeline-completion-v1",
                "source": source_sha,
                "video": os.getenv("TJR_SOURCE_VIDEO_ID", ""),
                "channel": os.getenv("TJR_MODAL_CHANNEL_ID", ""),
                "code": code,
                "reviewer_profile": _review_model_profile(),
                "settings": {
                    name: os.getenv(name, "")
                    for name in (
                        "CLIPPER_RENDER_PRESET",
                        "CLIPPER_RENDER_THREADS",
                        "TJR_CAPTION_STYLE",
                        "TJR_RENDER_SAFETY_LIMIT",
                        "CLIPPER_APPROVED_ASSET_SHA256",
                    )
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _save_pipeline_completion(result: Path, source_sha: str) -> None:
    report = result / "tjr-youtube-qa-report.json"
    if not report.is_file() or json.loads(report.read_text()).get("selected_clip_count", 0) <= 0:
        return
    files = {}
    for path in result.rglob("*"):
        if not path.is_file() or "assets" in path.relative_to(result).parts:
            continue
        if path.name == "pipeline-completion.json":
            continue
        if path.suffix not in {".json", ".mp4", ".ass", ".srt", ".png", ".txt"}:
            continue
        with path.open("rb") as media:
            files[str(path.relative_to(result))] = hashlib.file_digest(media, "sha256").hexdigest()
    (result / "pipeline-completion.json").write_text(
        json.dumps(
            {
                "identity": _pipeline_identity(source_sha),
                "source_sha256": source_sha,
                "complete": True,
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )


def _restore_completed_pipeline(root: Path) -> bool:
    cached_root = os.getenv("TJR_RENDER_CACHE_ROOT", "").strip()
    if not cached_root or not os.getenv("TJR_SOURCE_VIDEO_ID", "").strip():
        return False
    for manifest in Path(cached_root).rglob("pipeline-completion.json"):
        saved = json.loads(manifest.read_text())
        if saved.get("complete") is not True or saved.get("identity") != _pipeline_identity(
            saved.get("source_sha256", "")
        ):
            continue
        files = saved.get("files", {})
        if not files or any(Path(name).is_absolute() or ".." in Path(name).parts for name in files):
            continue
        valid = True
        for name, digest in files.items():
            path = manifest.parent / name
            if not path.is_file():
                valid = False
                break
            with path.open("rb") as media:
                if hashlib.file_digest(media, "sha256").hexdigest() != digest:
                    valid = False
                    break
        if not valid:
            continue
        destination = root / "attempt-1" / manifest.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            out = destination / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest.parent / name, out)
        shutil.copyfile(manifest, destination / manifest.name)
        (destination / "pipeline-state.json").write_text(
            json.dumps(
                {
                    "source_sha256": saved["source_sha256"],
                    "pipeline": "complete_cache_hit",
                    "source_acquisition": "skipped",
                    "transcript": "cache_hit",
                    "editorial": "cache_hit",
                    "rendering": "cache_hit",
                },
                indent=2,
            )
            + "\n"
        )
        _save_pipeline_completion(destination, saved["source_sha256"])
        print(
            "PIPELINE_CACHE_HIT: source acquisition, transcript, models and rendering skipped",
            flush=True,
        )
        return True
    return False


def _restore_source_cache(root: Path) -> dict[str, Any] | None:
    cache_root = os.getenv("TJR_SOURCE_CACHE_ROOT", "").strip()
    requested = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip()
    if not requested:
        return None
    manifests = list(Path(cache_root).rglob("staged-original.json")) if cache_root else []
    channel = os.getenv("TJR_MODAL_CHANNEL_ID", "")
    if (
        os.getenv("TJR_PERSIST_SOURCE") == "1"
        and re.fullmatch(r"[A-Za-z0-9_-]{11}", requested)
        and re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel)
    ):
        root.mkdir(parents=True, exist_ok=True)
        index = root / "cached-source-index.json"
        result = subprocess.run(
            [
                "modal",
                "volume",
                "get",
                VOLUME,
                f"cache/{channel}/{requested}/manifest.json",
                str(index),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and index.is_file():
            manifests.insert(0, index)
    for manifest in manifests:
        try:
            cached = json.loads(manifest.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(cached, dict):
            continue
        if cached.get("video_id") != requested or cached.get("channel_id") != os.getenv(
            "TJR_MODAL_CHANNEL_ID"
        ):
            continue
        cached.pop("source_cache_local_path", None)
        try:
            local = _transfer_verified_original(cached, root / "cached-source", verify_media=False)
        except (RuntimeError, OSError, subprocess.CalledProcessError):
            print("SOURCE_CACHE_MISS: unavailable or invalid original bytes", flush=True)
            continue
        cached["source_cache_local_path"] = str(local.resolve())
        cached["source_cache_reused"] = True
        root.mkdir(parents=True, exist_ok=True)
        (root / "staged-original.json").write_text(json.dumps(cached, indent=2) + "\n")
        print(
            "SOURCE_CACHE_HIT: exact verified bytes restored; YouTube download skipped", flush=True
        )
        return cached
    return None


def _acquire_original(excluded: set[str], root: Path) -> dict[str, Any]:
    if os.getenv("TJR_MODAL_USE_STAGED") == "1":
        return _load_staged_original(root)
    cached = _restore_source_cache(root)
    if cached is not None:
        return cached
    env = {
        **os.environ,
        "TJR_MODAL_STAGE_ORIGINAL": "1",
        "TJR_MODAL_EXCLUDE_VIDEO_IDS": ",".join(sorted(excluded)),
    }
    stage = root / "staged-original.json"
    report_path = root / "verified-original-egress.json"
    requested = env.get("TJR_SOURCE_VIDEO_ID", "").strip()
    # Only transient challenges for one explicitly selected source may retry.
    # Each Modal app invocation keeps the same GCP route and original image.
    limit = 3 if requested else 1
    for attempt in range(1, limit + 1):
        stage.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        command = ["modal", "run", "-m", "scripts.tjr_modal_probe"]
        result = subprocess.run(command, env=env, check=False)
        report: dict[str, Any] = {}
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            (root / f"acquisition-attempt-{attempt}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        if result.returncode == 0:
            staged = _load_staged_original(root)
            if requested and staged.get("video_id") != requested:
                raise RuntimeError("acquisition retry changed the explicitly selected source")
            return staged
        attempts = report.get("attempts") or []
        retryable = (
            report.get("status") == "YOUTUBE_EGRESS_BOT_CHALLENGE"
            and bool(attempts)
            and all(
                item.get("url") == f"https://www.youtube.com/watch?v={requested}"
                and item.get("stage") == "metadata"
                and item.get("reason") == "YOUTUBE_IP_OR_LOGIN_CHALLENGE"
                for item in attempts
            )
        )
        if not retryable or attempt == limit:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(
            f"YOUTUBE_TRANSIENT_CHALLENGE_RETRY: {attempt + 1}/{limit} same source and GCP route",
            flush=True,
        )
    raise RuntimeError("acquisition exhausted its bounded retry limit")


def _transfer_verified_original(
    staging: dict[str, Any], destination: Path, *, verify_media: bool = True
) -> Path:
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
        or channel_id not in {"UCf1q6dhccWr6eQEcFFnJSbA"}
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise RuntimeError("Modal original manifest failed channel, source or hash validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = destination.with_suffix(Path(remote).suffix)
    try:
        cached_local = staging.get("source_cache_local_path")
        if cached_local and Path(cached_local).is_file():
            import shutil

            shutil.copyfile(cached_local, original)
        else:
            subprocess.run(["modal", "volume", "get", VOLUME, remote, str(original)], check=True)
        if not original.is_file():
            raise RuntimeError("Modal volume transfer did not produce an original")
        with original.open("rb") as src:
            actual = hashlib.file_digest(src, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError("Modal original source transfer SHA-256 mismatch")
        if verify_media:
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
    brief: Path = Path("campaigns/reach-double-coverage-dedicated.yaml"),
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
        if os.getenv("TJR_PERSIST_SOURCE") == "1":
            # Retain the verified remote bytes and a deterministic index, so reuse
            # survives GitHub artifact expiry and explicitly selected old checkpoints.
            index = probe_root / "source-cache-index.json"
            probe_root.mkdir(parents=True, exist_ok=True)
            retained = {
                key: value for key, value in staging.items() if key != "source_cache_local_path"
            }
            index.write_text(json.dumps(retained, indent=2) + "\n")
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "put",
                    VOLUME,
                    str(index),
                    f"cache/{staging['channel_id']}/{staging['video_id']}/manifest.json",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        env_manifest = str((attempt_root / "source.json").resolve())
        # This exact source is authoritative after verification. Editorial weakness
        # must produce an audited zero-clip result, never a different source.
        os.environ["TJR_BROWSER_CAPTURE_FILE"] = env_manifest
        os.environ["TJR_REQUIRE_STAGED_ORIGINAL"] = "1"
        os.environ["TJR_SOURCE_VIDEO_ID"] = str(staging["video_id"])
        probe_root.mkdir(parents=True, exist_ok=True)
        (probe_root / "pipeline-source-state.json").write_text(
            json.dumps(
                {
                    "source_sha256": staging["source_sha256"],
                    "video_id": staging["video_id"],
                    "source": "cache_hit" if staging.get("source_cache_reused") else "acquired",
                    "remote_original_retained": os.getenv("TJR_PERSIST_SOURCE") == "1",
                },
                indent=2,
            )
            + "\n"
        )
        result = render_youtube_previews(attempt_root, brief)
        _save_pipeline_completion(result, staging["source_sha256"])
        print("REAL_VERIFIED_TJR_RENDER_ARTIFACTS:", result, flush=True)
        return result
    finally:
        try:
            if staging is not None and os.getenv("TJR_PERSIST_SOURCE") != "1":
                _purge_remote(staging, attempt_root / "remote-cleanup-error.json")
        finally:
            if original is not None:
                original.unlink(missing_ok=True)
            (attempt_root / "source.json").unlink(missing_ok=True)
            os.environ.pop("TJR_BROWSER_CAPTURE_FILE", None)
            os.environ.pop("TJR_REQUIRE_STAGED_ORIGINAL", None)
            os.environ.pop("TJR_SOURCE_VIDEO_ID", None)


def prepare_watermark_cache(brief_path: Path, manifest_path: Path) -> None:
    """Validate the exact approved image before acquiring a costly video original."""
    from clipper.brief import load_brief
    from clipper.pipeline import _download_asset, _verify_image_asset

    brief = load_brief(brief_path)
    expected = os.environ.get("CLIPPER_APPROVED_ASSET_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not brief.watermark_url:
        raise RuntimeError("approved watermark URL and pinned SHA-256 are required")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    local = manifest_path.with_suffix(".png")
    remote = f"assets/{expected}.png"
    result = subprocess.run(
        ["modal", "volume", "get", VOLUME, remote, str(local)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        local.unlink(missing_ok=True)
        bootstrap = os.getenv("CLIPPER_APPROVED_ASSET_BOOTSTRAP_URL", "").strip()
        if not bootstrap:
            raise RuntimeError("approved watermark is absent from the persistent asset cache")
        try:
            _download_asset(bootstrap, local, expected_kind="media")
            _verify_image_asset(local, expected)
            subprocess.run(
                ["modal", "volume", "put", VOLUME, str(local), remote],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            local.unlink(missing_ok=True)
            raise RuntimeError(
                "approved watermark bootstrap failed validation or storage"
            ) from None
    _verify_image_asset(local, expected)
    manifest_path.write_text(
        json.dumps(
            {
                "source_url": brief.watermark_url,
                "sha256": expected,
                "path": str(local.resolve()),
                "cache_path": remote,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("APPROVED_WATERMARK_VERIFIED:", expected, flush=True)


def main() -> int:
    try:
        if sys.argv[1:] == ["--prepare-watermark"]:
            prepare_watermark_cache(
                Path("campaigns/reach-double-coverage-dedicated.yaml"),
                Path(os.environ["CLIPPER_IMAGE_ASSET_CACHE_MANIFEST"]),
            )
        elif sys.argv[1:] == ["--acquire-original"]:
            reused = _restore_completed_pipeline(Path("tjr-modal-artifacts"))
            output = os.getenv("GITHUB_OUTPUT", "")
            if output:
                with Path(output).open("a") as file:
                    file.write(f"pipeline_reused={'true' if reused else 'false'}\n")
            if not reused:
                _acquire_original(set(), Path("tjr-modal-probe"))
        else:
            run_modal_production()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"TJR_MODAL_PRODUCTION_FAILED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
