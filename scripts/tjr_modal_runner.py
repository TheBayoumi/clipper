"""GitHub-runner orchestration of one verified Modal original and shared TJR editing.

Source discovery may skip ineligible live/short/unavailable uploads before staging,
but once one original is verified and staged, editorial weakness never causes a
silent switch to an older video. Zero creator-grade clips is a valid no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from clipper.brief import load_brief
from clipper.editorial_run import (
    EditorialRunConfig,
    load_editorial_run_config,
    write_editorial_run_config,
)
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


def _pipeline_identity(source_sha: str, *, run_config: EditorialRunConfig | None = None) -> str:
    from scripts.tjr_semantic_editor import _review_model_profile

    project = Path(__file__).resolve().parents[1]
    files = [
        "scripts/tjr_modal_runner.py",
        "scripts/tjr_youtube_preview.py",
        "scripts/tjr_semantic_editor.py",
        "src/clipper/editorial.py",
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
    settings = {
        name: os.getenv(name, "")
        for name in (
            "CLIPPER_RENDER_PRESET",
            "CLIPPER_RENDER_THREADS",
            "TJR_CAPTION_STYLE",
            "TJR_RENDER_SAFETY_LIMIT",
            "CLIPPER_APPROVED_ASSET_SHA256",
        )
    }
    if run_config is not None:
        settings["TJR_CAPTION_STYLE"] = run_config.caption_style
        settings["TJR_RENDER_SAFETY_LIMIT"] = str(run_config.render_safety_limit)
    return hashlib.sha256(
        json.dumps(
            {
                "version": "integrated-pipeline-completion-v1",
                "source": source_sha,
                "video": (
                    run_config.source_video_id
                    if run_config is not None
                    else os.getenv("TJR_SOURCE_VIDEO_ID", "")
                ),
                "channel": (
                    run_config.target_channel_id
                    if run_config is not None
                    else os.getenv("TJR_MODAL_CHANNEL_ID", "")
                ),
                "code": code,
                "reviewer_profile": _review_model_profile(),
                "settings": settings,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _save_pipeline_completion(
    result: Path, source_sha: str, *, run_config: EditorialRunConfig | None = None
) -> None:
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
                "identity": _pipeline_identity(source_sha, run_config=run_config),
                "source_sha256": source_sha,
                "complete": True,
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )


def _restore_completed_pipeline(root: Path) -> bool:
    # A requested fresh assessment must not restore old completed render output.
    if os.getenv("TJR_FORCE_FULL_CANDIDATE_REASSESSMENT", "").lower() == "true":
        return False
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


# A verified ~330 MB source should never be fetched twice or left waiting for
# the Modal CLI's unbounded internal retries. Keep this below the editorial gate.
_SOURCE_TRANSFER_TIMEOUT_SECONDS = 600


def _source_transfer_timeout() -> int:
    setting = os.getenv("TJR_SOURCE_TRANSFER_TIMEOUT_SECONDS", "600")
    try:
        seconds = int(setting)
    except ValueError as exc:
        raise RuntimeError("SOURCE_TRANSFER_TIMEOUT_INVALID: expected integer seconds") from exc
    if not 30 <= seconds <= 1800:
        raise RuntimeError("SOURCE_TRANSFER_TIMEOUT_INVALID: permitted range 30..1800 seconds")
    return seconds


def _recover_source_from_fallback(
    staged: dict[str, Any], destination: Path, *, fallback_root: str
) -> Path | None:
    """Restore GitHub's separately archived verified original, never arbitrary media."""
    if not fallback_root:
        return None
    source = Path(fallback_root) / (
        "cached-source" + Path(str(staged.get("source_remote_path", ""))).suffix
    )
    manifest = Path(fallback_root) / "staged-original.json"
    if not source.is_file() or not manifest.is_file():
        return None
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"SOURCE_FALLBACK_MANIFEST_INVALID: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(saved, dict) or any(
        saved.get(key) != staged.get(key) for key in ("video_id", "channel_id", "source_sha256")
    ):
        raise RuntimeError("SOURCE_FALLBACK_IDENTITY_MISMATCH: refusing unrelated archived source")
    expected = str(staged["source_sha256"])
    with source.open("rb") as video:
        actual = hashlib.file_digest(video, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SOURCE_FALLBACK_SHA256_MISMATCH: expected={expected} actual={actual}")
    print(f"SOURCE_FALLBACK_VERIFIED: {source.name} sha256={actual}", flush=True)
    return source


def _restore_source_cache(
    root: Path, *, env: dict[str, str] | None = None
) -> dict[str, Any] | None:
    settings = os.environ if env is None else env
    cache_root = settings.get("TJR_SOURCE_CACHE_ROOT", "").strip()
    requested = settings.get("TJR_SOURCE_VIDEO_ID", "").strip()
    if not requested:
        return None
    channel = settings.get("TJR_MODAL_CHANNEL_ID", "")
    manifests: list[Path] = []
    failures: list[dict[str, str]] = []
    if (
        settings.get("TJR_PERSIST_SOURCE") == "1"
        and re.fullmatch(r"[A-Za-z0-9_-]{11}", requested)
        and re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel)
    ):
        root.mkdir(parents=True, exist_ok=True)
        index = root / "cached-source-index.json"
        try:
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
                manifests.append(index)
            elif result.returncode != 0:
                failures.append(
                    {
                        "candidate": "modal-source-index",
                        "error": f"exit={result.returncode} stderr={result.stderr[-450:]}",
                    }
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(
                {"candidate": "modal-source-index", "error": f"{type(exc).__name__}: {exc}"}
            )
    if cache_root:
        manifests.extend(sorted(Path(cache_root).rglob("staged-original.json")))
    # The remote index and prior GitHub manifest often reference the SAME remote
    # original. Trying the same ~330 MB object twice created the 94-minute failure.
    attempted: set[tuple[str, str]] = set()
    for manifest in manifests:
        try:
            cached = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            failures.append({"candidate": str(manifest), "error": f"manifest: {exc}"})
            continue
        if not isinstance(cached, dict):
            failures.append({"candidate": str(manifest), "error": "manifest is not an object"})
            continue
        if cached.get("video_id") != requested or cached.get("channel_id") != channel:
            continue
        key = (str(cached.get("source_remote_path")), str(cached.get("source_sha256")))
        if key in attempted:
            print(f"SOURCE_CACHE_DUPLICATE_SKIPPED: {key[0]}", flush=True)
            continue
        attempted.add(key)
        cached.pop("source_cache_local_path", None)
        destination = root / "cached-source"
        try:
            fallback = _recover_source_from_fallback(
                cached,
                destination.with_suffix(Path(str(cached.get("source_remote_path"))).suffix),
                fallback_root=settings.get("TJR_SOURCE_FALLBACK_ROOT", "").strip(),
            )
            if fallback is not None:
                cached["source_cache_local_path"] = str(fallback.resolve())
                # Every fallback is copied and checked through the SAME proof path.
                local = _transfer_verified_original(cached, destination, verify_media=False)
            else:
                print(
                    f"SOURCE_CACHE_TRANSFER_START: remote={key[0]} "
                    f"deadline_seconds={_source_transfer_timeout()}",
                    flush=True,
                )
                local = _transfer_verified_original(cached, destination, verify_media=False)
        except (
            RuntimeError,
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failures.append({"candidate": key[0], "error": failure[-1400:]})
            print(
                f"SOURCE_CACHE_RESTORE_FAILED: remote={key[0]} detail={failure[-1400:]}",
                flush=True,
            )
            continue
        cached["source_cache_local_path"] = str(local.resolve())
        cached["source_cache_reused"] = True
        root.mkdir(parents=True, exist_ok=True)
        (root / "staged-original.json").write_text(json.dumps(cached, indent=2) + "\n")
        print(
            "SOURCE_CACHE_HIT: exact verified bytes restored; YouTube download skipped",
            flush=True,
        )
        return cached
    if failures:
        root.mkdir(parents=True, exist_ok=True)
        report = root / "source-cache-diagnostics.json"
        report.write_text(
            json.dumps(
                {
                    "video_id": requested,
                    "channel_id": channel,
                    "unique_media_attempts": len(attempted),
                    "failures": failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"SOURCE_CACHE_DIAGNOSTICS: {report}", flush=True)
    return None


def _acquire_original(
    excluded: set[str], root: Path, *, env_overrides: dict[str, str] | None = None
) -> dict[str, Any]:
    env = {**os.environ, **(env_overrides or {})}
    if env.get("TJR_MODAL_USE_STAGED") == "1":
        return _load_staged_original(root)
    cached = _restore_source_cache(root, env=env)
    if cached is not None:
        return cached
    # Explicit cached-editorial recovery is not permission to reacquire the
    # entire podcast if its persisted original has gone missing.
    if env.get("TJR_EDITORIAL_CACHE_ROOT") and env.get("TJR_SOURCE_VIDEO_ID"):
        raise RuntimeError(
            "EDITORIAL_RECOVERY_SOURCE_CACHE_MISSING: exact original unavailable; "
            "refusing YouTube reacquisition"
        )
    env["TJR_MODAL_STAGE_ORIGINAL"] = "1"
    env["TJR_MODAL_EXCLUDE_VIDEO_IDS"] = ",".join(sorted(excluded))
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
            print(
                f"MODAL_SOURCE_TRANSFER: remote={remote} timeout={_source_transfer_timeout()}s",
                flush=True,
            )
            try:
                transfer = subprocess.run(
                    ["modal", "volume", "get", VOLUME, remote, str(original)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=_source_transfer_timeout(),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"MODAL_SOURCE_TRANSFER_TIMEOUT: remote={remote} "
                    f"deadline={_source_transfer_timeout()}s"
                ) from exc
            if transfer.returncode != 0:
                raise RuntimeError(
                    f"MODAL_SOURCE_TRANSFER_FAILED: remote={remote} "
                    f"exit={transfer.returncode} stderr={transfer.stderr[-900:]}"
                )
        if not original.is_file():
            raise RuntimeError("Modal volume transfer did not produce an original")
        with original.open("rb") as src:
            actual = hashlib.file_digest(src, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"MODAL_SOURCE_SHA256_MISMATCH: remote={remote} "
                f"expected={expected} actual={actual} size={original.stat().st_size}"
            )
        expected_size = staging.get("size_bytes")
        if (
            type(expected_size) is int
            and expected_size > 0
            and original.stat().st_size != expected_size
        ):
            raise RuntimeError(
                f"MODAL_SOURCE_SIZE_MISMATCH: remote={remote} "
                f"expected={expected_size} actual={original.stat().st_size}"
            )
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


def _persist_source_cache_index(staging: dict[str, Any], probe_root: Path) -> str:
    """Idempotently retain verified original provenance without rewriting cached bytes.

    A matching manifest fetched from Modal is already authoritative after the
    original has passed SHA-256 verification. When a saved GitHub artifact is
    the only available manifest, refresh the remote index explicitly: Modal's
    volume put refuses to overwrite existing files without --force.
    """
    retained = {
        key: value
        for key, value in staging.items()
        if key not in {"source_cache_local_path", "source_cache_reused"}
    }
    remote_index = probe_root / "cached-source-index.json"
    if staging.get("source_cache_reused") is True and remote_index.is_file():
        try:
            old_index = json.loads(remote_index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_index = None
        if isinstance(old_index, dict) and all(
            old_index.get(name) == retained.get(name)
            for name in (
                "video_id",
                "channel_id",
                "source_remote_path",
                "source_sha256",
                "public_video_url",
                "status",
            )
        ):
            print(
                "SOURCE_MANIFEST_CACHE_HIT: existing Modal index matches verified source",
                flush=True,
            )
            return "cache_hit"
    probe_root.mkdir(parents=True, exist_ok=True)
    index = probe_root / "source-cache-index.json"
    index.write_text(json.dumps(retained, indent=2) + "\n", encoding="utf-8")
    remote_path = f"cache/{staging['channel_id']}/{staging['video_id']}/manifest.json"
    command = ["modal", "volume", "put", "--force", VOLUME, str(index), remote_path]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "MODAL_SOURCE_MANIFEST_UPLOAD_FAILED: "
            f"exit={completed.returncode} "
            f"stderr={completed.stderr[-1000:]}"
        )
    print("SOURCE_MANIFEST_UPSERT: verified exact-source index retained", flush=True)
    return "uploaded"


def run_modal_production(
    *,
    root: Path | None = None,
    brief: Path | None = None,
    probe_root: Path = Path("tjr-modal-probe"),
    run_config: EditorialRunConfig | None = None,
) -> Path:
    if run_config is not None:
        run_config.validate()
        configured_channel = run_config.target_channel_id
        if not configured_channel:
            channels = load_brief(run_config.brief).source_channel_ids
            if len(channels) != 1:
                raise ValueError(
                    "configured Modal run needs target_channel_id when brief has multiple channels"
                )
            configured_channel = channels[0]
        if root is not None and root.resolve() != run_config.artifact_root.resolve():
            raise ValueError("Modal artifact root differs from the validated run config")
        if brief is not None and brief.resolve() != run_config.brief.resolve():
            raise ValueError("Modal brief differs from the validated run config")
        root, brief = run_config.artifact_root, run_config.brief
    root = root or Path("tjr-modal-artifacts")
    brief = brief or Path("campaigns/reach-double-coverage-dedicated.yaml")
    root.mkdir(parents=True, exist_ok=True)
    attempt_root = root / "attempt-1"
    attempt_root.mkdir(parents=True, exist_ok=True)
    staging: dict[str, Any] | None = None
    original: Path | None = None
    legacy_source_environment = (
        {
            name: os.environ.get(name)
            for name in (
                "TJR_BROWSER_CAPTURE_FILE",
                "TJR_REQUIRE_STAGED_ORIGINAL",
                "TJR_SOURCE_VIDEO_ID",
            )
        }
        if run_config is None
        else None
    )
    try:
        acquisition_overrides = (
            {
                "TJR_SOURCE_VIDEO_ID": run_config.source_video_id,
                "TJR_MODAL_CHANNEL_ID": configured_channel,
            }
            if run_config is not None
            else None
        )
        staging = _acquire_original(set(), probe_root, env_overrides=acquisition_overrides)
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
        manifest_persistence = (
            _persist_source_cache_index(staging, probe_root)
            if os.getenv("TJR_PERSIST_SOURCE") == "1"
            else "not_requested"
        )
        env_manifest = str((attempt_root / "source.json").resolve())
        # This exact source is authoritative after verification. Editorial weakness
        # must produce an audited zero-clip result, never a different source.
        if run_config is None:
            os.environ["TJR_BROWSER_CAPTURE_FILE"] = env_manifest
            os.environ["TJR_REQUIRE_STAGED_ORIGINAL"] = "1"
            os.environ["TJR_SOURCE_VIDEO_ID"] = str(staging["video_id"])
        selected_config = None
        if run_config is not None:
            if run_config.source_video_id and run_config.source_video_id != staging["video_id"]:
                raise RuntimeError("configured source differs from verified staged original")
            if configured_channel != staging["channel_id"]:
                raise RuntimeError("configured channel differs from verified staged original")
            selected_config = replace(
                run_config,
                artifact_root=attempt_root,
                source_video_id=str(staging["video_id"]),
                target_channel_id=str(staging["channel_id"]),
                browser_capture_file=Path(env_manifest),
                require_staged_original=True,
            )
            selected_config.validate()
            write_editorial_run_config(selected_config, attempt_root / "editorial-run.json")
        probe_root.mkdir(parents=True, exist_ok=True)
        (probe_root / "pipeline-source-state.json").write_text(
            json.dumps(
                {
                    "source_sha256": staging["source_sha256"],
                    "video_id": staging["video_id"],
                    "source": "cache_hit" if staging.get("source_cache_reused") else "acquired",
                    "remote_original_retained": os.getenv("TJR_PERSIST_SOURCE") == "1",
                    "source_manifest_persistence": manifest_persistence,
                },
                indent=2,
            )
            + "\n"
        )
        result: Path = render_youtube_previews(attempt_root, brief, run_config=selected_config)
        _save_pipeline_completion(result, staging["source_sha256"], run_config=selected_config)
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
            if legacy_source_environment is not None:
                for name, value in legacy_source_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-watermark", action="store_true")
    actions.add_argument("--acquire-original", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.config is not None and (args.prepare_watermark or args.acquire_original):
            raise ValueError("--config applies to the editorial/render stage only")
        if args.prepare_watermark:
            prepare_watermark_cache(
                Path("campaigns/reach-double-coverage-dedicated.yaml"),
                Path(os.environ["CLIPPER_IMAGE_ASSET_CACHE_MANIFEST"]),
            )
        elif args.acquire_original:
            reused = _restore_completed_pipeline(Path("tjr-modal-artifacts"))
            output = os.getenv("GITHUB_OUTPUT", "")
            if output:
                with Path(output).open("a") as file:
                    file.write(f"pipeline_reused={'true' if reused else 'false'}\n")
            if not reused:
                probe = Path("tjr-modal-probe")
                staging = _acquire_original(set(), probe)
                local = staging.get("source_cache_local_path")
                if not isinstance(local, str) or not Path(local).is_file():
                    verified = _transfer_verified_original(
                        staging, probe / "cached-source", verify_media=False
                    )
                    staging["source_cache_local_path"] = str(verified.resolve())
                    probe.joinpath("staged-original.json").write_text(
                        json.dumps(staging, indent=2) + "\n", encoding="utf-8"
                    )
                print(
                    "SOURCE_PREFLIGHT_VERIFIED: exact original SHA-256 verified "
                    "and durable fallback staging prepared",
                    flush=True,
                )
        else:
            config = load_editorial_run_config(args.config) if args.config is not None else None
            run_modal_production(run_config=config)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"TJR_MODAL_PRODUCTION_FAILED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
