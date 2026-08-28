from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.getenv("CLIPPER_MODAL_PIPELINE_APP", "clipper-production-pipeline")
MODEL_APP = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
DEPLOYED_GIT_SHA = os.getenv("CLIPPER_DEPLOYED_GIT_SHA", "").strip().lower()
MEDIA_ROOT = "/media"
ARTIFACT_ROOT = "/artifacts"

app = modal.App(APP_NAME)
media_cache = modal.Volume.from_name("clipper-media-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("clipper-production-artifacts", create_if_missing=True)

if modal.is_local():
    youtube_secret = modal.Secret.from_dict(
        {"CLIPPER_YOUTUBE_COOKIES_B64": os.environ.get("CLIPPER_YOUTUBE_COOKIES_B64", "")}
    )
else:
    youtube_secret = modal.Secret.from_dict({})

media_image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
    .env({"CLIPPER_DEPLOYED_GIT_SHA": DEPLOYED_GIT_SHA})
    .apt_install("ffmpeg", "git")
    .uv_pip_install(
        "yt-dlp[default]>=2026.7.4,<2027",
        "bgutil-ytdlp-pot-provider==1.3.1",
    )
    .run_commands(
        "git clone --depth 1 --branch 1.3.1 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "
        "/root/bgutil-ytdlp-pot-provider",
        "cd /root/bgutil-ytdlp-pot-provider/server && npm ci && npx tsc",
    )
)

runner_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install("modal>=1.5.2,<2", "huggingface-hub>=1.24,<2")
    .env({"CLIPPER_DEPLOYED_GIT_SHA": DEPLOYED_GIT_SHA})
)


def _assert_expected_git_sha(payload: dict[str, Any]) -> None:
    expected = str(payload.get("expected_git_sha") or "").strip().lower()
    if not expected:
        return
    if not DEPLOYED_GIT_SHA:
        raise RuntimeError("production pipeline worker has no embedded deployment SHA")
    if expected != DEPLOYED_GIT_SHA:
        raise RuntimeError(
            "production pipeline worker SHA mismatch: "
            f"expected={expected} deployed={DEPLOYED_GIT_SHA}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("ffprobe returned invalid source metadata")
    return payload


def _source_evidence(
    path: Path,
    *,
    video_id: str,
    video_url: str,
    volume_path: str,
    authenticated: bool,
    reused: bool,
) -> dict[str, Any]:
    payload = _probe(path)
    streams = payload.get("streams")
    stream_list = streams if isinstance(streams, list) else []
    video = next(
        (
            item
            for item in stream_list
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        {},
    )
    audio = next(
        (
            item
            for item in stream_list
            if isinstance(item, dict) and item.get("codec_type") == "audio"
        ),
        {},
    )
    raw_format = payload.get("format")
    fmt = raw_format if isinstance(raw_format, dict) else {}
    return {
        "video_id": video_id,
        "source_url": video_url,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "volume_path": volume_path,
        "mount_path": str(path),
        "container": path.suffix.lower().lstrip(".") or "mkv",
        "quality_policy": "highest_available_no_transcode",
        "authentication": "youtube_cookies" if authenticated else "anonymous",
        "authenticated": authenticated,
        "reused": reused,
        "duration_seconds": float(fmt.get("duration") or 0.0),
        "video": {
            "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
            "codec": str(video.get("codec_name") or "unknown"),
            "pixel_format": str(video.get("pix_fmt") or "unknown"),
            "frame_rate": str(video.get("avg_frame_rate") or "unknown"),
        },
        "audio": {
            "codec": str(audio.get("codec_name") or "unknown"),
            "sample_rate": str(audio.get("sample_rate") or "unknown"),
            "channels": int(audio.get("channels") or 0),
        },
    }


def _existing_source(video_id: str, video_url: str) -> dict[str, Any] | None:
    index_path = Path(MEDIA_ROOT) / "source-index" / f"{video_id}.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(index, dict):
        return None
    volume_path = str(index.get("volume_path") or "")
    expected = str(index.get("sha256") or "")
    target = Path(MEDIA_ROOT) / volume_path.lstrip("/")
    if not volume_path.startswith("/inputs/") or not target.is_file() or not expected:
        return None
    if _sha256(target) != expected:
        return None
    evidence = _source_evidence(
        target,
        video_id=video_id,
        video_url=video_url,
        volume_path=volume_path,
        authenticated=bool(index.get("authenticated")),
        reused=True,
    )
    return evidence if evidence["sha256"] == expected else None


@app.function(
    image=media_image,
    volumes={MEDIA_ROOT: media_cache},
    secrets=[youtube_secret],
    timeout=7200,
    memory=4096,
    scaledown_window=2,
)
def acquire_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Acquire one exact authorized target into the content-addressed media volume."""
    _assert_expected_git_sha(payload)
    video_url = str(payload.get("video_url") or "").strip()
    video_id = str(payload.get("video_id") or "").strip()
    if not video_url.startswith("https://"):
        raise ValueError("source acquisition requires an https video_url")
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not video_id or any(character not in safe for character in video_id):
        raise ValueError("source acquisition requires a safe video_id")

    media_cache.reload()
    existing = _existing_source(video_id, video_url)
    if existing is not None:
        return existing

    staging = Path(MEDIA_ROOT) / "staging" / video_id
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    output_template = staging / "source.%(ext)s"
    cookie_path = staging / "youtube.cookies.txt"
    encoded_cookies = os.getenv("CLIPPER_YOUTUBE_COOKIES_B64", "").strip()
    cookies_available = bool(encoded_cookies)
    if cookies_available:
        try:
            cookie_bytes = base64.b64decode(encoded_cookies, validate=True)
        except ValueError as exc:
            raise RuntimeError("CLIPPER_YOUTUBE_COOKIES_B64 is not valid base64") from exc
        if not cookie_bytes.strip():
            raise RuntimeError("CLIPPER_YOUTUBE_COOKIES_B64 decoded to an empty cookie file")
        cookie_path.write_bytes(cookie_bytes)
        cookie_path.chmod(0o600)

    provider_arg = "youtubepot-bgutilscript:server_home=/root/bgutil-ytdlp-pot-provider/server"
    attempts: list[tuple[str, list[str], bool]] = [
        (
            "bgutil_default_mweb",
            [
                "--extractor-args",
                "youtube:player_client=default,mweb",
                "--extractor-args",
                provider_arg,
            ],
            False,
        ),
        ("bgutil_default_clients", ["--extractor-args", provider_arg], False),
        (
            "bgutil_embedded_android_vr",
            [
                "--extractor-args",
                "youtube:player_client=web_embedded,android_vr",
                "--extractor-args",
                provider_arg,
            ],
            False,
        ),
    ]
    if cookies_available:
        attempts.append(
            (
                "cookies_bgutil_default_mweb",
                [
                    "--extractor-args",
                    "youtube:player_client=default,mweb",
                    "--extractor-args",
                    provider_arg,
                ],
                True,
            )
        )

    base_command = [
        "yt-dlp",
        "--verbose",
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--concurrent-fragments",
        "4",
        "--format",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mkv",
        "--output",
        str(output_template),
        "--print",
        "after_move:filepath",
    ]
    completed: subprocess.CompletedProcess[str] | None = None
    authenticated = False
    acquisition_errors: list[str] = []
    try:
        for strategy, extractor_options, use_cookies in attempts:
            for partial in staging.glob("source.*"):
                partial.unlink(missing_ok=True)
            command = [*base_command, *extractor_options]
            if use_cookies:
                command.extend(["--cookies", str(cookie_path)])
            command.append(video_url)
            try:
                candidate = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1500,
                )
            except subprocess.CalledProcessError as exc:
                detail = "\n".join(
                    part.strip() for part in (exc.stderr, exc.stdout) if part and part.strip()
                )[-6000:]
                acquisition_errors.append(f"[{strategy}] {detail}")
                continue
            completed = candidate
            authenticated = use_cookies
            break
    finally:
        cookie_path.unlink(missing_ok=True)

    if completed is None:
        detail = "\n\n".join(acquisition_errors)[-12000:]
        raise RuntimeError(
            "yt-dlp source acquisition exhausted all configured strategies"
            + (" including authenticated fallback" if cookies_available else "")
            + f":\n{detail}"
        )

    printed = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    source = printed[-1] if printed else Path()
    if not source.is_file() or source.stat().st_size <= 0:
        candidates = [path for path in staging.glob("source.*") if path.is_file()]
        if not candidates:
            raise RuntimeError("yt-dlp completed without creating a source master")
        source = max(candidates, key=lambda item: item.stat().st_size)

    digest = _sha256(source)
    suffix = source.suffix.lower() or ".mkv"
    volume_path = f"/inputs/{digest}{suffix}"
    target = Path(MEDIA_ROOT) / volume_path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if _sha256(target) != digest:
            raise RuntimeError("content-addressed source master hash mismatch")
        source.unlink(missing_ok=True)
    else:
        source.replace(target)

    evidence = _source_evidence(
        target,
        video_id=video_id,
        video_url=video_url,
        volume_path=volume_path,
        authenticated=authenticated,
        reused=False,
    )
    if evidence["video"]["width"] <= 0 or evidence["video"]["height"] <= 0:
        raise RuntimeError("source master has no decodable video stream")
    index_path = Path(MEDIA_ROOT) / "source-index" / f"{video_id}.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    target.with_suffix(".source.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    media_cache.commit()
    shutil.rmtree(staging, ignore_errors=True)
    return evidence


class VolumeSourceClient:
    """Serve every exact pre-acquired target from the mounted content-addressed volume."""

    def __init__(self, source_items: list[dict[str, Any]]) -> None:
        from clipper.models import VideoCandidate

        self._records: dict[str, dict[str, Any]] = {}
        self._videos: list[Any] = []
        for item in source_items:
            evidence = item.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError("source item requires evidence")
            video_id = str(item.get("video_id") or evidence.get("video_id") or "").strip()
            channel_id = str(item.get("channel_id") or "").strip()
            canonical_url = str(
                item.get("canonical_url") or evidence.get("source_url") or ""
            ).strip()
            source_path = Path(str(evidence.get("mount_path") or ""))
            source_sha256 = str(evidence.get("sha256") or "")
            duration = float(evidence.get("duration_seconds") or 0.0)
            if not video_id or not channel_id or not canonical_url.startswith("https://"):
                raise ValueError("source item requires video_id, channel_id, and canonical_url")
            if video_id in self._records:
                raise ValueError(f"duplicate mounted source video_id: {video_id}")
            if evidence.get("quality_policy") != "highest_available_no_transcode":
                raise RuntimeError(f"source master violates no-downgrade policy: {video_id}")
            self._records[video_id] = {
                "path": source_path,
                "sha256": source_sha256,
                "duration": duration,
                "evidence": evidence,
            }
            self._videos.append(
                VideoCandidate(
                    video_id=video_id,
                    title="Authorized production target",
                    channel_id=channel_id,
                    channel_title="Authorized target channel",
                    url=canonical_url,
                    duration_seconds=duration,
                )
            )
        if not self._videos:
            raise ValueError("at least one mounted source is required")

    @property
    def videos(self) -> list[Any]:
        return list(self._videos)

    def discover(self, brief: Any) -> list[Any]:
        # Transitional protocol method. Production pipeline resolves exact targets and
        # validates that this list matches them; no discovery occurs here.
        del brief
        return self.videos

    def download_subtitles(self, video: Any, work_dir: Path, language: str) -> None:
        del video, work_dir, language
        return None

    def download_media(self, video: Any, work_dir: Path) -> Path:
        del work_dir
        record = self._records.get(str(video.video_id))
        if record is None:
            raise RuntimeError(f"mounted source requested unknown video: {video.video_id}")
        source_path = Path(record["path"])
        expected = str(record["sha256"])
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise RuntimeError(
                f"mounted source master failed SHA-256 verification: {video.video_id}"
            )
        return source_path

    def download_media_span(
        self,
        video: Any,
        start: float,
        end: float,
        work_dir: Path,
    ) -> Any:
        from clipper.fixture import SpanMedia

        source_path = self.download_media(video, work_dir)
        record = self._records[str(video.video_id)]
        duration = float(record["duration"])
        if start < -1e-6 or end > duration + 1e-6:
            raise RuntimeError(
                f"requested render span is outside master {video.video_id}: {start:.3f}-{end:.3f}"
            )
        return SpanMedia(source_path, 0.0, duration, str(record["sha256"]))

    def evidence_by_video(self) -> dict[str, dict[str, Any]]:
        return {video_id: dict(record["evidence"]) for video_id, record in self._records.items()}


def _run_editorial_capacity_probe(
    worker: Any,
    *,
    task: str,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run one non-generating editorial probe inside the shared producer lifecycle."""

    from clipper.providers.modal import invoke_editorial_capacity_probe

    response = invoke_editorial_capacity_probe(
        worker.capacity_probe.remote,
        task=task,
        payload=raw_payload,
        execution_id=os.getenv("CLIPPER_EXECUTION_ID", ""),
        expected_git_sha=os.getenv("CLIPPER_ACCEPTANCE_SHA", ""),
    )
    probe = response["value"]
    return {str(key): value for key, value in probe.items()}


def _editorial_acceptance_probe(
    run_dir: Path,
    brief_yaml: str,
    video_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Measure the legacy raw evidence and verify token-aware repartition inside Modal."""

    from clipper.brief import load_brief
    from clipper.canonical import CanonicalTimeline
    from clipper.editorial_capacity import token_aware_repartition
    from clipper.multimodal_timeline import build_multimodal_timeline
    from clipper.source_hazards import SourceHazardClassifier, campaign_context
    from clipper.visual import VisualTimeline

    brief_path = Path(f"/tmp/clipper-editorial-probe-{uuid.uuid4().hex}.yaml")
    brief_path.write_text(brief_yaml, encoding="utf-8")
    try:
        brief = load_brief(brief_path)
    finally:
        brief_path.unlink(missing_ok=True)

    worker = modal.Cls.from_name(MODEL_APP, "EditorialModel")()
    results: list[dict[str, Any]] = []
    for video_id in video_ids:
        timeline = CanonicalTimeline.from_dict(
            json.loads((run_dir / "canonical" / f"{video_id}.json").read_text(encoding="utf-8"))
        )
        visual = VisualTimeline.from_dict(
            json.loads((run_dir / "visual-scout" / f"{video_id}.json").read_text(encoding="utf-8"))
        )
        multimodal = build_multimodal_timeline(timeline, visual)
        if not timeline.words:
            raise RuntimeError(f"editorial acceptance probe has no canonical words: {video_id}")

        source_start = timeline.words[0].source_start
        source_end = timeline.words[-1].source_end
        raw_payload: dict[str, Any] = {
            "campaign": campaign_context(brief),
            "instruction": (
                "Classify the entire supplied source interval into exhaustive chronological "
                "segments. Fuse speech and multimodal evidence. Ordinary source material is "
                "editorial_content. Use unknown when evidence is insufficient; uncertainty "
                "must never be converted into an automatic PASS."
            ),
            "words": SourceHazardClassifier._word_payload(
                timeline,
                0,
                len(timeline.words),
            ),
            "capacity_repartitionable": True,
            "multimodal_evidence": SourceHazardClassifier._legacy_multimodal_payload(
                multimodal,
                source_start,
                source_end,
            ),
        }
        task = f"source_hazards:acceptance_probe:{video_id}"
        details = _run_editorial_capacity_probe(
            worker,
            task=task,
            raw_payload=raw_payload,
        )
        repartition = token_aware_repartition(
            timeline,
            0,
            len(timeline.words),
            details,
        )
        if repartition is None:
            raise RuntimeError(
                "live raw-evidence capacity probe did not produce a token-aware repartition"
            )
        repartition_event = repartition.telemetry(stage=task)
        evidence = {
            "video_id": video_id,
            "capacity_probe": details,
            "repartition": repartition_event,
        }
        results.append(evidence)
        print(
            json.dumps(
                {
                    "event": "editorial_acceptance_probe_result",
                    "execution_id": os.getenv("CLIPPER_EXECUTION_ID", ""),
                    **evidence,
                },
                sort_keys=True,
            )
        )

    result = {"status": "PASS", "sources": results}
    (run_dir / "editorial-acceptance-probe.json").write_text(
        json.dumps(result, indent=2) + chr(10),
        encoding="utf-8",
    )
    artifact_volume.commit()
    return result


@app.function(image=runner_image, timeout=60, scaledown_window=2)
def deployment_identity() -> dict[str, Any]:
    return {"app": APP_NAME, "deployed_git_sha": DEPLOYED_GIT_SHA}


@app.function(
    image=runner_image,
    volumes={MEDIA_ROOT: media_cache, ARTIFACT_ROOT: artifact_volume},
    timeout=21600,
    memory=8192,
    scaledown_window=2,
)
def run_full_cycle(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one autonomous production DAG over all explicitly targeted source masters."""
    from clipper.pipeline import PipelineSettings, run_pipeline
    from clipper.providers.factory import speech_providers

    raw_sources = payload.get("sources")
    if (
        not isinstance(raw_sources, list)
        or not raw_sources
        or not all(isinstance(item, dict) for item in raw_sources)
    ):
        raise ValueError("run_full_cycle requires a non-empty sources array")
    source_items = [dict(item) for item in raw_sources]
    brief_yaml = str(payload.get("brief_yaml") or "")
    if not brief_yaml.strip():
        raise ValueError("run_full_cycle requires a campaign brief")

    render = bool(payload.get("render", True))
    editorial_acceptance_probe = bool(payload.get("editorial_acceptance_probe", False))
    fresh_inference = bool(payload.get("fresh_inference", False))
    resume_from_run_id = str(payload.get("resume_from_run_id") or "").strip() or None
    if fresh_inference and resume_from_run_id is not None:
        raise ValueError("fresh inference cannot be combined with a resume_from_run_id")
    execution_mode = "fresh-inference" if fresh_inference else "content-addressed-resume"
    execution_id = str(payload.get("execution_id") or "").strip()
    if len(execution_id) != 32 or any(
        character not in "0123456789abcdef" for character in execution_id.lower()
    ):
        raise ValueError("run_full_cycle execution_id must be a 32-character hexadecimal ID")

    requested_git_sha = str(payload.get("git_sha") or "").strip().lower()
    if len(requested_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in requested_git_sha
    ):
        raise ValueError("run_full_cycle git_sha must be a full hexadecimal commit SHA")
    if not DEPLOYED_GIT_SHA:
        raise RuntimeError("production pipeline worker has no embedded deployment SHA")
    if requested_git_sha != DEPLOYED_GIT_SHA:
        raise RuntimeError(
            "production pipeline worker SHA mismatch: "
            f"expected={requested_git_sha} deployed={DEPLOYED_GIT_SHA}"
        )

    max_gpu_seconds = float(payload.get("max_gpu_seconds") or 0.0)
    max_estimated_usd = float(payload.get("max_estimated_usd") or 0.0)
    if max_gpu_seconds <= 0 or max_estimated_usd <= 0:
        raise ValueError("run_full_cycle requires positive compute budget limits")

    os.environ.update(
        {
            "GITHUB_SHA": requested_git_sha,
            "CLIPPER_ACCEPTANCE_SHA": requested_git_sha,
            "CLIPPER_EXECUTION_ID": execution_id.lower(),
            "CLIPPER_MAX_GPU_SECONDS": str(max_gpu_seconds),
            "CLIPPER_MAX_ESTIMATED_USD": str(max_estimated_usd),
        }
    )

    media_cache.reload()
    artifact_volume.reload()
    source = VolumeSourceClient(source_items)
    brief_path = Path(f"/tmp/clipper-brief-{uuid.uuid4().hex}.yaml")
    brief_path.write_text(brief_yaml, encoding="utf-8")

    os.environ.update(
        {
            "CLIPPER_MODAL_APP": MODEL_APP,
            "CLIPPER_COMPUTE_PROFILE": "balanced",
            "CLIPPER_VISUAL_SCOUT": "true",
            "CLIPPER_VISUAL_REVIEW": "true",
            "CLIPPER_VISUAL_ESCALATION": "false",
            "CLIPPER_RENDER_PROFILE": "production",
            "CLIPPER_ARTIFACT_ROOT": ARTIFACT_ROOT,
            "CLIPPER_CACHE_ROOT": f"{ARTIFACT_ROOT}/_cache",
        }
    )
    settings = PipelineSettings.from_env()
    if fresh_inference:
        settings = replace(
            settings,
            cache_root=Path(ARTIFACT_ROOT) / "_fresh-cache" / uuid.uuid4().hex,
        )
    asr, alignment, diarization = speech_providers(settings.compute_profile)

    try:
        run_dir = run_pipeline(
            brief_path,
            settings=settings,
            source_client=source,
            transcription_provider=asr,
            alignment_provider=alignment,
            diarization_provider=diarization,
            render=render,
            checkpoint_commit=artifact_volume.commit,
        )
    finally:
        brief_path.unlink(missing_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("production pipeline returned an invalid manifest")

    run_relative = "/" + str(run_dir.relative_to(Path(ARTIFACT_ROOT))).replace(os.sep, "/")
    failed = (render and manifest.get("status") not in {"SUCCESS", "DEGRADED"}) or (
        not render and manifest.get("status") == "FAILED"
    )
    if failed:
        raw_errors = manifest.get("errors")
        errors = (
            [dict(item) for item in raw_errors if isinstance(item, dict)]
            if isinstance(raw_errors, list)
            else []
        )
        metadata = manifest.get("run_metadata")
        git_sha = metadata.get("git_sha") if isinstance(metadata, dict) else requested_git_sha
        artifact_volume.commit()
        print(
            json.dumps(
                {
                    "event": "production_cycle_terminal",
                    "execution_id": execution_id.lower(),
                    "status": "FAIL",
                    "pipeline_status": manifest.get("status"),
                    "review_status": "NOT_REVIEWABLE",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return {
            "status": "FAIL",
            "run_volume": "clipper-production-artifacts",
            "run_path": run_relative,
            "sources": source_items,
            "pipeline_status": manifest.get("status"),
            "status_reason": manifest.get("status_reason"),
            "errors": errors,
            "git_sha": git_sha,
            "rendered": len(manifest.get("rendered_clips") or []),
            "reviewable": len(manifest.get("submission_shortlist") or []),
            "review_status": "NOT_REVIEWABLE",
        }

    editorial_probe_result: dict[str, Any] | None = None
    if editorial_acceptance_probe:
        editorial_probe_result = _editorial_acceptance_probe(
            run_dir,
            brief_yaml,
            tuple(video.video_id for video in source.videos),
        )

    metadata = manifest.get("run_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("production manifest is missing run_metadata")
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("production manifest is missing source hashes")

    source_execution: list[dict[str, object]] = []
    for video in source.videos:
        evidence = source.evidence_by_video()[video.video_id]
        expected = str(evidence.get("sha256") or "")
        if str(source_hashes.get(video.video_id) or "") != expected:
            raise RuntimeError(
                f"pipeline did not process the Modal-acquired source hash: {video.video_id}"
            )
        source_execution.append(
            {
                "video_id": video.video_id,
                "mode": "modal-native",
                "source_volume": "clipper-media-cache",
                "source_mount_path": str(evidence.get("mount_path") or ""),
                "source_sha256": expected,
            }
        )
    metadata["source_execution"] = source_execution
    metadata["execution_mode"] = execution_mode
    if not fresh_inference:
        metadata["resume"] = {
            "from_run_id": resume_from_run_id,
            "mode": "content-addressed-stage-resume",
            "cache_root": str(settings.cache_root),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    artifact_volume.commit()
    terminal_review_status = "PENDING_ACTUAL_MP4_REVIEW" if render else "NOT_RENDERED"
    print(
        json.dumps(
            {
                "event": "production_cycle_terminal",
                "execution_id": execution_id.lower(),
                "status": "PASS",
                "pipeline_status": manifest.get("status"),
                "review_status": terminal_review_status,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "status": "PASS",
        "execution_mode": execution_mode,
        "execution_id": execution_id.lower(),
        "deployed_git_sha": DEPLOYED_GIT_SHA,
        "run_volume": "clipper-production-artifacts",
        "run_path": run_relative,
        "sources": source_items,
        "pipeline_status": manifest.get("status"),
        "eligible_quality_moments": int(
            (metadata.get("quality_yield") or {}).get("eligible_quality_moments", 0)
            if isinstance(metadata.get("quality_yield"), dict)
            else 0
        ),
        "rendered": len(manifest.get("rendered_clips") or []),
        "reviewable": len(manifest.get("submission_shortlist") or []),
        "review_status": "PENDING_ACTUAL_MP4_REVIEW" if render else "NOT_RENDERED",
        "editorial_acceptance_probe": editorial_probe_result,
    }
