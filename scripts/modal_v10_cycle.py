from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.getenv("CLIPPER_V10_MODAL_APP", "clipper-v10-cycle")
MODEL_APP = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
MEDIA_ROOT = "/media"
ARTIFACT_ROOT = "/artifacts"

app = modal.App(APP_NAME)
media_cache = modal.Volume.from_name("clipper-media-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("clipper-v10-artifacts", create_if_missing=True)

if modal.is_local():
    youtube_secret = modal.Secret.from_dict(
        {"CLIPPER_YOUTUBE_COOKIES_B64": os.environ.get("CLIPPER_YOUTUBE_COOKIES_B64", "")}
    )
else:
    youtube_secret = modal.Secret.from_dict({})

media_image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
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
    duration = float(fmt.get("duration") or 0.0)
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
        "duration_seconds": duration,
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
    if evidence["sha256"] != expected:
        return None
    return evidence


@app.function(
    image=media_image,
    volumes={MEDIA_ROOT: media_cache},
    secrets=[youtube_secret],
    timeout=7200,
    memory=4096,
    scaledown_window=2,
)
def acquire_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Acquire the highest available authorized source directly inside Modal."""

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
            "yt-dlp source acquisition exhausted all public bgutil strategies"
            + (" and the optional authenticated fallback" if cookies_available else "")
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
    def __init__(self, evidence: dict[str, Any], channel_id: str) -> None:
        from clipper.models import VideoCandidate

        self.source_path = Path(str(evidence["mount_path"]))
        self.source_sha256 = str(evidence["sha256"])
        self.duration = float(evidence["duration_seconds"])
        self.video = VideoCandidate(
            video_id=str(evidence["video_id"]),
            title="Authorized production source",
            channel_id=channel_id,
            channel_title="Authorized source channel",
            url=str(evidence["source_url"]),
            duration_seconds=self.duration,
        )

    def discover(self, brief: Any) -> list[Any]:
        del brief
        return [self.video]

    def download_subtitles(self, video: Any, work_dir: Path, language: str) -> None:
        del video, work_dir, language
        return None

    def download_media(self, video: Any, work_dir: Path) -> Path:
        del work_dir
        if video.video_id != self.video.video_id:
            raise RuntimeError("volume source requested the wrong video")
        if not self.source_path.is_file() or _sha256(self.source_path) != self.source_sha256:
            raise RuntimeError("mounted source master failed SHA-256 verification")
        return self.source_path

    def download_media_span(
        self,
        video: Any,
        start: float,
        end: float,
        work_dir: Path,
    ) -> Any:
        from clipper.fixture import SpanMedia

        del work_dir
        self.download_media(video, Path("."))
        if start < -1e-6 or end > self.duration + 1e-6:
            raise RuntimeError(f"requested render span is outside master: {start:.3f}-{end:.3f}")
        return SpanMedia(self.source_path, 0.0, self.duration, self.source_sha256)


_TARGETED_RECOVERY_PLANS = (("c14", "p3"), ("c5", "p1"))
_RECOVERED_FINALISTS = (("c3", "p3"), ("c6", "p1"), ("c11", "p1"), ("c2", "p4"))
_APPROVED_REPLAY_RESULTS = (*_RECOVERED_FINALISTS, ("c11", "p2"))
_CLIP_SIDECAR_SUFFIXES = (
    ".ass",
    ".caption-audit.json",
    ".render.json",
    ".tracking-preflight.json",
    ".tracking.json",
)


def _direct_volume_child(root: str, child_name: str, *, label: str) -> Path:
    """Resolve one direct Volume child without permitting path traversal."""

    if not child_name or Path(child_name).name != child_name or child_name in {".", ".."}:
        raise ValueError(f"invalid {label}: {child_name!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child_name).resolve()
    if resolved.parent != resolved_root:
        raise ValueError(f"{label} must be a direct child of {root}")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _replace_path_prefix(value: Any, source_prefix: str, destination_prefix: str) -> Any:
    if isinstance(value, str):
        if value == source_prefix:
            return destination_prefix
        if value.startswith(source_prefix + "/"):
            return destination_prefix + value[len(source_prefix) :]
        return value
    if isinstance(value, list):
        return [_replace_path_prefix(item, source_prefix, destination_prefix) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, source_prefix, destination_prefix)
            for key, item in value.items()
        }
    return value


def _edit_plan_from_payload(payload: dict[str, Any]) -> Any:
    from clipper.models import EditorialBeat, EditPlan, SourceSpan

    raw_spans = payload.get("source_spans")
    if not isinstance(raw_spans, list) or len(raw_spans) != 1:
        raise RuntimeError("targeted recovery requires one contiguous source span")
    raw_beats = payload.get("beats", [])
    if not isinstance(raw_beats, list):
        raise RuntimeError("edit plan beats must be a list")
    return EditPlan(
        plan_id=str(payload["plan_id"]),
        video_id=str(payload["video_id"]),
        concept_id=str(payload["concept_id"]),
        variant_id=str(payload["variant_id"]),
        hook_mode=str(payload["hook_mode"]),  # type: ignore[arg-type]
        source_spans=tuple(
            SourceSpan(start=float(item["start"]), end=float(item["end"]))
            for item in raw_spans
            if isinstance(item, dict)
        ),
        hook_text=str(payload["hook_text"]) if payload.get("hook_text") is not None else None,
        beats=tuple(
            EditorialBeat(
                start=float(item["start"]),
                end=float(item["end"]),
                beat_type=str(item["beat_type"]),  # type: ignore[arg-type]
                strength=float(item.get("strength") or 0.0),
                text=str(item["text"]) if item.get("text") is not None else None,
            )
            for item in raw_beats
            if isinstance(item, dict)
        ),
        caption_platform=str(payload["caption_platform"]),
        score=float(payload["score"]),
        transcript_fingerprint=str(payload["transcript_fingerprint"]),
        caption_start_source_time=(
            float(payload["caption_start_source_time"])
            if payload.get("caption_start_source_time") is not None
            else None
        ),
        caption_start_word=(
            str(payload["caption_start_word"])
            if payload.get("caption_start_word") is not None
            else None
        ),
    )


def _transcript_segments(payload: object) -> list[Any]:
    from clipper.models import TranscriptSegment, TranscriptWord

    if not isinstance(payload, list):
        raise RuntimeError("source transcript must be a list")
    segments: list[Any] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("source transcript contains a non-object segment")
        raw_words = item.get("words", [])
        if not isinstance(raw_words, list):
            raise RuntimeError("source transcript words must be a list")
        segments.append(
            TranscriptSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                words=tuple(
                    TranscriptWord(
                        start=float(word["start"]),
                        end=float(word["end"]),
                        text=str(word["text"]),
                    )
                    for word in raw_words
                    if isinstance(word, dict)
                ),
            )
        )
    return segments


def _copy_clip_evidence(source_clip: Path, destination_clip: Path, run_dir: Path) -> None:
    destination_clip.parent.mkdir(parents=True, exist_ok=True)
    if source_clip.resolve() != destination_clip.resolve():
        shutil.copy2(source_clip, destination_clip)
    for suffix in _CLIP_SIDECAR_SUFFIXES:
        source = source_clip.with_suffix(suffix)
        if not source.is_file():
            raise RuntimeError(f"required render evidence is missing: {source}")
        destination = destination_clip.with_suffix(suffix)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        if suffix in {".ass", ".caption-audit.json"}:
            evidence_dir = run_dir / "captions"
        elif suffix in {".tracking.json", ".tracking-preflight.json"}:
            evidence_dir = run_dir / "tracking"
        else:
            continue
        evidence_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, evidence_dir / destination.name)


def _rendered_clip_payload(
    *, plan: Any, output_path: Path, source_url: str, source_hash: str
) -> dict[str, Any]:
    from clipper.models import RenderedClip

    clip = plan.to_clip_candidate("")
    return RenderedClip(
        video_id=plan.video_id,
        output_path=str(output_path),
        start=clip.start,
        end=clip.end,
        score=plan.score,
        source_url=source_url,
        concept_id=plan.concept_id,
        plan_id=plan.plan_id,
        hook_mode=plan.hook_mode,
        render_sha256=source_hash,
    ).to_dict()


@app.function(
    image=runner_image,
    volumes={MEDIA_ROOT: media_cache, ARTIFACT_ROOT: artifact_volume},
    timeout=7200,
    memory=8192,
    scaledown_window=2,
)
def recover_finalists(payload: dict[str, Any]) -> dict[str, Any]:
    """Render and review only the two explicitly approved replacement finalists."""

    from clipper.pipeline import PipelineSettings
    from clipper.providers.factory import vision_provider
    from clipper.qc import run_technical_qc
    from clipper.render import FFmpegRenderer
    from clipper.visual_ai import review_rendered_clip, tracking_transition_sample_times

    source_run_id = str(payload.get("source_run_id") or "")
    recovery_id = str(payload.get("recovery_id") or "")
    requested_raw = payload.get("plan_keys")
    if not isinstance(requested_raw, list):
        raise ValueError("targeted recovery requires plan_keys")
    requested = tuple(
        (str(item.get("concept_id") or ""), str(item.get("plan_id") or ""))
        for item in requested_raw
        if isinstance(item, dict)
    )
    if requested != _TARGETED_RECOVERY_PLANS:
        raise ValueError("targeted recovery is restricted to c14/p3 and c5/p1 in that order")
    reuse_raw = payload.get("reuse_plan_keys", [])
    if not isinstance(reuse_raw, list):
        raise ValueError("targeted recovery reuse_plan_keys must be a list")
    reused_targeted_keys = tuple(
        (str(item.get("concept_id") or ""), str(item.get("plan_id") or ""))
        for item in reuse_raw
        if isinstance(item, dict)
    )
    if reused_targeted_keys not in ((), (("c14", "p3"),)):
        raise ValueError("targeted recovery may reuse only the already-passed c14/p3 canary")
    base_run_id = str(payload.get("base_run_id") or "")
    if bool(reused_targeted_keys) != bool(base_run_id):
        raise ValueError("targeted recovery reuse requires a base_run_id and reuse_plan_keys")
    freshly_rendered_keys = tuple(
        key for key in _TARGETED_RECOVERY_PLANS if key not in reused_targeted_keys
    )

    prior_recovery = payload.get("prior_review_recovery")
    if not isinstance(prior_recovery, dict):
        raise ValueError("targeted recovery requires the approved prior review evidence")
    prior_results_raw = prior_recovery.get("results")
    if not isinstance(prior_results_raw, list):
        raise ValueError("prior review evidence has no results")
    prior_results: dict[tuple[str, str], dict[str, Any]] = {}
    for item in prior_results_raw:
        if not isinstance(item, dict):
            raise ValueError("prior review result must be an object")
        key = (str(item.get("concept_id") or ""), str(item.get("plan_id") or ""))
        report = item.get("report")
        if (
            key not in _APPROVED_REPLAY_RESULTS
            or not isinstance(report, dict)
            or report.get("decision") != "PASS"
            or report.get("issues") not in ([], ())
        ):
            raise RuntimeError(f"prior visual review is not an approved PASS: {key}")
        if key in prior_results:
            raise RuntimeError(f"duplicate prior visual review result: {key}")
        prior_results[key] = item
    if set(prior_results) != set(_APPROVED_REPLAY_RESULTS):
        raise RuntimeError(
            "prior review evidence does not contain exactly the five approved replays"
        )

    media_cache.reload()
    artifact_volume.reload()
    source_run_dir = _direct_volume_child(ARTIFACT_ROOT, source_run_id, label="source run ID")
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    recovery_suffix = _direct_volume_child("/tmp", recovery_id, label="recovery ID").name
    output_run_id = f"{source_run_id}-targeted-{recovery_suffix}"
    output_run_dir = _direct_volume_child(ARTIFACT_ROOT, output_run_id, label="output run ID")
    partial_run_dir = _direct_volume_child(
        ARTIFACT_ROOT, f".{output_run_id}.partial", label="partial output run ID"
    )
    if output_run_dir.exists() or partial_run_dir.exists():
        raise RuntimeError(f"refusing to overwrite targeted recovery: {output_run_id}")

    original_manifest = _load_json_object(source_run_dir / "manifest.json")
    if original_manifest.get("status") != "FAILED":
        raise RuntimeError("targeted recovery requires the expected failed source manifest")
    edit_plan_payloads = original_manifest.get("edit_plans")
    concept_payloads = original_manifest.get("clip_concepts")
    discovered_videos = original_manifest.get("discovered_videos")
    if not isinstance(edit_plan_payloads, list) or not isinstance(concept_payloads, list):
        raise RuntimeError("source manifest is missing edit plans or concepts")
    if not isinstance(discovered_videos, list):
        raise RuntimeError("source manifest is missing discovered videos")
    base_run_dir: Path | None = None
    base_rendered_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if base_run_id:
        base_run_dir = _direct_volume_child(ARTIFACT_ROOT, base_run_id, label="base run ID")
        base_manifest = _load_json_object(base_run_dir / "manifest.json")
        if base_manifest.get("status") != "SUCCESS":
            raise RuntimeError("targeted reuse requires a successful base run")
        base_rendered = base_manifest.get("rendered_clips")
        if not isinstance(base_rendered, list):
            raise RuntimeError("targeted reuse base run has no rendered clips")
        base_rendered_by_key = {
            (str(item.get("concept_id") or ""), str(item.get("plan_id") or "")): item
            for item in base_rendered
            if isinstance(item, dict)
        }
        if any(key not in base_rendered_by_key for key in reused_targeted_keys):
            raise RuntimeError("targeted reuse base run is missing an approved plan")
    plan_by_key = {
        (str(item.get("concept_id") or ""), str(item.get("plan_id") or "")): item
        for item in edit_plan_payloads
        if isinstance(item, dict)
    }
    concept_by_id = {
        str(item.get("concept_id") or ""): item
        for item in concept_payloads
        if isinstance(item, dict)
    }
    all_final_keys = (*_RECOVERED_FINALISTS, *_TARGETED_RECOVERY_PLANS)
    if any(key not in plan_by_key for key in all_final_keys):
        raise RuntimeError("source manifest is missing a selected edit plan")
    if any(concept_id not in concept_by_id for concept_id, _ in all_final_keys):
        raise RuntimeError("source manifest is missing a selected concept")

    selected_video_ids = {str(plan_by_key[key].get("video_id") or "") for key in all_final_keys}
    if len(selected_video_ids) != 1:
        raise RuntimeError("targeted recovery requires exactly one source video")
    video_id = selected_video_ids.pop()
    source_index = _load_json_object(Path(MEDIA_ROOT) / "source-index" / f"{video_id}.json")
    source_path = Path(str(source_index.get("mount_path") or ""))
    source_sha256 = str(source_index.get("sha256") or "")
    if (
        not source_path.is_file()
        or not source_path.as_posix().startswith(f"{MEDIA_ROOT}/inputs/")
        or len(source_sha256) != 64
        or _sha256(source_path) != source_sha256
    ):
        raise RuntimeError("mounted source master failed targeted recovery verification")
    source_url = str(source_index.get("source_url") or "")
    if not source_url:
        for item in discovered_videos:
            if isinstance(item, dict) and str(item.get("video_id") or "") == video_id:
                source_url = str(item.get("url") or "")
                break
    if not source_url:
        raise RuntimeError("source URL evidence is missing")

    transcript_payload = _load_json_object(source_run_dir / "transcript.json")
    segments = _transcript_segments(transcript_payload.get(video_id))
    watermark_path = source_run_dir / "assets" / "watermark.png"
    if not watermark_path.is_file():
        raise RuntimeError("required campaign watermark is missing")

    os.environ.update(
        {
            "CLIPPER_MODAL_APP": MODEL_APP,
            "CLIPPER_COMPUTE_PROFILE": "balanced",
            "CLIPPER_VISUAL_REVIEW": "true",
            "CLIPPER_VISUAL_ESCALATION": "false",
            "CLIPPER_RENDER_PROFILE": "production",
        }
    )
    settings = PipelineSettings.from_env()
    renderer = FFmpegRenderer(
        speaker_focus=settings.speaker_focus,
        zoom_factor=settings.speaker_zoom,
        speaker_sample_fps=settings.speaker_sample_fps,
        speaker_switch_margin=settings.speaker_switch_margin,
        speaker_min_reframe_seconds=settings.speaker_min_reframe_seconds,
        speaker_max_reframe_seconds=settings.speaker_max_reframe_seconds,
        speaker_seconds_per_crop=settings.speaker_seconds_per_crop,
        speaker_hold_threshold=settings.speaker_hold_threshold,
        speaker_reversal_guard_seconds=settings.speaker_reversal_guard_seconds,
        speaker_window_seconds=settings.speaker_window_seconds,
        speaker_min_detection_coverage=settings.speaker_min_detection_coverage,
        profile=settings.render_profile,
    )
    reviewer = vision_provider("balanced")

    try:
        partial_run_dir.mkdir(parents=True, exist_ok=False)
        for directory_name in ("assets", "canonical", "edit-plans"):
            source_directory = source_run_dir / directory_name
            if source_directory.is_dir():
                shutil.copytree(
                    source_directory,
                    partial_run_dir / directory_name,
                    dirs_exist_ok=True,
                )
        for filename in (
            "brief.normalized.json",
            "concept-ranking.json",
            "coverage.json",
            "hook-variants.json",
            "story-moments.json",
            "transcript.json",
        ):
            source_file = source_run_dir / filename
            if source_file.is_file():
                shutil.copy2(source_file, partial_run_dir / filename)

        rendered_clips: list[dict[str, Any]] = []
        technical_qc: list[dict[str, Any]] = []
        editorial_qc: list[dict[str, Any]] = []
        render_attempts: list[dict[str, Any]] = []
        new_review_payloads: list[dict[str, Any]] = []
        new_rendered_clips: list[dict[str, Any]] = []

        for attempt_number, key in enumerate(_RECOVERED_FINALISTS, start=1):
            result = prior_results[key]
            clip_name = str(result.get("clip") or "")
            if Path(clip_name).name != clip_name or not clip_name.endswith(".mp4"):
                raise RuntimeError(f"invalid prior clip name: {clip_name!r}")
            source_clip = source_run_dir / "clips" / clip_name
            if not source_clip.is_file():
                raise FileNotFoundError(source_clip)
            destination_clip = partial_run_dir / "clips" / clip_name
            _copy_clip_evidence(source_clip, destination_clip, partial_run_dir)
            qc_source = source_run_dir / "qc" / f"{source_clip.stem}.json"
            qc_payload = _load_json_object(qc_source)
            if qc_payload.get("status") != "PASS" or qc_payload.get("issues") not in ([], ()):
                raise RuntimeError(f"prior technical QC is not PASS: {key}")
            qc_payload["concept_id"] = key[0]
            qc_payload["plan_id"] = key[1]
            qc_payload["recovered"] = True
            video_payload = qc_payload.get("video")
            if isinstance(video_payload, dict):
                video_payload["path"] = str(destination_clip)
            captions_payload = qc_payload.get("captions")
            if isinstance(captions_payload, dict):
                captions_payload["audit_path"] = str(
                    destination_clip.with_suffix(".caption-audit.json")
                )
            (partial_run_dir / "qc").mkdir(parents=True, exist_ok=True)
            (partial_run_dir / "qc" / f"{destination_clip.stem}.json").write_text(
                json.dumps(qc_payload, indent=2) + "\n", encoding="utf-8"
            )
            technical_qc.append(qc_payload)
            report = result["report"]
            review_payload = {
                **report,
                "concept_id": key[0],
                "plan_id": key[1],
                "models": [result["model"]],
                "usage": [result["usage"]],
                "recovered": True,
                "recovery_source": "approved_visual_review_replay",
            }
            editorial_qc.append(review_payload)
            (partial_run_dir / "visual-review").mkdir(parents=True, exist_ok=True)
            (partial_run_dir / "visual-review" / f"{destination_clip.stem}.json").write_text(
                json.dumps(review_payload, indent=2) + "\n", encoding="utf-8"
            )
            source_frames = source_run_dir / "visual-review" / source_clip.stem / "frames"
            if not source_frames.is_dir():
                raise RuntimeError(f"prior visual review frame set is missing: {source_frames}")
            shutil.copytree(
                source_frames,
                partial_run_dir / "visual-review" / destination_clip.stem / "frames",
            )
            plan = _edit_plan_from_payload(plan_by_key[key])
            rendered_clips.append(
                _rendered_clip_payload(
                    plan=plan,
                    output_path=destination_clip,
                    source_url=source_url,
                    source_hash=_sha256(destination_clip),
                )
            )
            render_attempts.append(
                {
                    "attempt": attempt_number,
                    "concept_id": key[0],
                    "plan_id": key[1],
                    "queue": "approved-recovery",
                    "status": "ACCEPTED",
                    "technical_qc": "PASS",
                    "editorial_qc": review_payload,
                }
            )

        for offset, key in enumerate(_TARGETED_RECOVERY_PLANS, start=1):
            plan = _edit_plan_from_payload(plan_by_key[key])
            concept = concept_by_id[key[0]]
            clip = plan.to_clip_candidate(str(concept.get("text") or ""))
            attempt_number = len(_RECOVERED_FINALISTS) + offset
            if key in reused_targeted_keys:
                if base_run_dir is None:
                    raise RuntimeError("targeted reuse base run is unavailable")
                base_rendered = base_rendered_by_key[key]
                clip_name = Path(str(base_rendered.get("output_path") or "")).name
                if not clip_name.endswith(".mp4"):
                    raise RuntimeError(f"targeted reuse clip path is invalid for {key}")
                source_clip = base_run_dir / "clips" / clip_name
                destination_clip = partial_run_dir / "clips" / clip_name
                _copy_clip_evidence(source_clip, destination_clip, partial_run_dir)
                qc_payload = _load_json_object(base_run_dir / "qc" / f"{source_clip.stem}.json")
                review_payload = _load_json_object(
                    base_run_dir / "visual-review" / f"{source_clip.stem}.json"
                )
                if qc_payload.get("status") != "PASS" or qc_payload.get("issues") not in ([], ()):
                    raise RuntimeError(f"targeted reuse technical QC is not PASS: {key}")
                if review_payload.get("decision") != "PASS" or review_payload.get("issues") not in (
                    [],
                    (),
                ):
                    raise RuntimeError(f"targeted reuse visual review is not PASS: {key}")
                qc_payload["reused_from_run_id"] = base_run_id
                review_payload["reused_from_run_id"] = base_run_id
                (partial_run_dir / "qc" / f"{destination_clip.stem}.json").write_text(
                    json.dumps(qc_payload, indent=2) + "\n", encoding="utf-8"
                )
                (partial_run_dir / "visual-review").mkdir(parents=True, exist_ok=True)
                (partial_run_dir / "visual-review" / f"{destination_clip.stem}.json").write_text(
                    json.dumps(review_payload, indent=2) + "\n", encoding="utf-8"
                )
                shutil.copytree(
                    base_run_dir / "visual-review" / source_clip.stem / "frames",
                    partial_run_dir / "visual-review" / destination_clip.stem / "frames",
                )
                rendered_clips.append(
                    _rendered_clip_payload(
                        plan=plan,
                        output_path=destination_clip,
                        source_url=source_url,
                        source_hash=_sha256(destination_clip),
                    )
                )
                technical_qc.append(qc_payload)
                editorial_qc.append(review_payload)
                render_attempts.append(
                    {
                        "attempt": attempt_number,
                        "concept_id": key[0],
                        "plan_id": key[1],
                        "queue": "passed-targeted-reuse",
                        "status": "ACCEPTED",
                        "technical_qc": "PASS",
                        "editorial_qc": review_payload,
                    }
                )
                continue
            output_path = (
                partial_run_dir
                / "clips"
                / f"attempt-{attempt_number:02d}-{key[0]}-{key[1]}-{plan.hook_mode}.mp4"
            )
            rendered_path = renderer.render(
                source_path,
                output_path,
                clip,
                segments,
                watermark_path,
                plan,
            )
            qc_payload = run_technical_qc(
                rendered_path,
                expected_duration=clip.duration,
                caption_path=rendered_path.with_suffix(".ass"),
                tracking_path=rendered_path.with_suffix(".tracking.json"),
                caption_platform=plan.caption_platform,
                watermark_required=True,
                watermark_present=True,
                caption_audit_path=rendered_path.with_suffix(".caption-audit.json"),
            )
            qc_payload["concept_id"] = key[0]
            qc_payload["plan_id"] = key[1]
            qc_payload["recovered"] = True
            if qc_payload.get("status") != "PASS":
                raise RuntimeError(
                    f"targeted technical QC failed for {key}: {qc_payload.get('issues')}"
                )
            (partial_run_dir / "qc").mkdir(parents=True, exist_ok=True)
            (partial_run_dir / "qc" / f"{rendered_path.stem}.json").write_text(
                json.dumps(qc_payload, indent=2) + "\n", encoding="utf-8"
            )
            technical_qc.append(qc_payload)
            _copy_clip_evidence(rendered_path, rendered_path, partial_run_dir)
            tracking_payload = json.loads(
                rendered_path.with_suffix(".tracking.json").read_text(encoding="utf-8")
            )
            if not isinstance(tracking_payload, dict):
                raise RuntimeError(f"tracking evidence is malformed for {key}")
            review, review_results = review_rendered_clip(
                rendered_path,
                reviewer,
                duration=clip.duration,
                output_dir=partial_run_dir / "visual-review" / rendered_path.stem / "frames",
                context={
                    "plan_id": plan.plan_id,
                    "concept_id": plan.concept_id,
                    "source_start": clip.start,
                    "source_end": clip.end,
                    "hook_mode": plan.hook_mode,
                    "technical_qc": qc_payload,
                    "review_scope": "user-approved-targeted-recovery",
                },
                transitions=tracking_transition_sample_times(
                    tracking_payload.get("transitions", [])
                ),
                escalation=None,
            )
            review_payload = review.to_dict()
            review_payload.update(
                {
                    "concept_id": key[0],
                    "plan_id": key[1],
                    "models": [item.model.to_dict() for item in review_results],
                    "usage": [asdict(item.usage) for item in review_results],
                    "recovered": True,
                    "recovery_source": "approved_targeted_render_review",
                }
            )
            if review.decision != "PASS" or review.issues:
                raise RuntimeError(
                    f"targeted visual review did not pass for {key}: {review_payload}"
                )
            editorial_qc.append(review_payload)
            new_review_payloads.append(review_payload)
            (partial_run_dir / "visual-review" / f"{rendered_path.stem}.json").write_text(
                json.dumps(review_payload, indent=2) + "\n", encoding="utf-8"
            )
            rendered_clips.append(
                _rendered_clip_payload(
                    plan=plan,
                    output_path=rendered_path,
                    source_url=source_url,
                    source_hash=_sha256(rendered_path),
                )
            )
            new_rendered_clips.append(rendered_clips[-1])
            render_attempts.append(
                {
                    "attempt": attempt_number,
                    "concept_id": key[0],
                    "plan_id": key[1],
                    "queue": "user-approved-targeted-recovery",
                    "status": "ACCEPTED",
                    "technical_qc": "PASS",
                    "editorial_qc": review_payload,
                }
            )

        physical_prefix = partial_run_dir.as_posix()
        stable_prefix = f"{ARTIFACT_ROOT}/{output_run_id}"
        rendered_clips = _replace_path_prefix(rendered_clips, physical_prefix, stable_prefix)
        technical_qc = _replace_path_prefix(technical_qc, physical_prefix, stable_prefix)
        for qc_payload, rendered in zip(technical_qc, rendered_clips, strict=True):
            qc_path = partial_run_dir / "qc" / f"{Path(rendered['output_path']).stem}.json"
            qc_path.write_text(json.dumps(qc_payload, indent=2) + "\n", encoding="utf-8")

        distinct_concepts = {str(item["concept_id"]) for item in rendered_clips}
        if len(rendered_clips) != 6 or len(distinct_concepts) != 6:
            raise RuntimeError("targeted recovery did not produce six distinct finalists")
        final_manifest = copy.deepcopy(original_manifest)
        now = datetime.now(UTC).isoformat()
        final_manifest.update(
            {
                "created_at": now,
                "status": "SUCCESS",
                "status_reason": None,
                "actual": {
                    "rendered_finalists": 6,
                    "submission_shortlist": 6,
                    "distinct_finalist_concepts": 6,
                    "distinct_shortlist_concepts": 6,
                },
                "render_attempts": render_attempts,
                "rendered_clips": rendered_clips,
                "submission_shortlist": list(rendered_clips),
                "technical_qc": technical_qc,
                "editorial_qc": editorial_qc,
                "errors": [],
            }
        )
        funnel = final_manifest.get("funnel")
        if not isinstance(funnel, dict):
            funnel = {}
            final_manifest["funnel"] = funnel
        funnel.update(
            {
                "render_plans": 6,
                "render_attempts": 6,
                "replacement_attempts": 2,
                "render_success": 6,
                "render_failures": 0,
                "technical_qc_pass": 6,
                "technical_qc_fail": 0,
                "editorial_qc_pass": 6,
                "editorial_qc_fail": 0,
                "visual_review_escalations": 0,
                "tracking_preflight_pass": 6,
                "tracking_preflight_fail": 0,
                "submission_shortlist": 6,
                "distinct_finalist_concepts": 6,
                "distinct_shortlist_concepts": 6,
            }
        )
        metadata = final_manifest.get("run_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            final_manifest["run_metadata"] = metadata
        new_review_usage = [
            usage
            for item in new_review_payloads
            for usage in item.get("usage", [])
            if isinstance(usage, dict)
        ]
        metadata["targeted_recovery"] = {
            "source_run_id": source_run_id,
            "source_manifest_status": original_manifest.get("status"),
            "source_manifest_status_reason": original_manifest.get("status_reason"),
            "completed_at": now,
            "approved_plan_keys": [
                {"concept_id": concept_id, "plan_id": plan_id}
                for concept_id, plan_id in _TARGETED_RECOVERY_PLANS
            ],
            "reused_passed_plan_keys": [
                {"concept_id": concept_id, "plan_id": plan_id}
                for concept_id, plan_id in _RECOVERED_FINALISTS
            ]
            + [
                {"concept_id": concept_id, "plan_id": plan_id}
                for concept_id, plan_id in reused_targeted_keys
            ],
            "excluded_duplicate_plan_key": {"concept_id": "c11", "plan_id": "p2"},
            "approved_data_boundary": (
                "source master remained in clipper-media-cache; only freshly rendered "
                "derived visual-review frame sets were sent to clipper-open-editor"
            ),
            "source_sha256": source_sha256,
            "vision_frame_set_counts": {
                item["concept_id"]: len(
                    list(
                        (
                            partial_run_dir
                            / "visual-review"
                            / Path(item["output_path"]).stem
                            / "frames"
                        ).glob("*.jpg")
                    )
                )
                for item in new_rendered_clips
            },
            "new_visual_review_estimated_cost_usd": sum(
                float(item.get("estimated_cost_usd") or 0.0) for item in new_review_usage
            ),
            "new_visual_review_gpu_seconds": sum(
                float(item.get("gpu_seconds") or 0.0) for item in new_review_usage
            ),
        }
        final_manifest["rejections"] = [
            item
            for item in original_manifest.get("rejections", [])
            if isinstance(item, dict) and str(item.get("stage") or "") != "render"
        ]
        review_manifest = {
            "status": "PENDING_HUMAN_REVIEW",
            "required": True,
            "clips": [
                {
                    "output_path": item["output_path"],
                    "plan_id": item["plan_id"],
                    "concept_id": item["concept_id"],
                    "technical_qc": "PASS",
                    "visual_qc": "PASS",
                    "human_review": "PENDING",
                }
                for item in rendered_clips
            ],
        }
        recovery_manifest = {
            "status": "PASS",
            "source_run_id": source_run_id,
            "output_run_id": output_run_id,
            "rendered_plan_keys": [
                {"concept_id": concept_id, "plan_id": plan_id}
                for concept_id, plan_id in freshly_rendered_keys
            ],
            "reused_targeted_plan_keys": [
                {"concept_id": concept_id, "plan_id": plan_id}
                for concept_id, plan_id in reused_targeted_keys
            ],
            "finalist_plan_keys": [
                {"concept_id": item["concept_id"], "plan_id": item["plan_id"]}
                for item in rendered_clips
            ],
            "manifest_status": "SUCCESS",
        }
        (partial_run_dir / "manifest.json").write_text(
            json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (partial_run_dir / "funnel.json").write_text(
            json.dumps(funnel, indent=2) + "\n", encoding="utf-8"
        )
        (partial_run_dir / "rejections.json").write_text(
            json.dumps(final_manifest["rejections"], indent=2) + "\n", encoding="utf-8"
        )
        (partial_run_dir / "editorial-review.json").write_text(
            json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (partial_run_dir / "targeted-recovery.json").write_text(
            json.dumps(recovery_manifest, indent=2) + "\n", encoding="utf-8"
        )
        partial_run_dir.replace(output_run_dir)
        artifact_volume.commit()
    except Exception:
        if partial_run_dir.exists():
            shutil.rmtree(partial_run_dir)
        raise

    run_relative = "/" + output_run_id
    return {
        "status": "PASS",
        "run_volume": "clipper-v10-artifacts",
        "run_path": run_relative,
        "source_run_id": source_run_id,
        "rendered_plan_keys": [
            {"concept_id": concept_id, "plan_id": plan_id}
            for concept_id, plan_id in freshly_rendered_keys
        ],
        "reused_targeted_plan_keys": [
            {"concept_id": concept_id, "plan_id": plan_id}
            for concept_id, plan_id in reused_targeted_keys
        ],
        "rendered_finalists": 6,
        "submission_shortlist": 6,
        "distinct_finalist_concepts": 6,
        "review_status": "PENDING_ACTUAL_MP4_REVIEW",
    }


@app.function(
    image=runner_image,
    volumes={MEDIA_ROOT: media_cache, ARTIFACT_ROOT: artifact_volume},
    timeout=21600,
    memory=8192,
    scaledown_window=2,
)
def run_full_cycle(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the canonical clipper pipeline inside Modal against the mounted source master."""

    from clipper.pipeline import PipelineSettings, run_pipeline
    from clipper.providers.factory import speech_providers

    source_evidence = payload.get("source")
    if not isinstance(source_evidence, dict):
        raise ValueError("run_full_cycle requires source evidence")
    brief_yaml = str(payload.get("brief_yaml") or "")
    channel_id = str(payload.get("channel_id") or "")
    if not brief_yaml.strip() or not channel_id:
        raise ValueError("run_full_cycle requires campaign brief and channel ID")
    if source_evidence.get("quality_policy") != "highest_available_no_transcode":
        raise RuntimeError("source master violates no-downgrade policy")

    render = bool(payload.get("render", True))
    fresh_inference = bool(payload.get("fresh_inference", False))
    resume_from_run_id = str(payload.get("resume_from_run_id") or "").strip() or None

    media_cache.reload()
    artifact_volume.reload()
    source = VolumeSourceClient(source_evidence, channel_id)
    brief_path = Path("/tmp/clipper-v10-brief.yaml")
    brief_path.write_text(brief_yaml, encoding="utf-8")

    os.environ.update(
        {
            "CLIPPER_MODAL_APP": MODEL_APP,
            "CLIPPER_EDITORIAL_ENGINE": "open",
            "CLIPPER_GROUNDING_ENGINE": "open",
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
    asr, alignment, diarization = speech_providers("balanced")

    run_dir = run_pipeline(
        brief_path,
        settings=settings,
        source_client=source,
        transcription_provider=asr,
        alignment_provider=alignment,
        diarization_provider=diarization,
        render=render,
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("production pipeline returned an invalid manifest")
    if render and manifest.get("status") != "SUCCESS":
        raise RuntimeError(
            f"production pipeline did not reach SUCCESS: {manifest.get('status_reason')}"
        )
    if not render and manifest.get("status") == "FAILED":
        raise RuntimeError(
            f"planning pipeline failed before render: {manifest.get('status_reason')}"
        )

    metadata = manifest.get("run_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("production manifest is missing run_metadata")
    source_meta = metadata.get("source_hashes")
    if not isinstance(source_meta, dict) or str(source_meta.get(source.video.video_id) or "") != (
        source.source_sha256
    ):
        raise RuntimeError("pipeline did not process the Modal-acquired source master hash")
    metadata["source_execution"] = {
        "mode": "modal-native",
        "local_source_reused": False,
        "source_volume": "clipper-media-cache",
        "source_mount_path": source.source_path.as_posix(),
        "source_sha256": source.source_sha256,
    }
    if resume_from_run_id is not None:
        metadata["resume"] = {
            "from_run_id": resume_from_run_id,
            "mode": "provenance-only",
            "local_source_reused": False,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    artifact_volume.commit()
    run_relative = "/" + str(run_dir.relative_to(Path(ARTIFACT_ROOT))).replace(os.sep, "/")
    return {
        "status": "PASS",
        "run_volume": "clipper-v10-artifacts",
        "run_path": run_relative,
        "source": source_evidence,
        "pipeline_status": manifest.get("status"),
        "rendered_finalists": len(manifest.get("rendered_clips") or []),
        "initial_shortlist": len(manifest.get("submission_shortlist") or []),
        "review_status": "PENDING_ACTUAL_MP4_REVIEW" if render else "NOT_RENDERED",
    }
