from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.getenv("CLIPPER_V10_MODAL_APP", "clipper-v10-cycle")
MODEL_APP = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
MEDIA_ROOT = "/media"
ARTIFACT_ROOT = "/artifacts"
NEMO_CACHE = "/nemo-cache"
SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"

app = modal.App(APP_NAME)
media_cache = modal.Volume.from_name("clipper-media-cache", create_if_missing=True)
artifact_volume = modal.Volume.from_name("clipper-v10-artifacts", create_if_missing=True)
nemo_cache = modal.Volume.from_name("clipper-nemo-cache", create_if_missing=True)

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
        "yt-dlp>=2026.7.4,<2027",
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
    .uv_pip_install("modal>=1.5.2,<2", "huggingface-hub>=1.24,<2")
)

nemo_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "torch>=2.8,<3",
        "numba==0.66.0",
        "nemo_toolkit[asr]>=2.5,<3",
        "huggingface-hub>=0.28,<2",
    )
    .env({"HF_HOME": NEMO_CACHE})
)

_sortformer: Any | None = None
_sortformer_revision: str | None = None


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
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
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
    """Acquire the highest available YouTube source once and retain the exact master."""

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


@app.function(
    image=nemo_image,
    gpu="L4",
    volumes={MEDIA_ROOT: media_cache, NEMO_CACHE: nemo_cache},
    timeout=7200,
    memory=24576,
    scaledown_window=2,
)
def diarize_open(payload: dict[str, Any]) -> dict[str, Any]:
    """Run public NeMo Streaming Sortformer diarization on a derived 16 kHz analysis WAV."""

    global _sortformer, _sortformer_revision
    import torch
    from huggingface_hub import HfApi
    from nemo.collections.asr.models import SortformerEncLabelModel

    source = Path(str(payload.get("source_path") or ""))
    if not source.is_file():
        raise ValueError(f"diarization source does not exist: {source}")
    started = time.perf_counter()
    scratch = Path("/tmp") / f"clipper-diar-{source.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    wav = scratch / "analysis-16k-mono.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if _sortformer is None:
            _sortformer = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL).eval()
            if torch.cuda.is_available():
                _sortformer = _sortformer.cuda()
            _sortformer_revision = str(HfApi().model_info(SORTFORMER_MODEL).sha or "unknown")
            nemo_cache.commit()
        raw = _sortformer.diarize(audio=[str(wav)], batch_size=1, verbose=False)
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
            raise RuntimeError("NeMo Sortformer returned invalid diarization output")
        turns: list[list[Any]] = []
        for line in raw[0]:
            parts = str(line).split()
            if len(parts) != 3:
                raise RuntimeError(f"invalid Sortformer segment: {line}")
            start, end, speaker = float(parts[0]), float(parts[1]), str(parts[2])
            if end > start:
                turns.append([start, end, speaker])
        if not turns:
            raise RuntimeError("NeMo Sortformer returned no speaker turns")
        duration = max(0.0, time.perf_counter() - started)
        peak = (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if torch.cuda.is_available()
            else None
        )
        return {
            "turns": turns,
            "model": {
                "model_id": SORTFORMER_MODEL,
                "revision": _sortformer_revision or "unknown",
            },
            "analysis_audio": {
                "sample_rate": 16000,
                "channels": 1,
                "derivative_only": True,
                "source_master_unchanged": True,
            },
            "usage": {
                "provider": "modal",
                "duration_seconds": duration,
                "gpu_type": "L4",
                "gpu_seconds": duration,
                "peak_vram_mb": peak,
                "estimated_cost_usd": duration * 0.000222,
            },
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class VolumeSourceClient:
    def __init__(self, evidence: dict[str, Any], channel_id: str) -> None:
        from clipper.models import VideoCandidate

        self.source_path = Path(str(evidence["mount_path"]))
        self.source_sha256 = str(evidence["sha256"])
        self.duration = float(evidence["duration_seconds"])
        self.video = VideoCandidate(
            video_id=str(evidence["video_id"]),
            title="Double Coverage authorized production source",
            channel_id=channel_id,
            channel_title="Double Coverage Podcast",
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


@app.function(
    image=runner_image,
    volumes={MEDIA_ROOT: media_cache, ARTIFACT_ROOT: artifact_volume},
    timeout=21600,
    memory=16384,
    scaledown_window=2,
)
def run_full_cycle(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the complete podcast pipeline and simulated HILP review in Modal."""

    from clipper.hilp import simulate_hilp_cycle, validate_hilp_evidence
    from clipper.pipeline import PipelineSettings, run_pipeline
    from clipper.providers.base import ModelIdentity
    from clipper.providers.factory import speech_providers
    from clipper.providers.modal_speech import ModalDiarizationProvider, ModalMediaBridge
    from clipper.render import FFmpegRenderer

    source_evidence = payload.get("source")
    if not isinstance(source_evidence, dict):
        raise ValueError("run_full_cycle requires source evidence")
    brief_yaml = str(payload.get("brief_yaml") or "")
    channel_id = str(payload.get("channel_id") or "")
    if not brief_yaml.strip() or not channel_id:
        raise ValueError("run_full_cycle requires campaign brief and channel ID")
    if source_evidence.get("quality_policy") != "highest_available_no_transcode":
        raise RuntimeError("source master violates no-downgrade policy")

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
    asr, alignment, _ = speech_providers("balanced")
    bridge = ModalMediaBridge("clipper-media-cache")
    diarization = ModalDiarizationProvider(
        app_name=APP_NAME,
        function_name="diarize_open",
        identity=ModelIdentity(
            SORTFORMER_MODEL,
            "runtime-resolved",
            "none",
            "modal-nemo-sortformer",
            "none",
            "canonical-timeline-v1",
        ),
        media_bridge=bridge,
    )

    run_dir = run_pipeline(
        brief_path,
        settings=settings,
        source_client=source,
        transcription_provider=asr,
        alignment_provider=alignment,
        diarization_provider=diarization,
        render=True,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "SUCCESS":
        raise RuntimeError(
            f"production pipeline did not reach SUCCESS: {manifest.get('status_reason')}"
        )
    source_meta = manifest.get("run_metadata", {}).get("source_hashes", {})
    if str(source_meta.get(source.video.video_id) or "") != source.source_sha256:
        raise RuntimeError("pipeline did not process the acquired source master hash")

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
    watermark = run_dir / "assets" / "watermark.png"
    hilp = simulate_hilp_cycle(
        run_dir,
        source_path=source.source_path,
        renderer=renderer,
        watermark_path=watermark if watermark.is_file() else None,
    )
    validate_hilp_evidence(hilp)
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
        "hilp_status": hilp.get("status"),
        "branches_exercised": hilp.get("branches_exercised"),
        "final_shortlist": hilp.get("final_shortlist"),
    }
