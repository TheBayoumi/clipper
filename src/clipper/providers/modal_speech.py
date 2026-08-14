from __future__ import annotations

import importlib
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..canonical import CanonicalTimeline, canonical_timeline_from_word_payloads
from .base import InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable
from .speech import apply_speaker_turns, apply_whisperx_alignment

LOGGER = logging.getLogger(__name__)
_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mka",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
}


class ModalMediaBridge:
    def __init__(self, volume_name: str = "clipper-media-cache") -> None:
        self.volume_name = volume_name
        self._uploaded: dict[tuple[str, str], str] = {}

    def _volume(self) -> Any:
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[modal]") from exc
        return modal.Volume.from_name(self.volume_name, create_if_missing=True)

    @staticmethod
    def _speech_source(source: Path, source_hash: str) -> Path:
        """Return a compact speech-only derivative for video inputs.

        Speech inference never needs the video stream. Uploading multi-gigabyte podcast
        masters through the local Modal client can exceed the Volume upload deadline, so
        video inputs are decoded once to 16 kHz mono PCM and cached beside the source.
        The derivative is keyed by the full source hash so it is safe to reuse across runs.
        """
        if source.suffix.lower() in _AUDIO_SUFFIXES:
            return source
        cache_dir = source.parent / ".speech-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{source_hash}.speech.wav"
        if target.is_file() and target.stat().st_size > 44:
            LOGGER.info(
                "reusing cached speech derivative %s (%d bytes)",
                target,
                target.stat().st_size,
            )
            return target

        temporary = cache_dir / f"{source_hash}.speech.tmp.wav"
        temporary.unlink(missing_ok=True)
        LOGGER.info(
            "extracting 16 kHz mono speech derivative from %s (%d bytes)",
            source,
            source.stat().st_size,
        )
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
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(temporary),
                ],
                check=True,
                timeout=1800,
            )
            if not temporary.is_file() or temporary.stat().st_size <= 44:
                raise RuntimeError("ffmpeg produced no usable speech derivative")
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        LOGGER.info(
            "speech derivative ready %s (%d bytes; %.1f%% of source)",
            target,
            target.stat().st_size,
            (target.stat().st_size / max(source.stat().st_size, 1)) * 100.0,
        )
        return target

    def ensure_uploaded(self, source: Path, source_hash: str) -> str:
        key = (str(source.resolve()), source_hash)
        cached = self._uploaded.get(key)
        if cached is not None:
            return cached

        transfer_source = self._speech_source(source, source_hash)
        suffix = transfer_source.suffix.lower() or ".bin"
        is_derivative = transfer_source != source
        remote_volume_path = (
            f"/inputs/{source_hash}.speech{suffix}"
            if is_derivative
            else f"/inputs/{source_hash}{suffix}"
        )
        remote_mount_path = f"/media{remote_volume_path}"
        volume = self._volume()
        try:
            existing = list(volume.listdir(remote_volume_path))
        except Exception:
            existing = []
        if not existing:
            LOGGER.info(
                "uploading speech media to Modal volume %s: %s -> %s (%d bytes)",
                self.volume_name,
                transfer_source,
                remote_volume_path,
                transfer_source.stat().st_size,
            )
            with volume.batch_upload(force=True) as upload:
                upload.put_file(str(transfer_source), remote_volume_path)
            LOGGER.info("Modal speech media upload complete: %s", remote_volume_path)
        else:
            LOGGER.info("reusing Modal speech media %s", remote_volume_path)
        self._uploaded[key] = remote_mount_path
        return remote_mount_path


class _ModalSpeechBase:
    def __init__(
        self,
        *,
        app_name: str,
        function_name: str,
        identity: ModelIdentity,
        media_bridge: ModalMediaBridge,
    ) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.identity = identity
        self.media_bridge = media_bridge

    def _function(self) -> Any:
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[modal]") from exc
        return modal.Function.from_name(self.app_name, self.function_name)

    def _resolved_identity(self, response: dict[str, Any]) -> ModelIdentity:
        raw = response.get("model")
        if not isinstance(raw, dict):
            return self.identity
        return ModelIdentity(
            str(raw.get("model_id") or self.identity.model_id),
            str(raw.get("revision") or self.identity.revision),
            self.identity.quantization,
            self.identity.inference_engine,
            self.identity.prompt_version,
            self.identity.schema_version,
        )

    @staticmethod
    def _usage(response: dict[str, Any]) -> InferenceUsage:
        raw = response.get("usage")
        usage: dict[str, Any] = raw if isinstance(raw, dict) else {}
        return InferenceUsage(
            provider="modal",
            started_at=str(usage.get("started_at") or "unknown"),
            duration_seconds=float(usage.get("duration_seconds") or 0.0),
            gpu_type=str(usage["gpu_type"]) if usage.get("gpu_type") else None,
            gpu_seconds=float(usage.get("gpu_seconds") or 0.0),
            peak_vram_mb=(
                float(usage["peak_vram_mb"]) if usage.get("peak_vram_mb") is not None else None
            ),
            input_units=int(usage.get("input_units") or 0),
            output_units=int(usage.get("output_units") or 0),
            estimated_cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
        )


class ModalTranscriptionProvider(_ModalSpeechBase):
    def transcribe(
        self, source: Path, *, video_id: str, source_hash: str
    ) -> ProviderResult[CanonicalTimeline]:
        remote_path = self.media_bridge.ensure_uploaded(source, source_hash)
        response = self._function().remote(
            {"source_path": remote_path, "video_id": video_id, "source_hash": source_hash}
        )
        if not isinstance(response, dict) or not isinstance(response.get("words"), list):
            raise ValueError("Modal transcription provider returned an invalid response")
        words = [item for item in response["words"] if isinstance(item, dict)]
        timeline = canonical_timeline_from_word_payloads(
            video_id,
            source_hash,
            words,
            transcript_source="modal-faster-whisper-large-v3-turbo",
        )
        return ProviderResult(timeline, self._resolved_identity(response), self._usage(response))


class ModalAlignmentProvider(_ModalSpeechBase):
    def align(self, source: Path, timeline: CanonicalTimeline) -> ProviderResult[CanonicalTimeline]:
        remote_path = self.media_bridge.ensure_uploaded(source, timeline.source_hash)
        response = self._function().remote(
            {"source_path": remote_path, "timeline": timeline.to_dict()}
        )
        raw_segments = response.get("segments") if isinstance(response, dict) else None
        if not isinstance(raw_segments, list):
            raise ValueError("Modal alignment provider returned an invalid response")
        segments = [item for item in raw_segments if isinstance(item, dict)]
        value = apply_whisperx_alignment(timeline, segments)
        return ProviderResult(value, self._resolved_identity(response), self._usage(response))


class ModalDiarizationProvider(_ModalSpeechBase):
    def diarize(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]:
        remote_path = self.media_bridge.ensure_uploaded(source, timeline.source_hash)
        response = self._function().remote({"source_path": remote_path})
        if isinstance(response, dict) and isinstance(response.get("error"), dict):
            error = response["error"]
            raise RuntimeError(
                f"Modal diarization failed: {error.get('type', 'RemoteError')}: "
                f"{error.get('message', 'unknown remote error')}"
            )
        raw_turns = response.get("turns") if isinstance(response, dict) else None
        if not isinstance(raw_turns, list):
            raise ValueError("Modal diarization provider returned an invalid response")
        turns: list[tuple[float, float, str]] = []
        for raw in raw_turns:
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise ValueError("Modal diarization turn is invalid")
            turns.append((float(raw[0]), float(raw[1]), str(raw[2])))
        value = apply_speaker_turns(timeline, turns)
        return ProviderResult(value, self._resolved_identity(response), self._usage(response))
