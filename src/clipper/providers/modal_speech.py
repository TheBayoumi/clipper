from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ..canonical import CanonicalTimeline, canonical_timeline_from_word_payloads
from .base import InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable
from .speech import apply_speaker_turns, apply_whisperx_alignment


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

    def ensure_uploaded(self, source: Path, source_hash: str) -> str:
        key = (str(source.resolve()), source_hash)
        cached = self._uploaded.get(key)
        if cached is not None:
            return cached
        suffix = source.suffix.lower() or ".bin"
        remote_volume_path = f"/inputs/{source_hash}{suffix}"
        remote_mount_path = f"/media{remote_volume_path}"
        volume = self._volume()
        try:
            existing = list(volume.listdir(remote_volume_path))
        except Exception:
            existing = []
        if not existing:
            with volume.batch_upload(force=True) as upload:
                upload.put_file(str(source), remote_volume_path)
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
