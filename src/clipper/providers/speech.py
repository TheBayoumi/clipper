from __future__ import annotations

import importlib
import os
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..canonical import CanonicalTimeline, CanonicalWord
from .base import InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable


def _usage(started_at: str, started: float, *, provider: str = "local") -> InferenceUsage:
    return InferenceUsage(
        provider=provider,
        started_at=started_at,
        duration_seconds=max(0.0, time.perf_counter() - started),
    )


def _started() -> tuple[str, float]:
    return datetime.now(UTC).isoformat(), time.perf_counter()


def _normalize_token(value: object) -> str:
    text = str(value or "").casefold().strip()
    return "".join(char for char in text if char.isalnum() or char == "'")


def _float_value(value: object, fallback: float) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return fallback


def _replace_words(
    timeline: CanonicalTimeline,
    replacements: dict[str, dict[str, object]],
) -> CanonicalTimeline:
    words: list[CanonicalWord] = []
    for word in timeline.words:
        update = replacements.get(word.word_id, {})
        words.append(
            CanonicalWord(
                word_id=word.word_id,
                text=word.text,
                source_start=_float_value(update.get("source_start"), word.source_start),
                source_end=_float_value(update.get("source_end"), word.source_end),
                speaker_id=(
                    str(update["speaker_id"])
                    if update.get("speaker_id") is not None
                    else word.speaker_id
                ),
                confidence=(
                    _float_value(update.get("confidence"), word.confidence or 0.0)
                    if update.get("confidence") is not None
                    else word.confidence
                ),
                timing_mode=str(update.get("timing_mode", word.timing_mode)),  # type: ignore[arg-type]
                transcript_source=str(update.get("transcript_source", word.transcript_source)),
            )
        )
    return CanonicalTimeline(timeline.video_id, timeline.source_hash, tuple(words))


def _alignment_segments(
    timeline: CanonicalTimeline, *, max_seconds: float = 30.0
) -> list[dict[str, object]]:
    if not timeline.words:
        return []
    groups: list[list[CanonicalWord]] = []
    current: list[CanonicalWord] = []
    group_start = timeline.words[0].source_start
    previous_end = group_start
    for word in timeline.words:
        if current and (
            word.source_start - previous_end > 1.0 or word.source_end - group_start > max_seconds
        ):
            groups.append(current)
            current = []
            group_start = word.source_start
        current.append(word)
        previous_end = word.source_end
    if current:
        groups.append(current)
    return [
        {
            "start": group[0].source_start,
            "end": group[-1].source_end,
            "text": " ".join(word.text for word in group),
            "word_ids": [word.word_id for word in group],
        }
        for group in groups
    ]


def _alignment_confidence(update: dict[str, object]) -> float:
    value = update.get("confidence")
    if isinstance(value, int | float) and 0.0 <= float(value) <= 1.0:
        return float(value)
    return -1.0


def _alignment_deviation(word: CanonicalWord, update: dict[str, object]) -> float:
    value = update.get("source_start")
    return abs(float(value) - word.source_start) if isinstance(value, int | float) else float("inf")


def _drop_nonmonotonic_alignment_updates(
    timeline: CanonicalTimeline,
    replacements: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Keep raw WhisperX timings only when they preserve immutable source word order."""
    accepted = dict(replacements)
    by_id = {word.word_id: word for word in timeline.words}
    while True:
        starts = [
            _float_value(accepted.get(word.word_id, {}).get("source_start"), word.source_start)
            for word in timeline.words
        ]
        violation = next(
            (index for index, (left, right) in enumerate(pairwise(starts)) if left > right),
            None,
        )
        if violation is None:
            return accepted
        left_word = timeline.words[violation]
        right_word = timeline.words[violation + 1]
        left = accepted.get(left_word.word_id)
        right = accepted.get(right_word.word_id)
        if (
            left is None and right is None
        ):  # pragma: no cover - the input timeline is already ordered
            raise ValueError("canonical source order is invalid before alignment")
        if left is None:
            accepted.pop(right_word.word_id, None)
            continue
        if right is None:
            accepted.pop(left_word.word_id, None)
            continue
        left_rank = (
            _alignment_confidence(left),
            -_alignment_deviation(by_id[left_word.word_id], left),
        )
        right_rank = (
            _alignment_confidence(right),
            -_alignment_deviation(by_id[right_word.word_id], right),
        )
        rejected_id = left_word.word_id if left_rank <= right_rank else right_word.word_id
        accepted.pop(rejected_id, None)


def apply_whisperx_alignment(
    timeline: CanonicalTimeline,
    aligned_segments: list[dict[str, object]],
) -> CanonicalTimeline:
    """Map WhisperX word timings back onto immutable canonical word IDs in source order."""
    canonical = list(timeline.words)
    cursor = 0
    replacements: dict[str, dict[str, object]] = {}
    for segment in aligned_segments:
        raw_words = segment.get("words")
        if not isinstance(raw_words, list):
            continue
        for raw in raw_words:
            if not isinstance(raw, dict):
                continue
            start = raw.get("start")
            end = raw.get("end")
            token = _normalize_token(raw.get("word"))
            if start is None or end is None or not token:
                continue
            match: int | None = None
            for index in range(cursor, min(len(canonical), cursor + 5)):
                if _normalize_token(canonical[index].text) == token:
                    match = index
                    break
            if match is None:
                continue
            word = canonical[match]
            aligned_start = float(start)
            aligned_end = float(end)
            if aligned_start < 0 or aligned_end <= aligned_start:
                continue
            score = raw.get("score")
            confidence = (
                float(score)
                if isinstance(score, int | float) and 0.0 <= float(score) <= 1.0
                else None
            )
            replacements[word.word_id] = {
                "source_start": aligned_start,
                "source_end": aligned_end,
                "confidence": confidence,
                "timing_mode": "aligned",
                "transcript_source": f"{word.transcript_source}+whisperx",
            }
            cursor = match + 1
    if not replacements:
        raise ValueError("WhisperX alignment produced no canonical word matches")
    stable = _drop_nonmonotonic_alignment_updates(timeline, replacements)
    if not stable:
        raise ValueError("WhisperX alignment produced no source-ordered canonical word matches")
    return _replace_words(timeline, stable)


def apply_speaker_turns(
    timeline: CanonicalTimeline,
    turns: list[tuple[float, float, str]],
) -> CanonicalTimeline:
    """Assign each word to the speaker turn with maximum temporal overlap."""
    replacements: dict[str, dict[str, object]] = {}
    for word in timeline.words:
        best_speaker: str | None = None
        best_overlap = 0.0
        for start, end, speaker in turns:
            overlap = max(0.0, min(word.source_end, end) - max(word.source_start, start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        if best_speaker is not None:
            replacements[word.word_id] = {"speaker_id": best_speaker}
    return _replace_words(timeline, replacements)


class FasterWhisperTranscriptionProvider:
    def __init__(
        self,
        model_id: str = "large-v3-turbo",
        revision: str = "main",
        *,
        device: str = "auto",
        compute_type: str = "int8_float16",
        schema_version: str = "canonical-timeline-v1",
    ) -> None:
        identity_model_id = model_id if "/" in model_id else f"faster-whisper/{model_id}"
        self.identity = ModelIdentity(
            identity_model_id, revision, compute_type, "faster-whisper", "none", schema_version
        )
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("faster_whisper")
            except ImportError as exc:
                raise ProviderUnavailable("install clipper[asr]") from exc
            self._model = module.WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                revision=self.identity.revision,
            )
        return self._model

    def transcribe(
        self, source: Path, *, video_id: str, source_hash: str
    ) -> ProviderResult[CanonicalTimeline]:
        started_at, started = _started()
        segments, _info = self._load().transcribe(
            str(source),
            word_timestamps=True,
            vad_filter=True,
        )
        words: list[CanonicalWord] = []
        index = 0
        for segment in segments:
            for raw in getattr(segment, "words", None) or ():
                start = getattr(raw, "start", None)
                end = getattr(raw, "end", None)
                text = str(getattr(raw, "word", "")).strip()
                if start is None or end is None or not text:
                    continue
                word_id = f"{video_id}:w{index:07d}"
                probability = getattr(raw, "probability", None)
                words.append(
                    CanonicalWord(
                        word_id,
                        text,
                        float(start),
                        float(end),
                        None,
                        float(probability) if probability is not None else None,
                        "word_exact",
                        "faster-whisper-large-v3-turbo",
                    )
                )
                index += 1
        if not words:
            raise ValueError("faster-whisper produced no timestamped words")
        return ProviderResult(
            CanonicalTimeline(video_id, source_hash, tuple(words)),
            self.identity,
            _usage(started_at, started),
        )


class WhisperXAlignmentProvider:
    def __init__(
        self,
        model_id: str | None = None,
        revision: str = "main",
        *,
        device: str = "cuda",
        language_code: str = "en",
        schema_version: str = "canonical-timeline-v1",
    ) -> None:
        self.model_id = model_id
        self.identity = ModelIdentity(
            model_id or "whisperx-forced-alignment",
            revision,
            "none",
            "whisperx",
            "none",
            schema_version,
        )
        self.device = device
        self.language_code = language_code

    def align(self, source: Path, timeline: CanonicalTimeline) -> ProviderResult[CanonicalTimeline]:
        if not timeline.words:
            raise ValueError("cannot align an empty canonical timeline")
        started_at, started = _started()
        try:
            whisperx = importlib.import_module("whisperx")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[alignment]") from exc
        audio = whisperx.load_audio(str(source))
        model_name: str | None = None
        if self.model_id is not None:
            try:
                hub = importlib.import_module("huggingface_hub")
            except ImportError as exc:
                raise ProviderUnavailable(
                    "huggingface_hub is required for pinned alignment"
                ) from exc
            model_name = str(
                hub.snapshot_download(
                    repo_id=self.model_id,
                    revision=self.identity.revision,
                )
            )
        model, metadata = whisperx.load_align_model(
            language_code=self.language_code,
            device=self.device,
            model_name=model_name,
            model_cache_only=model_name is not None,
        )
        segments = _alignment_segments(timeline)
        payload = [{k: v for k, v in item.items() if k != "word_ids"} for item in segments]
        aligned = whisperx.align(
            payload,
            model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        raw_segments = aligned.get("segments") if isinstance(aligned, dict) else None
        if not isinstance(raw_segments, list):
            raise ValueError("WhisperX returned no aligned segments")
        value = apply_whisperx_alignment(timeline, raw_segments)
        return ProviderResult(value, self.identity, _usage(started_at, started))


class PyannoteDiarizationProvider:
    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        revision: str = "main",
        *,
        token: str | None = None,
        device: str | None = None,
        schema_version: str = "canonical-timeline-v1",
    ) -> None:
        self.identity = ModelIdentity(
            model_id, revision, "none", "pyannote.audio", "none", schema_version
        )
        self.token = token or os.getenv("HF_TOKEN")
        self.device = device
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if not self.token:
            raise ProviderUnavailable(
                "HF_TOKEN is required after accepting the pyannote community-1 model agreement"
            )
        if self._pipeline is None:
            try:
                module = importlib.import_module("pyannote.audio")
            except ImportError as exc:
                raise ProviderUnavailable("install clipper[diarization]") from exc
            self._pipeline = module.Pipeline.from_pretrained(
                self.identity.model_id,
                token=self.token,
                revision=self.identity.revision,
            )
            if self.device:
                try:
                    torch = importlib.import_module("torch")
                    self._pipeline.to(torch.device(self.device))
                except (ImportError, AttributeError) as exc:
                    raise ProviderUnavailable("torch device support is unavailable") from exc
        return self._pipeline

    @staticmethod
    def _turns(output: object) -> list[tuple[float, float, str]]:
        diarization = getattr(output, "speaker_diarization", output)
        iterator = getattr(diarization, "itertracks", None)
        if not callable(iterator):
            raise ValueError("pyannote output has no speaker diarization tracks")
        turns: list[tuple[float, float, str]] = []
        for segment, _track, speaker in iterator(yield_label=True):
            start = float(segment.start)
            end = float(segment.end)
            if end > start:
                turns.append((start, end, str(speaker)))
        if not turns:
            raise ValueError("pyannote produced no speaker turns")
        return turns

    def diarize(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]:
        started_at, started = _started()
        output = self._load()(str(source))
        value = apply_speaker_turns(timeline, self._turns(output))
        return ProviderResult(value, self.identity, _usage(started_at, started))


class PassthroughDiarizationProvider:
    """Diagnostic-only diarization that preserves aligned words without speaker labels."""

    def __init__(self) -> None:
        self.identity = ModelIdentity(
            "none/passthrough-diarization",
            "v1",
            "none",
            "deterministic-passthrough",
            "none",
            "canonical-timeline-v1",
        )

    def diarize(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]:
        started_at, started = _started()
        return ProviderResult(
            timeline,
            self.identity,
            _usage(started_at, started, provider="degraded"),
            degraded=True,
        )
