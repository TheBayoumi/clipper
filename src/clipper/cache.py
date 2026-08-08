from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import (
    CampaignBrief,
    ClipConcept,
    EditorialScores,
    StoryMoment,
    TranscriptSegment,
    TranscriptWord,
)
from .providers.base import ModelIdentity

CACHE_SCHEMA_VERSION = "clipper-v10-analysis-1"


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def transcript_cache_key(
    video_id: str,
    source_hash: str,
    *,
    engine: str,
    model: str = "",
    language: str = "en",
) -> str:
    return stable_hash(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": "transcript",
            "video_id": video_id,
            "source_hash": source_hash,
            "engine": engine,
            "model": model,
            "language": language,
        }
    )


def analysis_cache_key(
    video_id: str,
    segments: list[TranscriptSegment],
    brief: CampaignBrief,
) -> str:
    campaign = brief.to_dict()
    return stable_hash(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": "editorial-analysis",
            "video_id": video_id,
            "transcript": [segment.to_dict() for segment in segments],
            "campaign": {
                "keywords": campaign["keywords"],
                "negative_keywords": campaign["negative_keywords"],
                "required_phrases": campaign["required_phrases"],
                "min_clip_seconds": campaign["min_clip_seconds"],
                "max_clip_seconds": campaign["max_clip_seconds"],
                "production": campaign["production"],
                "diversity": campaign["diversity"],
                "hooks": campaign["hooks"],
                "editorial": campaign["editorial"],
            },
        }
    )


def model_stage_cache_key(
    stage: str,
    *,
    source_hash: str,
    campaign: dict[str, object],
    model: ModelIdentity,
    payload: object,
    sampling: dict[str, object] | None = None,
) -> str:
    """Cache expensive model output by exact source/model/prompt/schema/config inputs."""
    return stable_hash(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": stage,
            "source_hash": source_hash,
            "model": model.to_dict(),
            "model_fingerprint": model.cache_fingerprint(sampling=sampling),
            "campaign": campaign,
            "payload": payload,
            "sampling": sampling or {},
        }
    )


class FileCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str, name: str) -> Path:
        return self.root / key[:2] / key / f"{name}.json"

    def read(self, key: str, name: str) -> object | None:
        path = self._path(key, name)
        if not path.is_file():
            return None
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
            return value
        except (OSError, json.JSONDecodeError):
            return None

    def write(self, key: str, name: str, payload: object) -> Path:
        path = self._path(key, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path


def transcript_segments_from_payload(payload: object) -> list[TranscriptSegment]:
    if not isinstance(payload, list):
        raise ValueError("cached transcript payload must be a list")
    segments: list[TranscriptSegment] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("cached transcript segment must be an object")
        raw_words = item.get("words")
        if raw_words is None:
            raw_words = []
        if not isinstance(raw_words, list):
            raise ValueError("cached transcript words must be a list")
        words = tuple(
            TranscriptWord(float(word["start"]), float(word["end"]), str(word["text"]))
            for word in raw_words
            if isinstance(word, dict)
        )
        segments.append(
            TranscriptSegment(float(item["start"]), float(item["end"]), str(item["text"]), words)
        )
    return segments


def story_moments_from_payload(payload: object) -> list[StoryMoment]:
    if not isinstance(payload, list):
        raise ValueError("cached story moments must be a list")
    moments: list[StoryMoment] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("scores"), dict):
            raise ValueError("cached story moment is invalid")
        moments.append(
            StoryMoment(
                moment_id=str(item["moment_id"]),
                video_id=str(item["video_id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                moment_type=str(item["moment_type"]),
                topic=str(item["topic"]),
                setup=str(item["setup"]),
                payoff=str(item["payoff"]),
                scores=EditorialScores(**item["scores"]),
                score=float(item["score"]),
                transcript_fingerprint=str(item["transcript_fingerprint"]),
            )
        )
    return moments


def clip_concepts_from_payload(payload: object) -> list[ClipConcept]:
    if not isinstance(payload, list):
        raise ValueError("cached clip concepts must be a list")
    concepts: list[ClipConcept] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("scores"), dict):
            raise ValueError("cached clip concept is invalid")
        concepts.append(
            ClipConcept(
                concept_id=str(item["concept_id"]),
                video_id=str(item["video_id"]),
                source_start=float(item["source_start"]),
                source_end=float(item["source_end"]),
                text=str(item["text"]),
                topic=str(item["topic"]),
                setup=str(item["setup"]),
                payoff=str(item["payoff"]),
                moment_type=str(item["moment_type"]),
                recommended_duration=float(item["recommended_duration"]),
                scores=EditorialScores(**item["scores"]),
                score=float(item["score"]),
                semantic_cluster=str(item["semantic_cluster"]),
                transcript_fingerprint=str(item["transcript_fingerprint"]),
            )
        )
    return concepts
