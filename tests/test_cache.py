import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from clipper.cache import (
    FileCache,
    analysis_cache_key,
    clip_concepts_from_payload,
    file_sha256,
    model_stage_cache_key,
    stable_hash,
    story_moments_from_payload,
    transcript_cache_key,
    transcript_segments_from_payload,
)
from clipper.models import CampaignBrief, TranscriptSegment, TranscriptWord
from clipper.providers.base import ModelIdentity


def brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "cache-test",
            "title": "Cache",
            "objective": "test",
            "allowed_video_ids": ["v"],
            "rights_confirmed": True,
            "min_clip_seconds": 10,
            "max_clip_seconds": 30,
        }
    )


def test_stable_hash_and_file_hash_are_deterministic(tmp_path: Path) -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == file_sha256(path) and len(file_sha256(path)) == 64


def test_cache_keys_change_with_relevant_inputs() -> None:
    one = transcript_cache_key("v", "hash", engine="asr", model="small", language="en")
    two = transcript_cache_key("v", "hash", engine="asr", model="medium", language="en")
    assert one != two

    identity = ModelIdentity(
        model_id="editor",
        revision="rev-a",
        quantization="bf16",
        inference_engine="test",
        prompt_version="editor",
        schema_version="schema-a",
    )
    base = model_stage_cache_key(
        "semantic_cores:1",
        source_hash="source-a",
        campaign=brief().to_dict(),
        model=identity,
        payload={"timeline": "a"},
        sampling={"temperature": 0.0},
    )
    changed_source = model_stage_cache_key(
        "semantic_cores:1",
        source_hash="source-b",
        campaign=brief().to_dict(),
        model=identity,
        payload={"timeline": "a"},
        sampling={"temperature": 0.0},
    )
    changed_payload = model_stage_cache_key(
        "semantic_cores:1",
        source_hash="source-a",
        campaign=brief().to_dict(),
        model=identity,
        payload={"timeline": "b"},
        sampling={"temperature": 0.0},
    )
    changed_model = model_stage_cache_key(
        "semantic_cores:1",
        source_hash="source-a",
        campaign=brief().to_dict(),
        model=ModelIdentity(
            model_id="editor",
            revision="rev-b",
            quantization="bf16",
            inference_engine="test",
            prompt_version="editor",
            schema_version="schema-a",
        ),
        payload={"timeline": "a"},
        sampling={"temperature": 0.0},
    )
    assert len({base, changed_source, changed_payload, changed_model}) == 4


def test_removed_heuristic_analysis_cache_fails_closed() -> None:
    segments = [TranscriptSegment(0, 2, "creator story")]
    with pytest.raises(RuntimeError, match="autonomous quality graph"):
        analysis_cache_key("v", segments, brief())


def test_file_cache_round_trip_and_corruption(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache")
    assert cache.read("a" * 64, "data") is None
    path = cache.write("a" * 64, "data", {"ok": True})
    assert cache.read("a" * 64, "data") == {"ok": True}
    path.write_text("{broken", encoding="utf-8")
    assert cache.read("a" * 64, "data") is None


def test_file_cache_concurrent_writers_use_unique_atomic_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache")
    key = "b" * 64
    barrier = threading.Barrier(8)
    original_replace = Path.replace

    def synchronized_replace(source: Path, target: Path) -> Path:
        if source.name.endswith(".tmp"):
            barrier.wait(timeout=5)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(
            executor.map(
                lambda _index: cache.write(key, "canonical", {"ok": True}),
                range(8),
            )
        )

    assert len(set(paths)) == 1
    assert cache.read(key, "canonical") == {"ok": True}
    assert not list(paths[0].parent.glob(".*.tmp"))


def test_transcript_cache_deserialization_preserves_words() -> None:
    payload = [
        {
            "start": 1.0,
            "end": 3.0,
            "text": "hello world",
            "speaker_id": "SPEAKER_00",
            "words": [
                {"start": 1.0, "end": 1.5, "text": "hello"},
                {"start": 1.6, "end": 2.2, "text": "world"},
            ],
        }
    ]
    segments = transcript_segments_from_payload(payload)
    assert segments == [
        TranscriptSegment(
            1.0,
            3.0,
            "hello world",
            (TranscriptWord(1.0, 1.5, "hello"), TranscriptWord(1.6, 2.2, "world")),
            "SPEAKER_00",
        )
    ]
    with pytest.raises(ValueError, match="list"):
        transcript_segments_from_payload({})
    with pytest.raises(ValueError, match="segment"):
        transcript_segments_from_payload(["bad"])
    with pytest.raises(ValueError, match="words"):
        transcript_segments_from_payload([{"start": 0, "end": 1, "text": "x", "words": {}}])


def test_historical_story_and_concept_cache_deserialization_is_read_only_compatible() -> None:
    scores = {"quality": 8.0, "confidence": 0.9}
    moment = {
        "moment_id": "m",
        "video_id": "v",
        "start": 1,
        "end": 20,
        "text": "creator story",
        "moment_type": "story",
        "topic": "creator",
        "setup": "setup",
        "payoff": "payoff",
        "scores": scores,
        "score": 8.0,
        "transcript_fingerprint": "fp",
    }
    concept = {
        "concept_id": "c",
        "video_id": "v",
        "source_start": 1,
        "source_end": 20,
        "text": "creator story",
        "topic": "creator",
        "setup": "setup",
        "payoff": "payoff",
        "moment_type": "story",
        "recommended_duration": 19,
        "scores": scores,
        "score": 8.0,
        "semantic_cluster": "cluster-01",
        "transcript_fingerprint": "fp",
    }
    assert story_moments_from_payload([moment])[0].moment_id == "m"
    assert clip_concepts_from_payload([concept])[0].concept_id == "c"
    for loader in (story_moments_from_payload, clip_concepts_from_payload):
        with pytest.raises(ValueError, match="list"):
            loader({})
        with pytest.raises(ValueError, match="invalid"):
            loader([{"scores": "bad"}])
