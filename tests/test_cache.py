from pathlib import Path

import pytest

from clipper.cache import (
    FileCache,
    analysis_cache_key,
    clip_concepts_from_payload,
    file_sha256,
    stable_hash,
    story_moments_from_payload,
    transcript_cache_key,
    transcript_segments_from_payload,
)
from clipper.models import CampaignBrief, TranscriptSegment, TranscriptWord


def brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "cache-test",
            "title": "Cache",
            "objective": "test",
            "keywords": ["creator"],
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
    segments = [TranscriptSegment(0, 2, "creator story")]
    key = analysis_cache_key("v", segments, brief())
    changed = CampaignBrief.from_dict(brief().to_dict() | {"keywords": ["money"]})
    assert key != analysis_cache_key("v", segments, changed)


def test_file_cache_round_trip_and_corruption(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache")
    assert cache.read("a" * 64, "data") is None
    path = cache.write("a" * 64, "data", {"ok": True})
    assert cache.read("a" * 64, "data") == {"ok": True}
    path.write_text("{broken", encoding="utf-8")
    assert cache.read("a" * 64, "data") is None


def test_transcript_cache_deserialization_preserves_words() -> None:
    payload = [
        {
            "start": 1.0,
            "end": 3.0,
            "text": "hello world",
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
        )
    ]
    with pytest.raises(ValueError, match="list"):
        transcript_segments_from_payload({})
    with pytest.raises(ValueError, match="segment"):
        transcript_segments_from_payload(["bad"])
    with pytest.raises(ValueError, match="words"):
        transcript_segments_from_payload([{"start": 0, "end": 1, "text": "x", "words": {}}])


def test_story_and_concept_cache_deserialization() -> None:
    scores = {
        "hook_strength": 8,
        "curiosity": 7,
        "payoff_strength": 8,
        "standalone_clarity": 9,
        "emotional_energy": 5,
        "information_value": 7,
        "controversy_or_tension": 4,
        "quoteability": 7,
        "specificity": 6,
        "campaign_relevance": 8,
        "story_completeness": 9,
        "retention_potential": 8,
    }
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
