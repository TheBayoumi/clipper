import pytest

from clipper.models import BriefValidationError, CampaignBrief, TranscriptSegment


def valid_data() -> dict:
    return {
        "campaign_id": "c1",
        "title": "AI clips",
        "objective": "Explain automation",
        "keywords": ["AI", "automation"],
        "source_channel_ids": ["UC123"],
        "rights_confirmed": True,
    }


def test_brief_parses_and_builds_query() -> None:
    brief = CampaignBrief.from_dict(valid_data())
    assert brief.search_query == "AI clips AI automation"
    assert brief.to_dict()["campaign_id"] == "c1"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"keywords": []}, "keywords"),
        ({"region_code": "USA"}, "region_code"),
        ({"clip_count": 0}, "clip_count"),
        ({"source_limit": 51}, "source_limit"),
        ({"max_clips_per_source": 0}, "max_clips_per_source"),
        ({"min_clip_seconds": 5}, "min_clip_seconds"),
        ({"max_clip_seconds": 181}, "max_clip_seconds"),
        ({"min_clip_seconds": 45, "max_clip_seconds": 20}, "less than"),
        ({"source_channel_ids": [], "allowed_video_ids": []}, "unrestricted"),
    ],
)
def test_brief_rejects_invalid_values(patch: dict, message: str) -> None:
    data = valid_data() | patch
    with pytest.raises(BriefValidationError, match=message):
        CampaignBrief.from_dict(data)


def test_brief_rejects_bad_root_and_missing_fields() -> None:
    with pytest.raises(BriefValidationError, match="root"):
        CampaignBrief.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(BriefValidationError, match="missing required"):
        CampaignBrief.from_dict({})
    with pytest.raises(BriefValidationError, match="list of strings"):
        CampaignBrief.from_dict(valid_data() | {"keywords": "AI"})


def test_transcript_segment_validation() -> None:
    segment = TranscriptSegment(1, 2.5, "hello")
    assert segment.duration == 1.5
    assert segment.to_dict()["text"] == "hello"
    with pytest.raises(ValueError, match="timestamps"):
        TranscriptSegment(2, 1, "bad")
    with pytest.raises(ValueError, match="empty"):
        TranscriptSegment(0, 1, " ")


def test_optional_list_none_becomes_empty_but_requires_other_allowlist() -> None:
    data = valid_data() | {"source_channel_ids": None, "allowed_video_ids": ["v"]}
    brief = CampaignBrief.from_dict(data)
    assert brief.source_channel_ids == []
