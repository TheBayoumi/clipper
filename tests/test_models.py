import pytest

from clipper.models import BriefValidationError, CampaignBrief, TranscriptSegment, TranscriptWord


def valid_data() -> dict:
    return {
        "campaign_id": "c1",
        "title": "AI clips",
        "objective": "Explain automation",
        "allowed_video_ids": ["video-1"],
        "rights_confirmed": True,
    }


def test_brief_parses_explicit_target_runtime_policy() -> None:
    brief = CampaignBrief.from_dict(valid_data())
    assert brief.allowed_video_ids == ["video-1"]
    assert brief.source_channel_ids == []
    assert brief.to_dict()["campaign_id"] == "c1"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"region_code": "USA"}, "region_code"),
        ({"min_clip_seconds": 0}, "min_clip_seconds"),
        ({"max_clip_seconds": 181}, "max_clip_seconds"),
        ({"min_clip_seconds": 45, "max_clip_seconds": 20}, "less than"),
        ({"allowed_video_ids": []}, "explicit target"),
        ({"watermark_url": "http://example.test/logo.png"}, "https"),
        (
            {
                "allowed_video_ids": ["video-1"],
                "source_media_urls": {"video-2": "https://example.test/video.mp4"},
            },
            "source_media_urls",
        ),
        (
            {
                "source_media_urls": {"video-1": "http://example.test/video.mp4"},
            },
            "https",
        ),
    ],
)
def test_brief_rejects_invalid_runtime_policy_values(patch: dict, message: str) -> None:
    with pytest.raises(BriefValidationError, match=message):
        CampaignBrief.from_dict(valid_data() | patch)


@pytest.mark.parametrize(
    "field",
    [
        "keywords",
        "negative_keywords",
        "required_phrases",
        "clip_count",
        "source_limit",
        "max_clips_per_source",
        "published_after",
        "production",
        "diversity",
        "hooks",
        "editorial",
    ],
)
def test_brief_rejects_obsolete_editorial_discovery_and_quota_fields(field: str) -> None:
    value: object = ["legacy"]
    if field in {"clip_count", "source_limit", "max_clips_per_source"}:
        value = 1
    elif field in {"production", "diversity", "hooks", "editorial"}:
        value = {}
    elif field == "published_after":
        value = "2026-01-01T00:00:00Z"
    with pytest.raises(BriefValidationError, match="obsolete editorial/discovery"):
        CampaignBrief.from_dict(valid_data() | {field: value})


def test_brief_rejects_bad_root_missing_fields_and_bad_lists() -> None:
    with pytest.raises(BriefValidationError, match="root"):
        CampaignBrief.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(BriefValidationError, match="missing required"):
        CampaignBrief.from_dict({})
    with pytest.raises(BriefValidationError, match="list of strings"):
        CampaignBrief.from_dict(valid_data() | {"allowed_video_ids": "video-1"})


def test_transcript_segment_validation() -> None:
    word = TranscriptWord(1.1, 1.5, "hello")
    assert word.duration == pytest.approx(0.4)
    assert word.to_dict()["text"] == "hello"
    segment = TranscriptSegment(1, 2.5, "hello", (word,))
    assert segment.duration == 1.5
    assert segment.to_dict()["text"] == "hello"
    with pytest.raises(ValueError, match="timestamps"):
        TranscriptSegment(2, 1, "bad")
    with pytest.raises(ValueError, match="empty"):
        TranscriptSegment(0, 1, " ")
    with pytest.raises(ValueError, match="word timestamps"):
        TranscriptWord(2, 1, "bad")
    with pytest.raises(ValueError, match="word text"):
        TranscriptWord(0, 1, " ")


def test_optional_lists_none_become_empty_with_explicit_target_preserved() -> None:
    data = valid_data() | {"source_channel_ids": None, "required_hashtags": None}
    brief = CampaignBrief.from_dict(data)
    assert brief.source_channel_ids == []
    assert brief.required_hashtags == []
    assert brief.allowed_video_ids == ["video-1"]


def test_source_media_urls_require_explicit_target_and_serialize() -> None:
    brief = CampaignBrief.from_dict(
        valid_data()
        | {
            "source_media_urls": {"video-1": "https://example.test/video.mp4"},
            "required_hashtags": ["#campaign"],
            "posting_requirements": ["public account"],
        }
    )
    payload = brief.to_dict()
    assert payload["source_media_urls"] == {"video-1": "https://example.test/video.mp4"}
    assert payload["required_hashtags"] == ["#campaign"]
    assert payload["posting_requirements"] == ["public account"]


def test_structured_acceptance_policy_parses_and_serializes() -> None:
    brief = CampaignBrief.from_dict(
        valid_data()
        | {
            "acceptance_policy": {
                "source_segments": {
                    "allow": ["editorial_content"],
                    "forbid": ["advertisement", "sponsor_read"],
                    "unknown": "escalate",
                    "safety_buffer_seconds": 0.25,
                },
                "branding": {
                    "supplied_campaign_assets_allowed": True,
                    "foreign_logos": "forbid",
                },
                "generated_media": {"ai_generated_source_video": "forbid"},
                "portrayal": {"negative_creator_portrayal": "forbid"},
                "language": {"on_screen_text": "en"},
                "editorial": {
                    "require_standalone_context": True,
                    "require_resolved_ending": True,
                    "minimum_boundary_confidence": 0.8,
                },
            }
        }
    )
    assert brief.acceptance_policy.enabled is True
    assert brief.acceptance_policy.source_segments.forbid == (
        "advertisement",
        "sponsor_read",
    )
    assert brief.acceptance_policy.branding.foreign_logos == "forbid"
    assert brief.to_dict()["acceptance_policy"]["editorial"][
        "minimum_boundary_confidence"
    ] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "policy",
    [
        "not-an-object",
        {"unrepresentable_hard_rule": "silently pass"},
        {"source_segments": "not-an-object"},
        {"source_segments": {"unsupported": True}},
        {"source_segments": {"allow": "editorial_content"}},
        {"source_segments": {"forbid": ["made_up_hazard"]}},
        {"source_segments": {"allow": ["promo"], "forbid": ["promo"]}},
        {"source_segments": {"safety_buffer_seconds": 10}},
        {"branding": "not-an-object"},
        {"branding": {"unsupported": True}},
        {"branding": {"foreign_logos": "maybe"}},
        {"branding": {"minimum_confidence": 2.0}},
        {"editorial": "not-an-object"},
        {"editorial": {"unsupported": True}},
        {"editorial": {"minimum_boundary_confidence": 2.0}},
        {"generated_media": "not-an-object"},
        {"generated_media": {"unsupported": True}},
        {"generated_media": {"ai_generated_source_video": "maybe"}},
        {"portrayal": "not-an-object"},
        {"portrayal": {"unsupported": True}},
        {"portrayal": {"negative_creator_portrayal": "maybe"}},
        {"language": "not-an-object"},
        {"language": {"unsupported": True}},
        {"language": {"on_screen_text": "english"}},
    ],
)
def test_acceptance_policy_rejects_unknown_or_unsafe_rules(policy: object) -> None:
    with pytest.raises(BriefValidationError):
        CampaignBrief.from_dict(valid_data() | {"acceptance_policy": policy})
