import pytest

from clipper.models import BriefValidationError, CampaignBrief, TranscriptSegment, TranscriptWord


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


def test_optional_list_none_becomes_empty_but_requires_other_allowlist() -> None:
    data = valid_data() | {"source_channel_ids": None, "allowed_video_ids": ["v"]}
    brief = CampaignBrief.from_dict(data)
    assert brief.source_channel_ids == []


def test_v8_nested_production_config_parses_and_serializes() -> None:
    brief = CampaignBrief.from_dict(
        valid_data()
        | {
            "production": {
                "candidate_pool_size": 40,
                "concept_count": 9,
                "variants_per_concept": 3,
                "final_render_budget": 7,
            },
            "diversity": {"semantic_similarity_threshold": 0.8, "max_concepts_per_topic": 1},
            "hooks": {"enabled": ["direct", "number"]},
            "editorial": {
                "platform": "instagram_reels",
                "max_punch_ins_per_clip": 1,
                "semantic_endings": True,
                "post_speech_tail_seconds": 0.3,
                "caption_max_lines": 2,
            },
        }
    )
    assert brief.production.candidate_pool_size == 40
    assert brief.production.final_render_budget == 7
    assert brief.diversity.semantic_similarity_threshold == 0.8
    assert brief.hooks.enabled == ("direct", "number")
    assert brief.editorial.platform == "instagram_reels"
    assert brief.to_dict()["production"]["concept_count"] == 9


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"production": {"candidate_pool_size": 2}}, "candidate_pool_size"),
        ({"production": {"concept_count": 0}}, "concept_count"),
        ({"production": {"variants_per_concept": 7}}, "variants_per_concept"),
        ({"production": {"final_render_budget": 25}}, "final_render_budget"),
        ({"diversity": {"semantic_similarity_threshold": 0.1}}, "semantic_similarity_threshold"),
        ({"diversity": {"max_concepts_per_topic": 0}}, "max_concepts_per_topic"),
        ({"hooks": {"enabled": ["fake"]}}, "unsupported hook"),
        ({"editorial": {"platform": "snapchat"}}, "platform"),
        ({"editorial": {"max_punch_ins_per_clip": 4}}, "max_punch"),
        ({"editorial": {"post_speech_tail_seconds": 2}}, "post_speech"),
        ({"editorial": {"caption_max_lines": 3}}, "caption_max_lines"),
    ],
)
def test_v8_nested_config_rejects_invalid_values(patch: dict, message: str) -> None:
    with pytest.raises(BriefValidationError, match=message):
        CampaignBrief.from_dict(valid_data() | patch)


def test_production_distinct_finalist_concept_validation() -> None:
    from clipper.models import BriefValidationError, ProductionConfig

    config = ProductionConfig.from_dict(
        {
            "candidate_pool_size": 36,
            "concept_count": 10,
            "variants_per_concept": 3,
            "final_render_budget": 6,
            "minimum_distinct_finalist_concepts": 3,
        }
    )
    assert config.minimum_distinct_finalist_concepts == 3
    with pytest.raises(BriefValidationError, match="final_render_budget"):
        ProductionConfig.from_dict(
            {
                "candidate_pool_size": 36,
                "concept_count": 10,
                "variants_per_concept": 3,
                "final_render_budget": 2,
                "minimum_distinct_finalist_concepts": 3,
            }
        )
    with pytest.raises(BriefValidationError, match="concept_count"):
        ProductionConfig.from_dict(
            {
                "candidate_pool_size": 36,
                "concept_count": 2,
                "variants_per_concept": 3,
                "final_render_budget": 6,
                "minimum_distinct_finalist_concepts": 3,
            }
        )


def test_v11_structured_acceptance_policy_parses_and_serializes() -> None:
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
def test_v11_acceptance_policy_rejects_unknown_or_unsafe_rules(policy: object) -> None:
    with pytest.raises(BriefValidationError):
        CampaignBrief.from_dict(valid_data() | {"acceptance_policy": policy})
