import json
from pathlib import Path

import pytest

from clipper.brief import load_brief, load_explicit_targets
from clipper.models import BriefValidationError

DATA = {
    "campaign_id": "c1",
    "title": "Title",
    "objective": "Goal",
    "targets": {
        "mode": "explicit",
        "videos": [
            {
                "video_id": "abc",
                "url": "https://www.youtube.com/watch?v=abc",
                "channel_id": "UC123",
            }
        ],
    },
    "rights": {"confirmed": True, "authorized_channels": ["UC123"]},
}


def test_load_json_and_yaml(tmp_path: Path) -> None:
    json_path = tmp_path / "brief.json"
    json_path.write_text(json.dumps(DATA), encoding="utf-8")
    assert load_brief(json_path).campaign_id == "c1"

    yaml_path = tmp_path / "brief.yaml"
    yaml_path.write_text(
        "campaign_id: c1\n"
        "title: Title\n"
        "objective: Goal\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: abc\n"
        "      url: https://www.youtube.com/watch?v=abc\n"
        "      channel_id: UC123\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC123]\n",
        encoding="utf-8",
    )
    assert load_brief(yaml_path).title == "Title"


def test_load_semantic_brief_has_no_keyword_or_quota_state(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text(
        "campaign_id: c1\n"
        "title: General podcast\n"
        "objective: Find the strongest self-contained moments semantically.\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: v\n"
        "      url: https://www.youtube.com/watch?v=v\n"
        "      channel_id: UC_AUTHORIZED\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC_AUTHORIZED]\n",
        encoding="utf-8",
    )
    brief = load_brief(path)
    assert brief.campaign_id == "c1"
    assert brief.objective == "Find the strongest self-contained moments semantically."
    assert not hasattr(brief, "keywords")
    assert not hasattr(brief, "clip_count")


def test_load_explicit_target_brief_without_output_quotas(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text(
        "campaign_id: c1\n"
        "title: Explicit campaign\n"
        "objective: Find every independently worthwhile moment.\n"
        "language: en\n"
        "region_code: US\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: v1\n"
        "      url: https://www.youtube.com/watch?v=v1\n"
        "      channel_id: UC_AUTHORIZED\n"
        "      media_url: https://media.example.test/v1.mkv\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC_AUTHORIZED]\n"
        "content_constraints:\n"
        "  min_clip_seconds: 20\n"
        "  max_clip_seconds: 45\n"
        "acceptance_policy:\n"
        "  generated_media:\n"
        "    synthetic_visuals: forbid\n",
        encoding="utf-8",
    )

    brief = load_brief(path)
    targets = load_explicit_targets(path)

    assert brief.allowed_video_ids == ["v1"]
    assert brief.source_channel_ids == []
    assert brief.source_media_urls == {"v1": "https://media.example.test/v1.mkv"}
    assert brief.rights_confirmed is True
    assert brief.min_clip_seconds == 20
    assert brief.max_clip_seconds == 45
    assert brief.acceptance_policy.ai_generated_source_video == "forbid"
    assert len(targets) == 1
    assert targets[0].video_id == "v1"
    assert targets[0].url == "https://www.youtube.com/watch?v=v1"
    assert targets[0].channel_id == "UC_AUTHORIZED"
    assert targets[0].media_url == "https://media.example.test/v1.mkv"


def test_campaign_watermark_is_rejected_when_campaign_assets_are_forbidden(
    tmp_path: Path,
) -> None:
    path = tmp_path / "watermark-forbidden.yaml"
    path.write_text(
        "campaign_id: c1\n"
        "title: Explicit campaign\n"
        "objective: Find worthwhile moments.\n"
        "watermark_url: https://assets.example.test/watermark.png\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: v1\n"
        "      url: https://www.youtube.com/watch?v=v1\n"
        "      channel_id: UC_AUTHORIZED\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC_AUTHORIZED]\n"
        "acceptance_policy:\n"
        "  enabled: true\n"
        "  branding:\n"
        "    supplied_campaign_assets_allowed: false\n",
        encoding="utf-8",
    )

    with pytest.raises(BriefValidationError, match="watermark_url is prohibited"):
        load_brief(path)


def test_explicit_target_channel_must_be_authorized(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text(
        "campaign_id: c1\n"
        "title: Explicit campaign\n"
        "objective: Find worthwhile moments.\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: v1\n"
        "      url: https://www.youtube.com/watch?v=v1\n"
        "      channel_id: UC_WRONG\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC_AUTHORIZED]\n",
        encoding="utf-8",
    )

    with pytest.raises(BriefValidationError, match=r"outside rights\.authorized_channels"):
        load_explicit_targets(path)
    with pytest.raises(BriefValidationError, match=r"outside rights\.authorized_channels"):
        load_brief(path)


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
def test_explicit_targets_reject_obsolete_editorial_discovery_and_quota_fields(
    tmp_path: Path, field: str
) -> None:
    data = dict(DATA)
    data[field] = {} if field in {"production", "diversity", "hooks", "editorial"} else 1
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BriefValidationError, match="legacy editorial/discovery"):
        load_brief(path)


def test_brief_normalization_contains_no_campaign_specific_compatibility_profile() -> None:
    source = Path("src/clipper/brief.py").read_text(encoding="utf-8")

    assert "historical Double Coverage" not in source
    assert "_inject_legacy_cache_compatibility" not in source
    assert '"candidate_pool_size": 36' not in source
    assert '"concept_count": 10' not in source
    assert '"variants_per_concept": 3' not in source
    assert '"final_render_budget": 6' not in source
    assert '"minimum_distinct_finalist_concepts": 3' not in source


def test_explicit_target_placeholders_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text(
        "campaign_id: c1\n"
        "title: Explicit campaign\n"
        "objective: Find every worthwhile moment.\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: REPLACE_WITH_AUTHORIZED_VIDEO_ID\n"
        "      url: https://www.youtube.com/watch?v=REPLACE_WITH_AUTHORIZED_VIDEO_ID\n"
        "      channel_id: UC_REPLACE_WITH_AUTHORIZED_CHANNEL_ID\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC_REPLACE_WITH_AUTHORIZED_CHANNEL_ID]\n",
        encoding="utf-8",
    )

    with pytest.raises(BriefValidationError, match="example placeholder"):
        load_brief(path)


def test_load_unknown_extension_falls_back_to_json_or_yaml(tmp_path: Path) -> None:
    json_path = tmp_path / "brief.txt"
    json_path.write_text(json.dumps(DATA), encoding="utf-8")
    assert load_brief(json_path).objective == "Goal"

    yaml_path = tmp_path / "brief.conf"
    yaml_path.write_text(
        "campaign_id: c1\n"
        "title: T\n"
        "objective: G\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: v\n"
        "      url: https://www.youtube.com/watch?v=v\n"
        "      channel_id: UC1\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC1]\n",
        encoding="utf-8",
    )
    assert load_brief(yaml_path).campaign_id == "c1"


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_brief("does-not-exist.json")


def test_bad_yaml_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- an\n- object\n", encoding="utf-8")
    with pytest.raises(BriefValidationError, match="root"):
        load_brief(bad)


def test_nested_template_source_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "brief.yaml"
    path.write_text(
        "campaign_id: c1\n"
        "title: T\n"
        "objective: G\n"
        "targets:\n"
        "  mode: explicit\n"
        "  videos:\n"
        "    - video_id: v\n"
        "      url: https://www.youtube.com/watch?v=v\n"
        "      channel_id: UC_REPLACE_WITH_AUTHORIZED_CHANNEL_ID\n"
        "rights:\n"
        "  confirmed: true\n"
        "  authorized_channels: [UC_REPLACE_WITH_AUTHORIZED_CHANNEL_ID]\n",
        encoding="utf-8",
    )
    with pytest.raises(BriefValidationError, match="example placeholder"):
        load_brief(path)
