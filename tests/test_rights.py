import pytest

from clipper.models import CampaignBrief, VideoCandidate
from clipper.rights import RightsError, assert_campaign_authorized, assert_video_allowed


def brief(**overrides) -> CampaignBrief:
    data = {
        "campaign_id": "c",
        "title": "t",
        "objective": "o",
        "keywords": ["k"],
        "source_channel_ids": ["UC1"],
        "rights_confirmed": True,
    }
    data.update(overrides)
    return CampaignBrief.from_dict(data)


def video(video_id="v1", channel_id="UC1") -> VideoCandidate:
    return VideoCandidate(video_id, "title", channel_id, "channel", f"https://youtu.be/{video_id}")


def test_authorized_channel_and_video_allowlist() -> None:
    assert_video_allowed(brief(), video())
    assert_video_allowed(brief(allowed_video_ids=["v2"]), video("v2", "other"))


def test_rights_confirmation_and_allowlist_are_enforced() -> None:
    with pytest.raises(RightsError, match="rights_confirmed"):
        assert_campaign_authorized(brief(rights_confirmed=False))
    with pytest.raises(RightsError, match="outside"):
        assert_video_allowed(brief(), video(channel_id="other"))
