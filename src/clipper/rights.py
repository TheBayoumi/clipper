from __future__ import annotations

from .models import CampaignBrief, VideoCandidate


class RightsError(PermissionError):
    """Raised when a source is outside the campaign's authorized source set."""


def assert_campaign_authorized(brief: CampaignBrief) -> None:
    if not brief.rights_confirmed:
        raise RightsError(
            "rights_confirmed must be true after verifying the Whop campaign "
            "permits clipping these sources"
        )


def assert_video_allowed(brief: CampaignBrief, video: VideoCandidate) -> None:
    assert_campaign_authorized(brief)
    if brief.allowed_video_ids and video.video_id in brief.allowed_video_ids:
        return
    if brief.source_channel_ids and video.channel_id in brief.source_channel_ids:
        return
    raise RightsError(
        f"video {video.video_id} from channel {video.channel_id} is outside the brief allow-list"
    )
