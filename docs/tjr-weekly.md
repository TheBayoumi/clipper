# TJR Weekly HD campaign

Source: https://reachclipping.com/TJR (verified 2026-09-25).
Branch: `feat/tjr-weekly-hd-actions`. Dedicated Double Coverage accounts such as
`@DCOvertimeClips` are **not** automatically eligible for this trading campaign.

## GitHub Actions

The [TJR Weekly HD clipping](../.github/workflows/tjr-weekly-hd.yml) workflow
runs tests, source-allowlist checks, and a real source render on pushes to this
branch. It can also be manually dispatched after GitHub registers the workflow
on the default branch. The branch-specific push trigger needs no merge.

It starts with a verified, official TJR YouTube source video (June 2026). To
change videos, verify the new upload belongs to the authorized source channel,
then update `campaigns/reach-tjr-weekly.yaml`. Do not add third-party reposts or
unverified campaign assets. The manifest QA fails when channel metadata differs.

Outputs are 1080x1920 H.264/AAC MP4, CRF 18, x264 slow, 30 fps with burned
English captions, plus SRT, source manifest, thumbnails and a strict ffprobe
and full decode QA report. 1080x1920 **output dimensions do not guarantee**
1080p source fidelity: inspect the downloaded original before publishing.
Rendering the same footage at 60 fps does not create genuine new source frames.

Artifacts are available for 14 days on the workflow run. Failure artifacts
may be incomplete. The runner needs access to the original YouTube media;
Google/YouTube can block GitHub-hosted IPs. Setting a `YOUTUBE_API_KEY`
repository secret enables API metadata discovery but does not guarantee
media downloads.

## Human acceptance before publishing

Review every clip for TJR on screen, accurate captions and timing, intact
meaning, no preexisting or added logo, no synthetic footage or voice, proper
framing, and appropriate handling of financial claims. Confirm Whop campaign
budget, eligibility, and your account's target-country audience first.
Use `#TJR`, publish yourself to an eligible platform, and submit the public
video URL to Whop within 30 minutes. The pipeline does not automate
publishing, audience generation, engagement or campaign submission.
