# TJR Weekly HD campaign

Official source and campaign requirements: https://reachclipping.com/TJR (reviewed 2026-09-25).
Implementation: `feat/tjr-weekly-hd-actions`. Review: PR #7.
A Double Coverage-specific account such as `@DCOvertimeClips` is **not**
automatically eligible for this trading-content campaign.

## GitHub Actions behavior

[The TJR Weekly HD workflow](../.github/workflows/tjr-weekly-hd.yml) runs
Ruff, strict mypy, pytest (95% minimum coverage), a campaign policy check and
a synthetic FFmpeg render canary on branch pushes and pull requests.

**Real footage is never fetched or rendered on a branch push.** GitHub-hosted
IP addresses triggered YouTube bot verification when fetching TJR's public
video, so the real run now uses an explicitly approved mirror of that same
original. The original video ID is pinned as `8PYgFVB0GHE` on the official
TJR YouTube channel. Do not replace it with a third-party repost.

## One-time source setup for a real campaign render

1. Independently obtain the authentic source video from the campaign-permitted
   TJR source. Review it to ensure that TJR appears, there are no forbidden
   preexisting logos, the original is at least 720p and the clip context is sound.
2. Upload that **unaltered original MP4** to a Google Drive file with Viewer
   access for anyone who has the link. Do not upload it to a public GitHub repo.
   The required URL shape is `https://drive.google.com/file/d/<ID>/view`.
3. Hash that exact MP4 before upload using `sha256sum tjr-original.mp4`
   (PowerShell: `Get-FileHash .\tjr-original.mp4 -Algorithm SHA256`).
   Do not replace or re-encode the file after obtaining its digest.
4. Set these repository **Actions secrets**, not public variables or commits:
   `TJR_SOURCE_MEDIA_URL` (the Drive Viewer URL) and
   `TJR_SOURCE_MEDIA_SHA256` (the exact 64-character digest).
   This action requires a repository administrator. Do not share any account
   cookies or upload your personal browser profile.
5. Verify the live remaining TJR campaign budget and your account eligibility
   in Whop. Public cached listings do not agree on the remaining funds.

Once the PR is merged and GitHub registers the workflow on the default branch,
go to **Actions → TJR Weekly HD clipping → Run workflow**. Select the branch
and explicitly check both **budget_confirmed** and **source_verified**.
The workflow validates the staged source link, downloads the original,
compares SHA-256 byte-for-byte against the approved file, verifies at least
720p source detail, renders and performs full output decode plus ffprobe QA.
The Drive URL is not printed and normalized briefs are not uploaded as public
artifacts. The source copy remains only in the ephemeral runner.

## Produced artifacts and checks

Each accepted result is a 1080 × 1920 H.264/AAC 30 fps MP4, x264 slow/CRF 18,
with English captions, no added logos and sound normalization. The ZIP
contains candidate MP4s, matching SRT files, thumbnails, a timestamped source
manifest and `tjr-qa-report.json`. Artifacts expire after 14 days.

**Technical acceptance is not campaign approval.** Before posting, manually
inspect every rendered video for actual TJR appearance, accurate captions,
correct context and framing, no embedded logos or synthetic footage, and
appropriate treatment of financial claims. Use a TJR/trading-relevant account,
verify that at least 50% of your audience is from USA/Canada/UK/Australia/NZ,
include `#TJR`, and submit the published URL to Whop within 30 minutes.
No TikTok publication, engagement or Whop submission is automated.
