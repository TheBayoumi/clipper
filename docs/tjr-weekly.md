# Reach TJR — YouTube-only production

Official campaign (checked 2026-09-26): https://reachclipping.com/TJR
Official sources: https://www.youtube.com/@TJRTrades and
https://www.youtube.com/@TRichesTrades

For this production, ONLY these two channels are accepted. Although Reach
also permits official Kick/Instagram/TikTok content featuring TJR, the older
Kick clips are NOT substitutes for this user's requested YouTube sources.

## Official campaign rules

- Source must actually feature TJR and not portray him negatively.
- Choose funny moments, memorable quotes, or genuine TJR reactions.
- Only TikTok, Instagram Reels and YouTube Shorts are accepted.
- At least 50% of the audience must be in USA, CA, AU, UK or NZ.
- Include #TJR in the published post's caption.
- No logos of ANY kind (including source overlays), AI-generated videos, or manipulated engagement/reposts.
- Current public campaign rate is $1.50 per 1,000 views on TikTok, Instagram and YouTube;
  $15 minimum and $500 maximum per approved submission. Budget is not verified.
- Accounts with a substantially different niche may be rejected; Double
  Coverage-only accounts should stay dedicated to that separate campaign.
- Minimum 5,000 views AND Reach approval before a payout; connect social
  accounts to Whop, post, then submit the public URL within 30 minutes.
- Verify live Whop budget and account eligibility before publication.
  The campaign page currently exposes budget placeholders, not a verified
  available-funds balance.

## Strictly verified direct YouTube workflow

On an approved branch push, GitHub Actions first runs format, mypy, tests
and synthetic encoding checks. When those pass, youtube_preview reads the
real official YouTube RSS upload feeds for BOTH allowlisted channel IDs,
sorts candidates by published time, and independently verifies each video's
YouTube metadata owner ID and video ID before attempting a real HD download.
It NEVER falls back to generic search, third-party reposts or Kick.

For the newest playable long-form video, it takes up to fourteen minutes of
real footage, verifies HD original dimensions, transcribes real English
speech, selects two non-overlapping 20–42-second moments, and renders true
1080×1920 H.264/AAC captioned review drafts. Both files undergo ffprobe
technical QA and a full media decode. The published source URL, owner,
publication date, clip timestamps, original hash, SRTs and preview frames
are included in the GitHub Actions artifact.

Human review MUST still verify that TJR is visible, the lines and captions
are accurate, no forbidden source logos are present, the original context
is intact, and the creative hook is compelling. The workflow does NOT
publish anything to TikTok or submit any clips to Whop.

## YouTube access failure and exact-source fallback

If YouTube blocks GitHub-hosted IP addresses, the job fails with precise
source-acquisition-errors.json diagnostics and a sanitized
`official-source-candidates.json` containing real video links discovered
from the two official channel feeds. Live YouTube RSS was observed returning
a channel ID without its `UC` prefix; this is accepted only when the exact
remaining ID matches an allowlisted channel. Real media download is still
separately verified and cannot be bypassed by RSS metadata alone. A passing synthetic render or an
unrelated Kick MP4 is never reported as a successful YouTube result.

The separate, manual render workflow remains available for an independently
obtained authorized original from the same official YouTube source. That
source must be unchanged, >=720p, and uploaded to a Google Drive viewer URL;
its URL and exact SHA-256 belong in repository Actions secrets
TJR_SOURCE_MEDIA_URL and TJR_SOURCE_MEDIA_SHA256. The existing manual path
is pinned to YouTube ID 8PYgFVB0GHE and is NOT presumed to be newest.
No browser cookies or personal profiles are requested or committed.

## Dated YouTube acquisition audit — 2026-09-26

GitHub Actions [run 36205868637](https://github.com/TheBayoumi/clipper/actions/runs/36205868637)
passed all code, policy and synthetic video tests and obtained 20 candidate
video IDs directly from the official two YouTube channel feeds. The archive
`tjr-real-youtube-hd-36205868637-1` includes
`official-source-candidates.json` and `source-acquisition-errors.json`.
This is verified channel *discovery*, not independently verified playback
metadata, visual content, or a completed edit.

Three feed-listed candidate videos from September 25, 2026 (UTC):
- @TJRTrades: https://www.youtube.com/watch?v=5JR-dmmcpfw
  (22:04:46 UTC, feed gave no usable title; duration/content not verified)
- @TRichesTrades: https://www.youtube.com/watch?v=p2LU37eat70
  ("Live Day Trading Making $18,350", 16:01:53 UTC; duration/content not verified)
- @TRichesTrades: https://www.youtube.com/watch?v=D9J3-dqV6JI
  ("TJR Reacts to the TJR and Aiden videos..", 13:54:41 UTC;
  duration/content not verified)

The actual media-acquisition job FAILED because YouTube requires browser
sign-in to confirm the GitHub runner is not a bot for both attempted videos,
even with Chrome impersonation. There were **zero YouTube MP4s** in that
artifact. Never substitute successful synthetic clips, unaudited sources
or previous Kick clips for genuine approved-channel YouTube footage.

To produce a new draft while direct YouTube downloads remain blocked,
independently obtain a genuine original from the exact official YouTube URL,
verify its owner and that TJR appears, then stage its unchanged HD MP4
privately with a matching SHA-256. **The manual fallback now accepts a `source_video_id` workflow input instead
of automatically choosing the old `8PYgFVB0GHE` ID.** It checks the selected
ID against live official channel feeds, pins the corresponding verified
channel and YouTube watch URL, and hashes the independently staged original.
This does NOT verify that a user-supplied MP4 visually matches the URL;
source authenticity still requires explicit human review. No live Whop funds, account eligibility,
or human editorial acceptance has been verified.

## GitHub-only optional authenticated access

On 2026-09-26, the latest GitHub-hosted public YouTube download attempt failed
with a YouTube bot-confirmation challenge for actual feed-listed approved
videos, even after attempting multiple player clients and a local PO-token
provider. A one-time diagnostic also found no independently verified playable
source for the same official YouTube ID on three public Piped endpoints and
one public Invidious endpoint. Those are *transport checks*, not new sources.

For an authorized dedicated YouTube viewing account, add an encrypted GitHub
Actions secret named `TJR_YOUTUBE_COOKIES_B64` containing a base64-encoded
Netscape-format cookie export from that dedicated account. Do NOT paste
cookies into GitHub commits, action inputs, logs, messages, or this chat.
Use a separate viewing account rather than primary personal accounts;
YouTube session cookies are sensitive credentials and can expire or be revoked.
The workflow creates a mode-600 temporary cookie jar on its ephemeral runner,
passes it only to yt-dlp, and deletes it at the end of the attempt. Neither
cookies nor the original source media are included in uploaded artifacts.
Providing cookies still does **not** guarantee playback from a data-center IP.

When the workflow is registered on the default branch, use `Actions → TJR
Weekly HD clipping → Run workflow`, select `youtube_direct` (no Drive copy)
for the cookie-based YouTube-only path, or `verified_mirror` for the
independently obtained approved-channel original with hash-pinned Google Drive
staging and explicit budget/source verification. The former is review-only;
no manual publication or Whop submission takes place.
