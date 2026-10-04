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

On a matching branch push or PR, GitHub Actions runs code, policy and synthetic
encoding checks only. Real media acquisition is explicitly manual: choose
`modal_direct` or `youtube_direct`, confirm the campaign budget, and select
`clip_limit` between 1 and 20. Manual runs default to `validate_only`, which
runs tests without sourcing media. An invalid production dispatch fails input
validation rather than returning a misleading green run with skipped jobs.
The Windows/macOS alternate YouTube workers start only if the primary Linux
YouTube worker fails, avoiding three duplicate rendering batches. No
real-generation job starts on ordinary pushes. The verified_mirror path
first downloads its independently approved Drive original and verifies its SHA-256,
video ID, approved channel, HD resolution and duration, then passes the unchanged
original to the same TJR editorial selector and Style B2 renderer. If verification
fails, it does not fall back to a different YouTube video.
When manually enabled, youtube_preview reads the
real official YouTube RSS upload feeds for BOTH allowlisted channel IDs,
sorts candidates by published time, and independently verifies each video's
YouTube metadata owner ID and video ID before attempting a real HD download.
It NEVER falls back to generic search, third-party reposts or Kick.

For a playable eligible long-form video, the workflow processes up to fourteen
minutes of original footage, checks source dimensions, transcribes English
speech, and ranks distinct 20–42-second moments using transcript sentence/pause
boundaries and the evidence-labeled
editorial rubric below. The configured editorial batch limit (12 per source by default), not a
two-clip cap, determines the maximum number of draft renders. Each selected
moment gets a distinct source-grounded headline. If its topic template repeats,
a short verbatim source excerpt is used; clips without a unique safe hook are
rejected. Style B2 displays each hook for the whole clip, with word-aligned
captions and a measured portrait-safe text box. Speech-boundary detection
is heuristic, so the first/last frames and narrative payoff still need review. Draft MP4s
receive 1080×1920 H.264/AAC encoding, ffprobe QA and full decode checks.
The source URL, owner, upload date, timestamps, SHA-256, captions, frame
previews and editorial audit accompany successful runs.

Human review MUST still verify that TJR is visible, the lines and captions
are accurate, no forbidden source logos are present, the original context
is intact, and the creative hook is compelling. The workflow does NOT
publish anything to TikTok or submit any clips to Whop.



## Editorial screening and scoring — `tjr-editorial-v1`

The transcript-first screen ranks **review drafts**, not clips approved for
publication. Its versioned, auditable weights sum to 100:

| Criterion | Weight | Candidate-time evidence |
| --- | ---: | --- |
| Opening | 25% | Reaction/question/action/specific stake in first 2 seconds, using word timestamps where available |
| Story | 25% | Lexical setup, tension and closing-payoff cues; penalize unfinished boundaries |
| Emotion | 10% | Reaction and emphasis *word* cues; tone and authenticity remain unverified |
| Visuals | 15% | No automated score: a reviewer must inspect footage and portrait framing |
| Retention | 25% | Distinct new terms in later thirds and repeated-trigram penalty as transcript proxies |

Each criterion records a 0–5 score **or null** when unassessed, its evidence
and whether it is a transcript proxy or manually verified. Without a visual
review, score coverage is **85/100 weight points**. The provisional
editorial score is `100 × observed_weighted_points / score_coverage`;
it is **not** an assumed 15-point visual score or a virality forecast.
With reviewer-supplied visual evidence, coverage becomes 100.

The **integrity gate** is separate from the numerical score. Unsupported
numeric claims in a headline cause candidate rejection. Other semantic and
contextual claims cannot be proven from keyword matching: their status stays
`unverified` until a reviewer explicitly approves or rejects them with
evidence. An explicit integrity failure rejects the candidate regardless
of its score. Even a passed rubric does not authorize campaign publication;
source rights, on-screen logos, campaign eligibility and final-render review
remain independent requirements.

For every successful real-source run,
`editorial-candidate-audit.json` contains rubric version, weights,
selected candidate timestamps, all criterion scores, evidence and basis,
score coverage, integrity status and rejection reasons. Per-clip QA reports
repeat the selected candidate's rubric evidence. The artifact is retained
by each real-source runner path alongside its MP4/SRT and technical QA.

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

## Exact YouTube source selection

The `youtube_direct` workflow now accepts the same `source_video_id` input
as the verified-mirror route. If a video ID is supplied, it MUST appear
in a current feed from one of the two campaign-listed channels; the
workflow must not silently substitute newer Shorts, Kick VODs or other
trading creators if download fails. Without an explicit ID, it selects
recent long-form videos from those two channels. The preferred recent
feed-listed example is `p2LU37eat70`, but neither its current
availability nor its media download from the runner is guaranteed.

## Fully automatic public YouTube acquisition (no cookie exports)

The default GitHub Actions job runs directly against videos from the two
Reach-listed official YouTube channels. It does **not** require users to
export browser cookies, sign in for each video, or manually upload source
MP4s. On each new runner it installs yt-dlp with its up-to-date JavaScript
solver and the experimental yt-dlp-getpot-wpc v1.1.2 plugin, starts the
runner's Chrome browser under a virtual display, and lets the plugin mint
**fresh per-video guest playback tokens on demand** for mweb and
web_safari. The older local bgutil provider failed to clear YouTube's
runner-level bot confirmation challenge; it is not treated as proof of
successful access.

The runner also attempts an actual Chrome watch-page load of up to two
feed-listed official YouTube videos. If Chrome's original YouTube player
identifies both the exact video and the exact allowlisted channel and
provides an HTTPS GoogleVideo video/audio pair at >=720p, FFmpeg can
acquire only the first 14 minutes of that original. The downloaded
file is re-probed, SHA-256 hashed, and checked by the main pipeline
before ASR and any final MP4 rendering. Signed CDN URLs and browser
session cookies never appear in uploaded artifacts; the sanitized
browser-probe.json records only the real playback status and success
or failure of media acquisition. If Chrome also receives a bot
challenge, automatic guest playback cannot overcome an IP-level block.

Successful acquisition STILL requires actual extracted video metadata
whose owner channel ID matches the exact Reach allowlist, a playable
high-definition original, a full media-decode check and real MP4
artifacts. Both dynamic-browser client variants and selected plain
YouTube clients are attempted when an approved video is accessible.
If YouTube blocks the GitHub-hosted runner IP *before* playback tokens
can help, the job must report that blocker and upload diagnostics, not
claim success or substitute Kick content.

A `TJR_YOUTUBE_COOKIES_B64` repository secret is an OPTIONAL
one-time fallback for users who voluntarily authorize a dedicated
viewer account. The public acquisition flow works without that secret
when YouTube permits anonymous browser playback. No secret is generated,
committed, requested per video, or silently synthesized.

## Optional authenticated-session fallback

On 2026-09-26, the latest GitHub-hosted public YouTube download attempt failed
with a YouTube bot-confirmation challenge for actual feed-listed approved
videos, even after attempting multiple player clients and a local PO-token
provider. A one-time diagnostic also found no independently verified playable
source for the same official YouTube ID on three public Piped endpoints and
one public Invidious endpoint. Those are *transport checks*, not new sources.

For an authorized dedicated YouTube viewing account, set the GitHub
Actions secret named `TJR_YOUTUBE_COOKIES_B64` containing a base64-encoded
Netscape-format cookie export from that dedicated account. Do NOT paste
cookies into GitHub commits, action inputs, logs, messages, or this chat.
Use a separate viewing account rather than primary personal accounts;
YouTube session cookies are sensitive credentials and can expire or be revoked.
The workflow creates a mode-600 temporary cookie jar on its ephemeral runner,
passes it only to yt-dlp, and deletes it at the end of the attempt. Neither
cookies nor the original source media are included in uploaded artifacts.
Providing cookies still does **not** guarantee playback from a data-center IP.

If the workflow is registered on the default branch, use `Actions → TJR
Weekly HD clipping → Run workflow`, select `youtube_direct` (no Drive copy)
for the cookie-based YouTube-only path, or `verified_mirror` for the
independently obtained approved-channel original with hash-pinned Google Drive
staging and explicit budget/source verification. The former is review-only;
no manual publication or Whop submission takes place.


### Production failure recovery and diagnostics

A Modal regional bot challenge on one upload now advances to the next
allowlisted upload rather than aborting the region after two challenges.
The editor first selects windows at normal sentence/0.7 s silence
boundaries; if nothing passes its unchanged editorial gates, it retries
using real word-aligned 0.35 s pauses (flagged for manual boundary review).
If a verified source still has no qualifying story, the production runner
may verify and transcribe one different official upload from the same
approved channel. An explicitly pinned video never switches sources.

Every editorial failure uploads its real transcript, strict/relaxed
candidate counts and rejection audit before source cleanup. Neither an
inaccessible channel nor a source with no qualifying moments counts as
a completed clip. No direct YouTube or unauthorized mirror is silently
substituted for the selected Modal transport.

## Evidence-preserving exchange review and qualification

The current production refinement calls the position-based delivered-span reviewer through the checkpointed review cache. It retains canonical setup and resolution spans through hook generation; excluded speech may identify a missing ending but may not supply the delivered payoff. The earlier quote-based focused reviewer remains available for legacy diagnostics, but is not the production default. New selection evaluates exchange boundaries without literal-headline eligibility. Compatible legacy positive windows remain reusable; legacy negative decisions are reassessed because their rejection could depend on the old headline gate. The source/transcript/proposal/model/bounds compatibility guard remains mandatory.

Editorial screening, ranking, duplicate handling and review-draft gates now live in the installable `clipper.editorial` module. `scripts.tjr_editorial` is a compatibility import only. The campaign workflow still calls the specialized source/reviewer/preview scripts, so this is the first package-consolidation step, not a completed config-only production entrypoint or a factual-accuracy qualification.

A blind source-question stage reads delivered speech without any proposed headline. It drafts semantic answers before constrained serialization, retaining exact source passages for actor/action, relationship, setting/reporting scope and actual/conditional outcomes. The subsequent hook checker records factual support separately from readable, coherent, central-exchange wording. One alternate hook may be attempted; failed hooks do not erase the retained exchange assessment. Model judgments are provisional and require source review.

Review calls persist individually in review-request-cache.json, keyed by source, model revision/hash, llama runtime, exact prompt/payload/schema/token budget, seed/temperature and completion implementation. Completed negative responses are reusable. A changed generation prompt can reuse unchanged boundary and source-question requests. Corrupt or mismatched response proofs are recomputed. Cache hits do not instantiate model weights. Raw requests/responses, token usage and per-call timing are retained for qualification.

The exact-request persistence and raw-trace implementation now lives in `clipper.editorial_request_cache`; `scripts.tjr_semantic_editor.ReviewRequestCache` is a compatibility constructor supplying the existing model completion and stage-fingerprint functions. The reviewer fingerprint includes the package class, so a future cache-logic change invalidates aggregate reviewer decisions without changing the selector stage. Local replay of an actual saved request from the pinned-source [v3 cut-obligation run](https://github.com/TheBayoumi/clipper/actions/runs/37180001369) produced one cache hit, zero model calls and zero model-seconds without loading weights. The current selector implementation still computes the legacy migration fingerprint `ee1ded6a8bfce3eee3855b7685c758d6573845c1b4d59ff0c1aa62a42e832b08`, compatible with the earlier 66-decision artifact's recorded editor-code hash. This is a package-consolidation/cache-preservation step, **not** factual-verifier qualification.

The existing workflow exposes reviewer_model_probe=evidence_qa in validate_only/editorial_preflight mode. It consumes the verified cached full transcript plus the completed factual-control artifact, checks all source passages against the transcript before loading weights, and exercises the actual production reviewer on six exchange controls and twelve frozen factual claims. Four transfer controls supplement the eight previously inspected claims. It performs no source acquisition, transcription, bulk selection or rendering. Every incorrect acceptance, rejection for the wrong factual dimension, omitted payoff, promotion mistake or invalid evidence blocks qualification. Both the report and request cache upload even after failure. The 45-minute limit bounds this diagnostic; production latency and generalization to other podcasts remain unverified.

Qualification is distinct from code CI and media delivery. A green code test does not establish semantic qualification. A passing single-source diagnostic does not establish general podcast quality, viral engagement or publication approval. Production verification requires nonzero MP4 delivery and independently checks canonical v7 span/QA provenance. Existing centered black caption plates, background blur and source-audio fidelity checks remain in place.

### Current editorial investigation

The pinned 4B and 30B GPU qualification in [run 37028628389](https://github.com/TheBayoumi/clipper/actions/runs/37028628389) produced zero contract errors but only 7/18 correct semantic controls each. Neither reviewer is production-qualified. The recorded responses show four separate failure mechanisms:

| Observed response | Consequence | Next discriminating check |
| --- | --- | --- |
| Complete business/fighter exchanges were labeled show intros, despite reasons describing conversation. | Valid clips are rejected before hook generation. | Ask for the host-to-audience introduction or commercial act, then check whether its cited span actually performs that purpose. Keep the six exchange labels frozen. |
| The excluded-payoff case was marked promotional, and its delivered ending was wrongly marked complete. The production path skipped the continuation call when promotion was detected. | One mistaken veto hid an independent boundary error. | The continuation call now runs whenever a delivered ending and excluded continuation exist; repeat the real-model qualification to measure its effect. |
| One short source answer retained the Instagram nonpayment but omitted the explicitly conditional podcast earnings. | A supported conditional headline was rejected. | Compare claim-specific source inspection with the current four-answer compression on unchanged source and claims; require correct actual-versus-conditional distinctions. |
| Quoted live-TV instructions were described as the meeting's actual setting; unclaimed dimensions were also marked uncertain. | False setting claims could pass while true headlines were vetoed. | Test explicit claim presence and reported-speech scope on the existing positive/negative minimal pairs. Evidence positions alone cannot repair this interpretation. |

These are hypotheses for controlled experiments, except for the confirmed continuation control-flow defect. Reuse compatible selector decisions and raw request caches. A changed prompt or model configuration requires fresh inference and must report false approvals, false rejections, the specific failed component, runtime and cache hits. Do not dispatch the full production run until the existing qualification controls pass and the reviewer has been checked on unseen source exchanges. The existing application and media path remain the implementation target.

The follow-up real-model diagnostic [37059525862](https://github.com/TheBayoumi/clipper/actions/runs/37059525862) used the same frozen transcript and controls. At head `12e1ab3`, the 30B reviewer improved from 2/6 to 6/6 correct exchange controls and from 5/12 to 7/12 factual controls, with zero contract errors. The 4B reviewer reached 3/6 exchange and 7/12 factual controls with one contract error. Hosted code CI passed, but the editorial gate correctly failed and no real media was rendered. The new purpose labels and explicit `not_claimed` state are useful diagnostic changes, not a production-qualified factual verifier.

The five remaining 30B factual failures are all false approvals: asserted podcast revenue, a quoted live-TV instruction recast as the meeting setting (two formulations), an unestablished opponent relationship (two formulations). Both models share most of these errors, so model consensus is not a safe gate. The blind source answer also misstates the quoted TV instruction as actual setting, while the claim auditor sometimes marks an explicit opponent role as `not_claimed` even when its reason recognizes the unsupported role. Another wording-only prompt tweak or vote cannot be represented as a fix. The next experiment needs claim-specific, independently sourced attribution/scope evidence, with a separate held-out control set before production integration.

The experimental `clipper.editorial_claims` audit was measured on the same twelve factual controls, separately from the unchanged production gate. The first run [37072349374](https://github.com/TheBayoumi/clipper/actions/runs/37072349374) failed all twelve extraction contracts on both models: model-supplied character offsets were unreliable and some attributions/connectives were dropped. Python now locates exact copied subclaim text and always audits the complete headline, so omitted subclaims cannot silently remove the original assertion. That revised run [37073234761](https://github.com/TheBayoumi/clipper/actions/runs/37073234761) remains a negative result:

| Exact-source model | Existing factual gate | Experimental whole-headline plus subclaims | New contract errors | New semantic errors |
| --- | ---: | ---: | ---: | ---: |
| Pinned 4B | 7/12 | 9/12 | 0 | 3 false rejections |
| 30B-A3B | 7/12 | 6/12 | 2 | 4 false approvals |

The 4B path rejected supported conditional podcast earnings and two supported source summaries. The 30B path still approved invented podcast revenue, converted quoted live-TV instructions into an actual setting, and inferred an opponent/actor role absent from the speech. Its two extraction errors were reported as contract failures, not counted as safe rejections. The 30B warm replay produced identical verdicts with zero model calls; the 4B old-path contract error prevented its replay. Combined old/new qualification used 83 model calls and 288 seconds for 4B, 81 calls and 232 seconds for 30B. This experiment is not a production fix: decomposition plus citations did not establish entailment, and the added inference cost is material. Keep the existing gate unqualified, retain the raw evidence, and do not promote either model or dispatch media production. A new design must address source speech-act/attribution scope directly, and then pass both these controls and held-out exchanges before integration.

The next run [37076517413](https://github.com/TheBayoumi/clipper/actions/runs/37076517413) added twelve **provisional transcript-only** claims from three different exchanges (training advice, UFC career planning, fighter promotion). The fixture is bound to exact video/source/transcript hashes and to three delivered MP4 source-time windows from artifact `36854533220`; it is not audio-verified gold. Labels were withheld from model requests. The run was diagnostic-only and did not acquire media or publish anything. Its separate held-out scores were:

| Model | Existing gate | Experimental whole-headline plus subclaims | Salient failure |
| --- | ---: | ---: | --- |
| Pinned 4B | 8/12; 1 false approval, 3 false rejections | 5/12; 1 contract error, 6 false rejections | It approved advice to spar *more* when the source says too much sparring. |
| 30B-A3B | 10/12; 2 false approvals | 7/12; 5 contract errors | It upgraded a future training reassessment and a hypothetical fighter-cut rule into events that had already happened. |

The 30B warm replay reproduced both old and experimental judgments with zero model calls. The 4B run did not qualify for warm replay because the combined old-path controls still had a contract error. Both models remain unqualified; the zero counted false approvals of the experimental path do not compensate for extraction failures, false rejections, or the small provisional sample. Exact MP4 SHA-256 values for pending human audio review are `77ec8ea92da940aa48e5bd2c46d1baec3a5f7eb3eb2ed62cec5596bd76cc65ea` (clip 05), `d525f3f5ef36029ff4fe2dd79b929ede41081b52a843880d6e33ed55664e2ed6` (clip 07), and `829ff0bd5038fe10865d051703c63a67733b28249fcb55bdc7613777a4d61ff4` (clip 15). An independent local ASR cross-check was completed later, but do not relabel these controls audio-verified without human review. The next verifier should test complete predicate–argument relations and speech-act modality, not free-form substrings or larger models alone. This direction is consistent with [QASemConsistency's predicate–argument QA formulation](https://aclanthology.org/2026.tacl-1.6/), while [claim-decomposition sensitivity](https://aclanthology.org/2024.starsem-1.13/) and [AttributionBench's observed attribution difficulty](https://aclanthology.org/2024.findings-acl.886/) caution against treating a citation or extracted fragment as a truth guarantee. Those papers motivate an experiment; they do not validate this app.

The first blind source-question implementation in `clipper.editorial_qa` was measured by [run 37078726162](https://github.com/TheBayoumi/clipper/actions/runs/37078726162). Its source-answer call did not receive the headline or claimed answer, but it required the model to reproduce exact source words and use one scope enum for event status and attribution. That contract was too brittle:

| Model | Frozen twelve factual controls | Provisional held-out twelve | Held-out false approvals | Held-out contract errors |
| --- | ---: | ---: | ---: | ---: |
| Pinned 4B | 6/12 | 4/12 | 0 | 6 |
| 30B-A3B | 5/12 | 5/12 | 0 | 1 |

The 30B held-out path additionally rejected six supported claims; the 4B path rejected two supported claims and had six held-out contract errors. The 30B warm replay reproduced its judgments with zero model calls. The production gate was not changed. The run demonstrates that blind source QA alone is not sufficient: literal-quote copying fails on paraphrase, and reported speech must be represented separately from whether the described event was actual, planned, conditional, or a quoted instruction. Its extra 229 (4B) and 196 (30B) model calls, including unchanged baseline controls, also make latency a design constraint. A revised contract must keep Python-owned evidence positions, separate event modality from attribution, and be re-measured on both control sets before production integration.

An independent local `faster-whisper` `base.en` pass (model revision `3d3d5dee26484f91867d81cb899cfcf72b96be6c`) decoded the three hash-pinned delivered MP4s on 2026-10-03. Clip 05's audio says Mighty Mouse thinks the speaker spars too much and that he will reassess *in the future*; clip 07 says fighters are independent contractors and **can** be cut even after going six-and-zero **if** boring; clip 15 recalls being told to make a career plan and contrasts fighting with sports the speaker could not play. These are independent ASR cross-checks of the source words, not human listening or a second semantic annotator. The held-out labels therefore remain provisional and must not be promoted to audio-verified gold. No model result should be described as production-qualified on this basis.

The revised source-first contract (Python-owned citations and separate event-status labels) was measured on the same exact source by [run 37095816629](https://github.com/TheBayoumi/clipper/actions/runs/37095816629). Code CI passed, but editorial qualification failed. The unchanged production gate still scored 8/12 (4B) and 10/12 (30B) on provisional held-out claims, with one and two false approvals respectively. The revised **experimental** gate scored:

| Model | Frozen factual controls | Provisional held-out controls | Held-out false approvals | Held-out false rejections | Held-out contract errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned 4B | 4/12 | 3/12 | 0 | 2 | 7 |
| 30B-A3B | 7/12 | 6/12 | 0 | 6 | 0 |

The 30B experimental gate rejected **all six supported held-out headlines**. Its zero false approvals are therefore not evidence of a useful verifier. The 4B run had five contract errors on frozen claims and seven on held-out claims. In the 30B trace, the source citation for Mighty Mouse's advice omitted the nearby unit naming Mighty Mouse, so the comparator called the true attribution uncertain. It treated “calls UFC fighters independent contractors” as a `quoted_instruction` while the source classified it as an actual statement. It also misclassified a hypothetical fighter-cut rule as an actual event and cited a fragment too narrow to establish the conditional. Splitting the enum and owning citation text in Python fixed neither semantic scope nor context selection. The 30B cold run made 204 model calls in 521 model-seconds; its warm replay made zero calls and reproduced the failure. The 4B cold run made 220 calls in 666 model-seconds. Do not promote this experiment, swap models, or launch production under an automated factual-approval claim. The next engineering step is an explicit claim/evidence review contract with separately verified attribution, modality and context; if those cannot be automatically qualified, production approval must be human, while the existing selector, cache and draft renderer remain intact.

The qualification aggregator previously returned `semantic_pass` from the six exchange and twelve frozen factual controls alone, even when requested held-out claims failed. It now requires every requested held-out result from the *unchanged production gate* to be contract-valid and semantically correct; experimental verifier scores remain separate. This closes a pass/fail accounting gap, not the factual-verification gap. Held-out labels are still provisional, so even a future green single-source run would not establish production quality. The context-selection traces also show that a proposal can legitimately be trimmed or shifted before review; an older rendered MP4 is not automatically a good editorial boundary. We will not force overlap with those old drafts merely to make a test pass. This caution is consistent with [DnDScore's finding that decontextualized subclaims need context-aware verification](https://aclanthology.org/2025.emnlp-main.1205/) and [RECV's evidence that model claim verification can fail on harder reasoning](https://aclanthology.org/2025.findings-acl.1059/); neither paper validates Clipper's reviewer.

The editor now retains a `claim_review_packet` in each assessed exchange that reaches headline generation and writes a separate `claim-review-packets.json` artifact, including an explicit empty list when no headlines survive. It binds the proposed headline and reviewed setup/resolution positions to exact source/transcript hashes, preserves all delivered units, and labels excluded before/after speech as context only. A fabricated or mispositioned reviewed quote rejects that candidate as `INVALID_CLAIM_REVIEW_PACKET` without aborting the source. Compatible selector/reviewer cache results remain reusable. `clipper.editorial_review` separately validates an atomic-claim attestation: the reviewed headline must exactly match the draft, every headline word must be covered, a whole-headline check remains mandatory, each cited unit range is checked, Python expands citations to adjacent source units to expose omitted conditions or referents, and reporting-act status is distinct from the embedded event's status. A model-authored record is never production-approved; even a complete self-declared human attestation is recorded as an attestation, not authenticated identity, technical QA or publication approval. This is the auditable review contract needed for a qualified verifier or human sign-off, **not** a demonstrated automatic factual fix. The existing production model gate is still unqualified.

Inspection of the saved [37095816629](https://github.com/TheBayoumi/clipper/actions/runs/37095816629) 30B trace found a specific source-first failure: the positive Mighty Mouse attribution cited only “he says I spar too much” and omitted the nearby named antecedent; other positives failed because the QA contract conflated a speaker's act of reporting with the status of the reported event. The claim-review contract now permits a supported *actual report of a state or opinion* with no embedded event, while still rejecting a bare `no_report`/`not_applicable` assertion. This repairs an ontology/contract inconsistency, **not** the model's ability to decide whether the source actually supports the assertion. No production approval follows from it.

The next diagnostic backend, `clipper.editorial_structured_claims`, measures that distinction end to end without changing production decisions. The model selects numbered headline-word spans and source-unit positions; Python reconstructs exact text, rejects omitted headline words or invalid citations, and separately asks for a source-only reporting/event classification before validating the whole-headline and atomic-claim record. Its source-scope call sees the full selected exchange, not just an isolated pronoun citation. The workflow's `structured_claim` mode measures this candidate on the twelve frozen and twelve provisional held-out factual controls, preserving raw calls and separate error counts; the unchanged production-gate baseline is referenced by exact proof hash rather than rerun. The candidate is diagnostic-only regardless of its score; one single-source test cannot qualify publication or replace human source review.

The first real-model attempt, [37115857543](https://github.com/TheBayoumi/clipper/actions/runs/37115857543), was canceled after its saved checkpoint showed the original route had spent 2,921 model-seconds on 72 inference calls while unnecessarily rerunning the unchanged production baseline. Its uploaded partial artifact contains all twelve new frozen-claim assessments: **0/12 passed**, all with contract errors (headline-word omissions or truncated assessments), plus only two of twelve held-out cases. The word-span response for “Bobby Green paces back and forth during live TV fighter meeting” omitted “back and forth.” A truncated assessment for the podcast-revenue headline also asserted that the source established podcast revenue, although the source only says Instagram drives the podcast. That is a substantive factual error visible in the raw response, independent of the truncation. The optimized replacement [37118706653](https://github.com/TheBayoumi/clipper/actions/runs/37118706653) correctly skipped old baseline inference and reached three frozen claims with three model calls, but each omitted headline words; it was canceled once the earlier checkpoint showed that the unchanged diagnostic contract had already failed all twelve. Both runs preserved partial artifacts. Neither qualifies an automatic reviewer or changes production. Repairing token limits or span coverage alone would not resolve the observed false revenue inference; a new approach must be evaluated for semantic accuracy, not merely valid JSON.

The Clipper package now has a packet-to-review handoff validator: it requires the caller to supply delivered speech independently reconstructed from the pinned transcript, and checks the draft headline, source identity, reviewed setup/resolution citations, and record against that speech. It returns a packet hash for audit, never production approval. This closes a provenance handoff gap; it does not establish that the source semantically supports a claim, authenticate a human reviewer, or integrate a qualified gate into the production selector/reviewer path.

Research check for the next verifier design: [QASemConsistency (TACL 2026)](https://aclanthology.org/2026.tacl-1.6/) represents a text's minimal predicate–argument relations as question/answer pairs, separating a wrong actor, location, cause, or other relation rather than treating a whole sentence or word span as one fact. This is a more appropriate *evaluation unit* for the observed revenue, TV-setting, and opponent-role failures than the current word-span extraction. It is not a turnkey automatic gate: the authors' stronger parser uses a 3B T5-XL model plus a separate model for copular constructions, reports parser-generated relation errors requiring annotation, and evaluates automatic entailment models that remain fallible. Their [published test results](https://aclanthology.org/2026.tacl-1.6.pdf) include balanced accuracies well below 100%, so no reported model score can be imported as Clipper's accuracy. The older [open QASem implementation](https://github.com/kleinay/QASem) is a possible experimental parser, but its documented installation targets older Python/Transformers versions and does not itself verify entailment. Before replacing the production gate, a relation-level experiment must separately measure (1) whether each headline relation and qualifier was extracted, (2) whether its cited source/antecedent and reporting scope are correct, and (3) whether source entailment agrees with independently reviewed labels on the frozen and genuinely held-out clips. A contract-valid citation, vote, or automatic score is not the acceptance criterion.

The [more recent `qasem-parser` package](https://pypi.org/project/qasem-parser/) is also not a justified drop-in for the specific opponent-role omission. Its documented checkpoint is a Flan-T5-large joint predicate parser, while [QASemConsistency's own stronger T5-XL parser](https://aclanthology.org/2026.tacl-1.6.pdf) reports that the underlying QASRL/QANom training data do not cover copular relations such as “X is Y”; the paper supplied those relations with a separate generator and manually filtered parser errors. Both saved 4B and 30B Clipper proofs classify `relationship_role=not_claimed` for headlines explicitly asserting an “opponent” role while their blind source fact says the relationship is unknown. This is a **claim-inventory recall** failure before entailment. Any parser candidate must first demonstrate coverage of that role, all other frozen relations, and independently audio-reviewed held-out relations; its package name or published aggregate F1 is not qualification. No QASem parser has been run or integrated into Clipper.

`tests/fixtures/issue8_frozen_relations.json` now records fifteen predicate–argument QA controls derived from the twelve frozen headlines, including positive/negative actor, setting, opponent-role, and conditional-versus-actual outcome pairs. `clipper.editorial_benchmark.load_frozen_relations` checks the exact saved proof hash, source/transcript identity, headline answer text, and cited transcript words before a model can be scored. These relation labels inherit the earlier transcript-only annotations; they are **not** new independent audio-reviewed gold, and the loader cannot validate their semantic truth. This fixture isolates a verifier's relation-level behavior; it does not make the production reviewer qualified.

The `relation_minicheck` validation-only workflow mode is a specialist-verifier diagnostic on those 15 QA relations plus all 12 original whole headlines. It scores a pinned [MiniCheck-Flan-T5-Large](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large) checkpoint on CPU using the [upstream scoring method](https://github.com/Liyan06/MiniCheck), records each source window, claim, input length and raw support probability, and checkpoints after every case. The model weight SHA-256 and exact transcript/proof identities are checked; overlong inputs fail rather than truncate. This mode neither selects clips nor alters the production gate. Even 27/27 on these transcript-derived controls would **not** establish audio-grounded or held-out factual accuracy; independent audio-reviewed controls and an integrated editorial review remain necessary before production qualification.

The first hosted diagnostic, [run 37134230546](https://github.com/TheBayoumi/clipper/actions/runs/37134230546) at commit `540b135`, completed all inferences with the pinned `MiniCheck-Flan-T5-Large` weight SHA-256 `41291881e13c6235ed47149cec903bee9493e45d9d7325587a9fa2e266c526c0` and the original proof/transcript identities. It scored 12/15 relations (two false approvals, one false rejection) and 8/12 whole headlines (four false approvals, no false rejections). The false approvals include Sean Shelby as the pacing actor (`p=0.646`), “live TV” as the setting of Bobby's pacing (`p=0.916`), and unsupported whole-headline variants about podcast revenue, live-TV setting and Sean pacing (`p=0.713`, `0.958`, `0.760`, `0.944`). The positive “Instagram views are not income” relation was falsely rejected (`p=0.251`). These are model probabilities, not calibrated truth or factual confidence. This result **rules out** adopting this specialist model as the production factual gate. The run's workflow failure is the intentional semantic fail-closed exit, not a model-download or source mismatch. The next repair must make the app's claim representation and source attribution handle actor, speech-act/quoted-setting, outcome, and conditionality explicitly, then evaluate that end-to-end on independently audio-reviewed held-out controls.

`clipper audio-review-manifest` now prepares those twelve held-out claims for an independent audio listener without exposing provisional expected labels, annotation reasons or mnemonic case IDs. It verifies the original three MP4 byte hashes against the committed held-out fixture and records the exact source/transcript and artifact identities. The generated local manifest is `work/issue8-blind-audio-review.json`; its verdict, reviewer and notes fields are intentionally empty. It is **not** an audio review result, qualification, or production approval. Reviewers must listen to the complete linked MP4s and mark unsupported or uncertain claims as such before any provisional transcript label can be promoted to gold.

After an independent listener fills every `audible_claim_verdict` (`supported`, `unsupported`, `uncertain`, or `inaudible`), `reviewer_id`, and `review_notes`, `clipper audio-review-check` reconstructs the blind manifest from the pinned fixture and actual MP4 hashes, rejects any changed source/headline/case metadata, and only then reports disagreements with the provisional transcript labels. The result retains `reviewer_identity_verified=false`, `audio_gold_qualified=false`, and `production_approved=false`: a self-reported reviewer ID is not authentication, and this comparison cannot silently promote provisional labels to gold or approve clips.

### Automated factual approval policy (October 3, 2026)

The production factual decision must be **fully automated**. Human listening may be used to establish independent benchmark labels, but cannot be a per-clip runtime sign-off or an escape hatch for a failed automatic gate. An unqualified automated claim record is now reported as `automated_unqualified`, not `needs_human_review`; the record remains `production_approved=false`. This status change is policy bookkeeping, **not** a new verifier or a fix to issue #8.

The replacement gate must constrain what can become a headline, not merely ask another model whether free-form prose is true. The model may propose an exchange, evidence positions and candidate claim relations. Clipper must own the canonical source text, enumerate every asserted actor/action, role, setting, quantity, negation, attribution and condition, preserve each relation's source scope, and construct or reject the final headline. Missing relations, ambiguous antecedents, incomplete conditional or quoted-speech scope, and unsupported paraphrases must abstain. A literal quote by itself is not sufficient: [decontextualization research](https://aclanthology.org/2021.tacl-1.27/) shows that excerpts may need their surrounding context to preserve meaning, and [contextomized-quote research](https://arxiv.org/abs/2302.04465) studies misleading headline quotes. The separately reviewed setup/payoff must still make the headline a coherent 4–14 word whole-exchange hook; selecting an easy but irrelevant source sentence does not count.

Qualification requires end-to-end results from the **integrated production path** on the pinned business/excluded-payoff/intro controls, frozen positive/negative relation pairs and independent held-out audio controls. Report false approvals and false rejections separately, claim-inventory omissions, the specific failed relation, abstention and yield, source/model hashes, latency and cache reuse. Do not fit thresholds or phrase rules to the frozen examples. A passing code suite, structurally valid citation, model confidence, or zero-delivery run is not qualification. Until the gate passes, do not dispatch the exact-source production render or label issue #8 fixed.

`clipper.editorial_headline` now provides a diagnostic candidate mechanism: the model selects two exact setup/payoff excerpts; Python materializes the headline and rejects fabricated words, mispositioned excerpts and clipped-off scope words. The `source_bound_headline` validation-only workflow mode runs this candidate through the actual position-based exchange reviewer on the three pinned real windows, records raw model calls and always reports `production_approved=false`. Its output is **not** claim-level contextual entailment: an exact quote can still imply a false setting, relationship or outcome when joined to another excerpt. This mechanism is an experiment toward a constrained headline architecture, not a production gate or permission to render.

The first exact-source probe at [37153745693](https://github.com/TheBayoumi/clipper/actions/runs/37153745693) completed all three exchange windows and nine model calls in 495.7 model-seconds, with zero request-cache hits. It generated **no headline candidates**: the complete business exchange was falsely rejected for a supposed missing contrast to the excluded tools list, while the model chose delivered unit 6 (“You got to transition…”) as the final point and ignored the actual final substantive unit 11 (“No, it just drives the podcast.”). The excluded-payoff window was correctly incomplete but incorrectly labeled an ad read for discussing sponsors; the intro/unfinished window was correctly rejected. This is an upstream boundary/speech-act failure, not evidence that the new headline materializer works on real media. The position reviewer now binds both delivered resolution and continuation comparison to a Python-selected final substantive unit, ignoring only generic trailing acknowledgements. The next validation-only replay must measure whether that structural repair actually changes the model’s boundary judgment; it reuses compatible earlier purpose calls and does not promote a quote or model vote to factual approval.

That replay, [37154872629](https://github.com/TheBayoumi/clipper/actions/runs/37154872629), also failed semantically. It reused three exact purpose requests and ran six changed model requests in 225.4 model-seconds. The complete business resolution now reaches the real final substantive unit 11, but the continuation model **still** calls the excluded tools list a missing contrast and vetoes the complete exchange. It again treats ordinary sponsor discussion as an ad read in the excluded-payoff window. In the intro window it now calls the dangling “where I'm like telling him” thought complete and treats the excluded continuation as optional; the actual intro/ad read still vetoes the aggregate window. No source-bound hook was generated in any of the three cases. This is stronger evidence that citation-position constraints alone do not repair dialogue-state or speech-act reasoning. Do not paper over the wrong reasons with final accept/reject totals or send the source to rendering. The next experiment must represent the delivered unresolved question/clause and the exact excluded answer or continuation as separate obligations, then test those obligations on the same controls before returning to headline generation.

`clipper.editorial_boundary` now provides that **diagnostic** obligation contract. It represents `none`, `question`, `clause`, `contrast` and `uncertain`, with exact delivered pending and excluded fulfilling positions when a missing payoff is claimed. Python reconstructs source text and rejects a question without a delivered question mark, a clause without a dangling final clause, or a missing contrast without contrastive excluded speech. The current proposal interface handles dangling final clauses deterministically; the model may choose only `none`, `question` or `contrast` for the remaining cases. The `cut_obligation` validation-only mode scores the three pinned windows and preserves raw requests; even a 3/3 result is insufficient for production or factual headline approval. This separates a concrete open obligation from mere topical similarity, but it remains unqualified until the real run and held-out controls are measured.

The first obligation probe [37155892551](https://github.com/TheBayoumi/clipper/actions/runs/37155892551) completed three model calls in 197.3 model-seconds. It failed the three-window semantic rule: for complete business the model chose `uncertain` despite explaining that the delivered final point was complete and excluded tools were optional; for the excluded-payoff window it correctly cited the 60-million-view premise and excluded nonpayment as a missing contrast; for the intro it recognized the unfinished “telling him” clause in prose but instead chose the opening “What kind of collab?” question and supplied no excluded answer, so the source contract rejected the response. Version 2 applies a deterministic incomplete-final-clause veto before inference and constrains the remaining model choices to `none`, `question` or `contrast`: no cited question/contrast means `none`, while an invalid model proposal still fails closed. This is an experiment to repair claim inventory and label consistency, **not** a production approval.

The version-2 exact-source replay [37156862592](https://github.com/TheBayoumi/clipper/actions/runs/37156862592) also failed the semantic rule. Its intro was correctly rejected by the deterministic unfinished-clause check without loading the model. The model called the complete business cut a missing contrast to the excluded tools/platform list; Python rejected that claim because the cited excluded text supplied no contrastive speech. Its excluded-payoff assessment truncated after an oversized free-form reason, so no verdict can be inferred for that case. Two model calls consumed 150.4 model-seconds, with zero cache hits. Version 3 removes the model-authored reason from the output schema and lowers the response budget to 80 tokens; Python supplies a neutral contract explanation. This addresses the truncation channel, not the model's false contrast judgment. It remains diagnostic-only; re-measure before changing production.

The version-3 exact-source replay [37180001369](https://github.com/TheBayoumi/clipper/actions/runs/37180001369) completed without contract or truncation errors (two model calls, 134.9 model-seconds, zero cache hits), but still failed the three-window semantic rule. The model now correctly selected `none` for the complete business cut while also selecting `none` for the excluded-payoff cut, which omits the nonpayment contrast. The deterministic unfinished-clause veto correctly rejected the intro. The change therefore fixed the response-shape problem and one false rejection, but exposed a false acceptance on the adjacent source window. A three-case score, especially one that fails, cannot justify production integration. Further prompt-label tuning against these same frozen windows risks fitting the examples; the next implementation needs an independently represented open discourse obligation and held-out evaluation, as well as the separate claim-level factual gate.

`clipper.editorial_question_state` is the next validation-only experiment. It inventories *all explicit transcript questions* before asking separately whether speech after each question answers it within the delivered cut and, only when unanswered there, whether excluded continuation answers it. The model returns source positions and a three-way answer state; Python reconstructs the cited speech, rejects invalid or missing positions, and keeps the result unapproved. This follows the distinction between a question and its answering obligation described by [Traum and Allen](https://aclanthology.org/P94-1001/) and the question-under-discussion representation explored by [Ko et al.](https://aclanthology.org/2023.findings-acl.710/), but neither paper validates this Clipper implementation. The `question_state` workflow mode tests the pinned complete-business and excluded-payoff windows without media acquisition. Transcript punctuation can miss implicit questions; cited source positions do not prove semantic responsiveness; a positive two-window signal is not cut-completeness or factual-headline qualification.

The first exact-source question-state run [37180643626](https://github.com/TheBayoumi/clipper/actions/runs/37180643626) inventoried five explicit questions in the excluded-payoff window and completed six model calls in 259.2 model-seconds, with zero cache hits. It reported one apparent missing answer, but the cited question was “How does it work out for you?” and the delivered response already says “It's working pretty good.” The model incorrectly labeled that delivered response unanswered, then cited the later Instagram nonpayment statement as its answer. This is a **false question/answer linkage**, not a successful payoff detection. The model also cited a broad sponsor-and-view-count range as an answer to “Do you get paid for purviews?”; source-position validity alone cannot establish that it answered the intended question. The report field `two_window_question_signal_pass` was misleading and has been removed in favor of raw signals plus `question_state_semantically_qualified=false`. No production path uses this result.

The `headline_materializer` validation-only route isolates the existing package-owned `clipper.editorial_headline` mechanism from the failing boundary reviewer. It uses benchmark-reviewed setup units 0–2 and delivered resolution unit 11 of the pinned complete-business exchange, asks the model only for exact source excerpts, lets Python construct the candidate, and records the legacy factual model audit separately. These positions are **benchmark annotations, never production selection rules**. Even a valid quote and a supportive audit do not grant factual approval: cross-unit attribution and implication can still be wrong. This probe tests whether the materializer can produce a coherent whole-exchange candidate at all before changing its production integration.

The isolated exact-source run [37181427223](https://github.com/TheBayoumi/clipper/actions/runs/37181427223) made one model call (219.5 model-seconds, zero cache hits) and selected verbatim “I'm like not making a dime off of it.” plus “No, it just drives the podcast.” Python correctly rejected the resulting 15-word hook against the required 4–14-word limit; no factual audit ran and no candidate was approved. More importantly, those two excerpts both rely on “it” and omit the Instagram antecedent. Simply relaxing the word cap or trimming one word would not solve contextual self-containment. The source needs at least the Instagram premise, the nonpayment contrast and the podcast resolution represented together, followed by claim-level contextual verification. The current two-excerpt materializer is not a production-quality headline architecture for this exchange.

Factual approval in production must be fully automated; annotators may create evaluation labels but cannot be a per-clip approval dependency. `clipper.editorial_source_answer` now isolates a claim-blind source-answer step: the model receives one atomic relation question and delivered speech, never the headline's proposed answer. Python requires a focused, exact source quotation for an `answered` response and requires abstention without a citation for `unknown` or `uncertain`. The `source_answer` validation-only mode runs the fifteen pinned relation questions against the hash-verified baseline proof and transcript, saves raw requests and a checkpoint per case, and deliberately exits nonzero. A quote proves provenance, **not** semantic correctness; this step cannot be used as an approval gate until the result is measured, an independent headline claim inventory is complete, and the integrated automatic verifier passes audio-reviewed held-out cases. Production behavior is unchanged.

The exact-source [source-answer run 37183064280](https://github.com/TheBayoumi/clipper/actions/runs/37183064280) completed all fifteen controls (fourteen model calls, one exact request-cache hit, 368.8 model-seconds). The model correctly abstained on the unsupported opponent role and already-earned podcast millions, and answered Bobby rather than Sean as the pacing actor. But for “What was the **actual broadcast setting** while Bobby paced?” it answered `"We're on live TV"` from a reported instruction at the fighter meeting. The cited source says Sean was going over instructions, not that Bobby paced on a live broadcast. This is a concrete scope/modality false answer despite exact source quotation and a blind question. For “What does the podcast drive?” it quoted `"drives the podcast"`, a self-referential fragment that does not answer what the podcast drives; a contract-valid substring can also be nonresponsive. Therefore the blind answerer is not qualified as an automatic factual approver. The workflow's nonzero result is intentional diagnostic behavior; the semantic errors, not the badge, determine the conclusion. Further prompt tuning on these same fifteen cases would not establish generalization. The next verifier needs an independent relation inventory, explicit quote/reporting scope and responsiveness checks, abstention on unresolved claims, and genuinely held-out audio-reviewed evaluation before integration.

The production YouTube render boundary now fails closed if editorial selection produces any clips while the current claim-packet format is still diagnostic-only. `clipper.editorial_review.require_automated_factual_approval_for_render` rejects that packet even if someone changes its `production_approved` field to `true`; the run records `factual_approval=blocked_unqualified` before FFmpeg starts. A zero-clip source still returns a normal no-op report. This is a temporary safety hold, not a successful automatic verifier or a production render. The selector and reviewer checkpoints remain intact for later qualified replay, but no previously cached positive headline can silently bypass the new boundary. Rendering can resume only after a separate, source-bound, claim-level automatic approval contract has been implemented and qualified; a human per-clip sign-off is not the intended runtime solution.

`clipper.editorial_claim_inventory` now isolates the first missing stage of that verifier: given only a proposed 4–14-word headline, it asks for independently checkable relation questions and answer-word positions, then Python reconstructs the exact headline spans, rejects invalid or duplicate records, and reports uncovered headline words. The `claim_inventory` validation-only workflow mode runs this on the twelve hash-verified frozen headlines and measures whether the fifteen annotated relation-kind/answer anchors appear in the inventory. Anchor recall is deliberately weaker than semantic inventory recall: a question can have the right answer word and relation label yet ask the wrong relation, or a whole-headline span can mask an omitted subclaim. No source truth is judged and `production_approved` remains false. This separates claim omission from the subsequent source-answer and scope errors before designing the integrated automatic gate.

The exact-source [claim-inventory run 37184325465](https://github.com/TheBayoumi/clipper/actions/runs/37184325465) completed twelve model calls in 346.3 model-seconds with no cache hits or contract errors. It found only **8/15 relation-kind/answer anchors**, and nine of twelve headlines had uncovered words. Its syntactically invalid answer spans included `Green was` for the opponent role, `podcast, not` for an Instagram/podcast relation, and `millions from` for podcast earnings. These are not factual judgments; they show that model-selected contiguous word offsets cannot be trusted as a complete claim inventory. The workflow exits nonzero intentionally. Do not add a token budget or prompt patch tuned to these headlines and mistake better JSON for generalization.

An isolated local exploration of [spaCy's English dependency parser](https://spacy.io/models/en/) (`spacy==3.8.7`, `en_core_web_sm==3.8.0`, not added to project dependencies) parsed all twelve frozen and twelve provisional held-out *headlines*. It identified the opponent noun as a copular attribute, `could` as a modal auxiliary, and live-TV phrases as prepositional attachments in the relevant examples. That makes syntax-owned predicate/argument obligations a concrete next design candidate: the parser, not the generation model, should identify obligatory headline roles, modifiers and clauses; the model may formulate questions for those fixed slots, but cannot drop or change their answer spans. This inspection does **not** validate parser accuracy, coreference, spoken-transcript scope, source entailment or held-out factual performance. The official model is trained on written web text, so running it on disfluent source speech without separate evaluation would be unjustified.

`clipper.editorial_syntax_inventory` now extracts those predicate/argument/qualifier obligations into package-owned, source-blind frames. `clipper syntax-inventory-benchmark` checks the pinned proof/transcript, runs both frozen and provisional held-out headlines with the official parser wheel pinned by its published SHA-256, and checkpoints every frame and parse warning. The local pinned run found all 15 frozen answer strings inside parsed role spans, but this is **lexical anchor presence**, not proof that all relations are semantically represented. It also raised 11 structural warnings on the 12 provisional held-out headlines, including an auxiliary outside any parsed clause in the invented salaried-employment headline and unresolved implied subjects in infinitives. This is useful architecture evidence and an explicit abstention signal, not a qualified automatic verifier or permission to render. The production dependency list and factual gate remain unchanged pending source-entailment work.

An exploratory exact dependency-frame alignment between the headline and spoken source matched **zero of fifteen** frozen relations, including clear positives: transcription disfluency, paraphrase and implicit subjects made it a zero-yield gate. It was not promoted into production. The separate `source_scope` diagnostic instead re-reads the complete delivered source units around each previously saved, claim-blind exact quotation. It asks whether that quotation answers the original relation question and whether its speech act is an actual event/state, a reported opinion, quoted instruction, conditional, negation or unresolved. Its input excludes the headline's proposed answer. The workflow pins the saved source-answer report, baseline proof, transcript and provenance to the same exact source and checkpoints all fifteen cases. These scope labels are model judgments, **not** source entailment or automatic approval; the existing production factual hold remains until a complete claim inventory and a separately qualified, fully automated verifier pass audio-reviewed held-out controls. Human annotation is for evaluation only, never per-clip runtime approval.

`tests/fixtures/issue8_source_answer_scope_gold.json` seals transcript-derived responsiveness and speech-act labels for all fifteen saved blind answers, bound to the SHA-256 of that exact diagnostic report. The probe validates **all** saved case identities and source units before any model call, never includes the gold labels in model input, and reports exact-pair mismatches after inference. A local contract replay against the pinned proof, transcript, source-answer report and gold fixture processed 15/15 cases; its injected constant classifier scored 3/15 as expected and says nothing about real-model accuracy. These labels do not constitute independent audio gold or a calibrated production threshold.

The scope score separates answered records that actually invoke the classifier from the source answerer's automatic abstentions. Three of the fifteen pinned answers abstained before scope inference; counting those as model successes would inflate a scope-model accuracy claim. The report therefore records both the overall exact-pair count and the answered-only denominator, plus any false-responsive labels on quotes annotated as nonanswers.

The three held-out MP4s and twelve-case blind review manifest have been rechecked against their pinned hashes. A separate [offline ASR cross-check](issue8-audio-asr-crosscheck.md) corroborates several key speech distinctions but also contains recognition errors; it is diagnostic evidence, not independent audio gold. All twelve blind verdicts remain unfilled.

The hosted [syntax-inventory run 37193565355](https://github.com/TheBayoumi/clipper/actions/runs/37193565355) completed its pinned, no-media evaluation. Its 12 frozen and 12 provisional held-out frame records exactly match the isolated local replay: 15/15 **lexical answer anchors present**, 11 parser warnings on held-out headlines, `production_approved=false`. The parser wheel URL was SHA-256-constrained at installation; the report separately states that it did not independently re-hash the installed wheel. The action's nonzero result is the probe's intentional diagnostic exit. This does not repair the source-answer false broadcast setting, certify semantic relation completeness, or permit render. The next integration must bind each parser-owned role and qualifier to independently evaluated source evidence and reject unsupported scope or unanswered obligations.
