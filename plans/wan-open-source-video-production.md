# Clipper original-book videos: production contract (30-60 seconds)

Branch: `feat/unified-niche-tts`. Review: PR #10. Publishing remains disabled.

## Hard product requirement

- **Each finished episode is 30-60 seconds**, inclusive, measured from real
  cached Kokoro narration, not predicted from a word count.
- One end-to-end GitHub production run must eventually produce one complete
  9:16, 1080x1920 MP4 at 24 fps, not just a short proof shot.
- A **variable number of meaningful shots** follows the narrative, not a
  hard-coded three-shot storyboard. Production shots typically require 2.5-6
  seconds of video and cover the narration without artificial padding.
- Preserve the recurring approved character identity, coherent scene geography,
  original book attribution, physical prop interactions, spoken narration,
  cinematic sound design, subtitles, and editorial continuity.
- Reuse existing source-hash, script, voice, model, keyframe and shot caches.
  Failed shots can be regenerated independently. No silent source replacement.
- Review actual rendered frames, audiovisual sync and full MP4. Codec and CI
  success alone do **not** constitute creative acceptance.

## Book selection and semantic storyboard (new)

- \`niches/book-editorial-catalog.json\` contains sourced, single-idea
  editorial candidates across Atomic Habits, Deep Work and The Psychology of
  Money. A source URL, factual claim, topical hook and physical visual
  demonstration are required for each selectable lesson.
- \`scripts/book_episode_selection.py\` implements two modes:
  (1) automatic rotation over **source-backed** books/lessons, scoring down
  recently accepted books and lessons, and (2) explicit \`book_id\` /
  \`lesson_id\` override (optional pillar filter). Neither writes new,
  unverified author/book claims. This selection is currently deterministic,
  not an LLM-trained taste/ranking system.
- \`niches/accepted-episode-history.json\` is currently empty because no
  episode has been accepted for publishing; persist an entry there **only
  after acceptance**, or book diversity will reset across workflow runs.
- The selected book/lesson compiles to an actual source-locked Qwen3 niche
  script config and an original semantic story outline via the same
  selection script. No manual storyboard image transfer is needed at
  this stage.
- \`niches/atomic-habits-approved-storyboard.json\` now has an adaptive
  \`production_storyboard\` narrative arc: pattern-breaking hook, relatable
  problem, book/author attribution, one mechanism, visible before/after
  action, truthful payoff, and one-step takeaway. The roles are flexible,
  **not** timestamps or a mandated count of rendered shots. Historical
  15s pilot is retained solely as a regression fixture.
- Cheap GitHub editorial action:
  \`/.github/workflows/book-editorial-selection.yml\`. Its actual
  \`workflow_dispatch\` CPU run #40135945366 passed seven new tests and
  selected **Atomic Habits / Make It Easy**, emitting:
  \`selection.json\`, \`qwen-niche-config.yaml\`, \`story-outline.json\`.
  It allocated **no GPU** and did **not** produce a video.
- The remaining integration is to have a genuine AI director build
  idea-aligned, meaningful scene actions and choose S2V/I2V/Animate models
  from that outline plus Kokoro word/phrase timing; then assemble one
  complete 30-60s MP4. The existing 15s pilot video workflow has not
  been secretly switched to this new production path.

## Current implementation boundaries

1. Qwen3 can generate and cache a book-lesson script. Kokoro can synthesize
   and cache narrated WAVs. The generic script prompt aims for roughly 80-100
   words, but text word count **does not prove** final video length.
2. `scripts/wan_episode_plan.py` is the production-specific versioned
   timeline contract. It accepts the existing scene-script JSON plus each
   corresponding **actual cached** 24kHz Kokoro WAV, validates total duration
   between 30 and 60 seconds, derives exact audio/frame boundaries, outputs
   a variable number of timed beat slots, and retains cache SHA256 evidence.
   It does not add fillers, chop voice audio, or repeat shots to meet time.
3. The production slots intentionally have
   `status=NEEDS_STORY_DIRECTION` and no video backend yet. A real open-source
   AI story director must provide each beat's visual action, camera, scene
   continuity, narration alignment and S2V/I2V/Animate routing before those
   slots become renderable. Equal audio-slot partitioning alone is **not**
   editorial shot selection and must be refined at speech/idea boundaries.
4. The existing `wan_storyboard_plan.py`,
   `wan_keyframe_contract.py` and
   `.github/workflows/wan-cinematic-production.yml` still prove the
   **separate historical 15-second Atomic Habits technical pilot**.
   They are not a valid production episode contract. Do not inflate or
   promote that pilot as a 30-60s complete video.
5. The successful open-source Qwen-to-Wan one-shot technical execution is
   https://github.com/TheBayoumi/clipper/actions/runs/40126182865 .
   It generated a talking-character scene, passed technical MP4 QA, and
   retained `publish_eligible=false`. It did **not** produce a finished episode.
6. Wan I2V and Animate-2 full action rendering, automated story-direction,
   exact shot stitching, final 30-60s full-video creative QA, and complete
   end-to-end production are **not yet proven** and remain gating work.


## Modal quota-failover configuration (2026-10-25)

The active cinematic workflow authenticates GitHub Actions to Modal using
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` and provisions the `clipper-wan-hf`
secret from GitHub's existing `HF_TOKEN`. This is not a Modal-hosted
video API: GitHub looks up a code-versioned deployed Modal GPU function,
deploys only when that function version is missing, and invokes it by name.
It does not use `modal run` in the current cinematic workflow.

Backup routing is implemented in
`.github/workflows/wan-cinematic-production.yml` and uses **four configured
GitHub Actions repository secret values**, representing two complete
workspace credentials:

- Workspace slot 2: `MODAL_TOKEN_ID_2` + `MODAL_TOKEN_SECRET_2`
- Workspace slot 3: `MODAL_TOKEN_ID_3` + `MODAL_TOKEN_SECRET_3`

Each credential pair must belong to an authorized, independent Modal
workspace. Extra tokens for the **same** workspace do not provide additional
quota. The workflow does not create credentials or enroll new workspaces;
the secret values must be provisioned securely by a workspace administrator.
Partial credential pairs fail closed; missing optional pairs are skipped.

For each uncached Qwen or Wan GPU stage, the workflow authenticates
the chosen workspace and reuses its existing code-versioned app. Only when
the specific app version is absent does it provision `clipper-wan-hf`
and deploy before invoking inference in that **same** workspace.
It attempts the preferred slot, then remaining configured slots
only when the *inference* subprocess terminates with an explicit quota
exhaustion error. Authentication, secret provisioning, deployment, CUDA OOM,
timeout, rate-limit, credit or spending-budget failures must not switch accounts.
The router proceeds in ascending workspace order, never revisiting a known
quota-exhausted workspace. The selected slot is propagated across stages and recorded in the uploaded
provenance evidence without recording token values.

Workspace Modal Volumes and deployed apps are isolated by workspace.
Weights may download and build anew in a backup workspace, increasing
latency and cost. The Qwen local-entrypoint reuses already generated frames
only after checking the request identity, source/model revision, output
size and actual PNG SHA256, so account failover does not need to regenerate
completed frames within a single GitHub run.

**Budget policy:** Account failover is intended for separately authorized
workspaces with approved budgets, not to circumvent platform/account usage
limitations or defeat a deliberately set global cost ceiling. Configure
per-workspace spending and a project-level total budget before enabling
backup credentials.

**Verification:** All three configured token pairs authenticated successfully
in credentials-mode run #40138631184. That run exposed a condition bug:
credentials mode also executed a cached-Qwen keyframe GPU stage (cache=REUSED).
The mode condition has since been corrected to prevent any credential-only
GPU dispatch. Expanded Mock-Modal subprocess tests exercise only-quota
failover, two successive quota failures, and fail-closed non-quota errors
without allocating a GPU. Actual quota-triggered cross-workspace inference
is not yet proven; credentials authentication alone is not compute entitlement.
The retired TRELLIS and legacy Wan technical qualification workflows are
not the active production account-router; do not infer they have fallback.



## Versioned deployments and shared shot cache

Implemented on `feat/unified-niche-tts` for the existing pilot workflow:

- `scripts/modal_versioned_dispatch.sh` computes an immutable application
  name using SHA256 of the GPU app source (`modal_qwen_keyframes.py` or
  `modal_wan_s2v.py`). It probes the deployed Modal function by app name
  and function name, reuses it when present, and deploys **only** a missing
  application version on the workspace selected for inference. Qwen and Wan
  CPU clients call `modal.Function.from_name(...).remote(...)`, not
  `modal run scripts/...`. Backup applications are installed lazily and
  only on explicit inference-quota exhaustion, without falling back on
  authentication, deployment, CUDA, budget, credit, model or timeout errors.
- `scripts/wan_shared_shot_cache.py` checks the most recent 15 completed
  runs of the **same GitHub workflow and branch** and restores verified shot
  candidates from the existing immutable GitHub Actions artifact bundle.
  A Qwen keyframe is checked against its request fingerprint, source artwork,
  storyboard/model revisions, output SHA256 and frame dimensions.
  A Wan MP4 is checked against image, audio, prompt, seed, pinned model
  revision and the recorded MP4 SHA256. Reusable results are independent of
  Modal workspace and copied to the current GitHub runner; the workflow
  skips Modal entirely when the relevant shot already exists.
- The existing `if: always()` artifact upload captures completed shots
  and evidence even when an inference stage fails. A later run can reuse
  **completed** shots from that failed workflow. Partially generated Wan
  frames, unfinished model inference, and workspace-specific model weights
  cannot be resumed or shared. The existing upload bundle is shared storage;
  artifacts are not yet individually indexed/uploaded immediately per shot.
- Modal's persistent model-weight Volumes remain workspace-specific.
  Switching accounts can still require the destination workspace to build
  and download weights on its first use, but **does not restart the
  orchestration**, re-run cached editorial/TTS, or regenerate an already
  validated shot.
- `mode: credentials` remains GPU-free; this feature must be verified
  using CPU preflight and simulated Modal calls before expensive inference.
  Model/frame creative approval remains independent of technical caching.

This is infrastructure for the current **three-shot pilot**, not evidence
that the planned 30–60-second directed episode, I2V shots, or final montage
is already complete.


## Live acceptance evidence — versioned deployment and shared shot cache

**Architecture milestone: ACCEPTED on 2026-10-25.** Evidence is bound to the
`feat/unified-niche-tts` branch and these exact GitHub Actions executions:

1. **GPU-free SDK compatibility qualification**
   [#40141969771](https://github.com/TheBayoumi/clipper/actions/runs/40141969771):
   all three authorized Modal token pairs authenticated. The Python SDK
   successfully identified an absent deployed function without starting a
   GPU. This superseded a failed #40141704718 probe that had called the
   unsupported `modal function info` CLI in pinned `modal==1.5.5`.
2. **Real Wan GPU candidate**
   [#40142029622](https://github.com/TheBayoumi/clipper/actions/runs/40142029622):
   restored the hash-verified Qwen hook keyframe and Kokoro voice;
   `SHOT_CACHE_MISS stage=speaking` for deterministic seed 725;
   `MODAL_DEPLOY_CREATED version=ae902bca2de71934 slot=1`;
   `MODAL_INFERENCE_COMPLETED slot=1`;
   `VIDEO_TECHNICALLY_VALID_HUMAN_CREATIVE_REVIEW_PENDING`.
   The archived, decoded MP4 is **4.8125 seconds at 640x1088**,
   SHA256 `c50dbc04d0234d523db52d06d880d72aba51be608fc8823b0faf89589fb59918`;
   2,707,741 bytes. The original Qwen frame SHA256 is
   `7f03c145af1c1c2f789aba11beadddccdc2649d8fc6bb613f454aa61ea778b67`.
   [Video + provenance artifact](https://github.com/TheBayoumi/clipper/actions/runs/40142029622/artifacts/12579988161).
   Primary workspace was used; no actual backup failover was needed.
3. **Cross-run cache-only replay**
   [#40144449633](https://github.com/TheBayoumi/clipper/actions/runs/40144449633):
   same seed 725, keyframe and S2V input hashes;
   `SHOT_CACHE_RESTORED_KEYFRAMES ... count=1`,
   `SHOT_CACHE_RESTORED_WAN ... count=1`,
   `SHOT_CACHE_CHECK stage=speaking complete=True`,
   `MODAL_S2V_SKIPPED_VERIFIED_MP4`,
   then full MP4 integrity/decode audit passed.
   The `require_cached_video=true` gate failed closed on any cache miss,
   forbidding a second GPU inference. This run proves the artifact survived
   GitHub jobs and was recovered independent of the original Modal process.
   An earlier replay attempt #40144406818 was blocked by Ruff formatting
   before any GPU stage; that defect was corrected and the successful
   replay supersedes it.

An independent archive inspection confirmed the actual MP4 SHA against both
`model-evidence.json` and `evidence/shot-qa.json`; spot-checked decoded
frames show the same stylized host in the warm studio holding a phone.
That is **not** human creative approval: full lip-sync, motion, narrative
quality and identity consistency remain unassessed.

**Boundaries:** This is a **single short speaking-shot infrastructure
qualification**, not the requested 30–60-second complete episode. In
particular, there is no approved action-shot I2V/Animate route, complete
adaptive cinematic director, final montage/subtitles/ambience, or 1080x1920
production video. Source artwork remains provisional below the 1024px
creative-reference threshold. GitHub artifact retention is 14 days, so
this is a bounded cross-run cache rather than permanent external object
storage. Simulated quota-only failover is covered by tests; **live backup
GPU inference after an actual quota error is not yet demonstrated.**



## Full-length production recovery — parallel-deploy race and inference deadlines (2026-10-25)

Evidence: [full-book run #40174311275](https://github.com/TheBayoumi/clipper/actions/runs/40174311275) completed two immutable candidates
(scene 0 and 2). Scene 1 failed because two GitHub render workers raced to
deploy the **same** Wan application version, not because invoking a shared
deployed function is invalid. Scene 3 reused the deployed app but the child
`Wan2.2/generate.py` process hit Clipper's own `subprocess.run(timeout=4800)`
after 80 minutes. Neither failure qualifies for backup-account routing.
The failed scene 3 artifact records a failure, not an MP4 cache hit.

Applied:
- A single dedicated `deploy_wan` GitHub job runs **after the immutable
  source/character preparation** and **before the parallel render matrix**.
  It uses `modal_versioned_dispatch.sh ... --deploy-only`, which authenticates,
  installs an app version only when missing and never invokes a GPU function.
  Render jobs depend on this gate; any overlapping deployment also retains the
  already-implemented winner-reuse race recovery.
- Removed the **4,800-second subprocess deadline** from the Wan worker.
  The Modal function now uses the **maximum 86,400 seconds / 24 hours**,
  rather than our earlier 7,200-second cap. This changes the immutable app
  version, so Modal deploys the new version once; persistent model-weight
  Volumes remain workspace scoped. Completed scenes 0 and 2 are recovered
  from hash-verified earlier GitHub artifacts and do **not** get rerendered.
- Removed all custom `timeout-minutes` values from the cinematic and
  full-book GitHub jobs; removed the book FFmpeg/ffprobe and shot artifact
  download subprocess deadlines. This removes **Clipper-imposed** aborts,
  not the hosting services' execution limits.
- Keep strict **quota-only** account failover, no fallback for deployment
  errors, CUDA errors, spending caps or execution timeouts, and per-shot
  artifact recovery after failures.

**Platform reality:** GitHub-hosted job execution is capped at 6 hours
(including when `timeout-minutes` is omitted), and Modal functions at
24 hours. Truly unbounded GPU execution is not possible on these platforms.
Very long shots must eventually be split into independently resumable
semantic scenes (or use durable detached Modal call IDs with later recovery)
if they exceed the GitHub job lifetime. No human creative approval or
30–60-second finished video is implied by the infrastructure tests.



## Current visual-and-narration corrections — October 25, 2026

- **Preserve scene 0:** Original picture from [#40174311275](https://github.com/TheBayoumi/clipper/actions/runs/40174311275),
  scene-00, video SHA256 `6ba8c31509d75a808d31f29af9f909f9dc04c76628346cf368b35d8ab4a286b1`,
  Qwen input SHA256 `7f03c145af1c1c2f789aba11beadddccdc2649d8fc6bb613f454aa61ea778b67`.
  This user's **visual** acceptance is handled by
  `scripts/wan_full_book_scene_zero.py`. All original audio is muted.
  New narration is separately sourced and a modest visual retime is permitted;
  lip-sync with new narration is **not** claimed.
- **Replace the defective book intro:** Cached `book_intro.png` from
  [#40190367185](https://github.com/TheBayoumi/clipper/actions/runs/40190367185)
  contains a Qwen-created black lower-third subtitle panel and gibberish
  physical book-cover text. This is a defective **source keyframe**, not merely
  unwanted ASS captions. Its Qwen prompt fingerprint is changed **only for
  book_intro**, requiring a new image. The prompt requires on the **physical
  cover** exactly `ATOMIC HABITS` and `JAMES CLEAR`, and forbids any
  generated subtitle panel or stray text. A source-frame panel detector fails
  before Wan GPU allocation if the same historical dark overlay is present.
  Physical book lettering and potential new gibberish still need frame-by-frame
  **human creative inspection**, not merely a passing FFmpeg audit.
- **Original baritone voice:** Full book production regenerates narration
  *only* from the cached four-scene Qwen script using Kokoro
  `hexgrad/Kokoro-82M` voice `am_adam`. This is an original deep, warm
  cinematic male narrator, **not** a Morgan Freeman clone/impersonation.
  TTS has its own content-addressed cache. A new audio hash invalidates all
  old synchronized Wan scene results except the explicitly video-only scene 0.
  Word-timed Whisper captions are recomputed from the actual new voice.
- **No unnecessary GPU:** Registered `full_book_prepare` validates
  source audio, voice duration, word captions and hashes on GitHub CPU.
  `full_book_visual` separately regenerates only the bad Qwen book shot so
  source image quality can be evaluated before expensive Wan inference.
  The full render restores completed verified scene artifacts; publishing
  remains disabled.
- **Known Modal blocker:** On [#40190367185](https://github.com/TheBayoumi/clipper/actions/runs/40190367185),
  scene 1 and scene 3 were rejected by Modal with
  `ConflictError: workspace ... is disabled` (not an explicit quota error).
  The **quota-only fallback policy is unchanged**. Backup workspaces MUST NOT
  be used for a disabled, unauthorized, authentication, budget or runtime-error
  failure; primary entitlement or an explicit qualifying quota response must
  be established before GPU renders resume.
- **Kokoro environment recovery:** A fresh GitHub runner upgraded native
  `murmurhash`, `cymem`, `preshed`, `srsly` wheels, causing a spaCy
  English tokenizer integer overflow. We pinned the native versions to
  the proven original draft job and added explicit `en_core_web_sm` loading
  as a CPU qualification gate; GPU rendering is not attempted on CPU failures.

Current target is an end-to-end **30–60s professionally edited portrait MP4**,
not simply a technically valid single scene. Do not mark the full episode as
complete until source book text, absence of baked subtitles, synchronized
voiceover, remaining action shots and final montage have been reviewed.



## Verified original male-narrator recovery and current GPU blocker

- [CPU source/voice/caption qualification #40195091920](https://github.com/TheBayoumi/clipper/actions/runs/40195091920) **passed**.
  The source-locked four-scene Qwen script generated all four new audio
  passages in Kokoro `am_adam`, producing exactly **37.270 seconds** of
  24 kHz mono male narration. The script/source fingerprint and voice/cache
  identity remain recorded in `plan.json`. Narrator voice is an original
  cinematic baritone, **not an imitation of Morgan Freeman**.
  The open-weight Whisper word alignment passed with **114 words and
  agreement=1.000**; the final ASS word timings were saved before any GPU
  inference. [Immutable preparation artifact](https://github.com/TheBayoumi/clipper/actions/runs/40195091920/artifacts/12598949670)
  contains original Kokoro WAV, segmented WAVs, captions, and evidence.
  `faster-whisper==1.2.1` was pinned alongside the previously qualified
  `av==18.1.0` because later PyAV 19 removed the
  `metadata_errors` argument required by the decoder.
- Scene zero picture is pinned to the prior immutable MP4 SHA and must be
  reused without old narration. The final assembler retimes its video-only
  content modestly against new voice, and flags lip-sync quality as a
  creative review requirement.
- The defective book keyframe's Qwen request changed only for
  `book_intro`; exact book-cover words `ATOMIC HABITS` and `JAMES
  CLEAR` are required on the physical prop, while in-scene subtitles and
  gibberish are explicitly forbidden. A known historical subtitle-box
  detector blocks bad repeat generations before Wan inference, although
  correctly spelled lettering still requires visual approval.
- **GPU is currently blocked** by primary Modal workspace
  `ConflictError: workspace ... is disabled` on
  [#40190367185](https://github.com/TheBayoumi/clipper/actions/runs/40190367185).
  This is not explicit quota exhaustion, so the strict quota-only fallback
  forbids an automatic switch to backups. No corrected Qwen book frame
  or new-voice Wan shot is claimed as rendered. Reactivating the
  authorized primary workspace (or receiving an actual quota-specific
  error under the agreed policy) is required before continuing GPU work.
- [CI #40195082878](https://github.com/TheBayoumi/clipper/actions/runs/40195082878)
  passed on the exact CPU acceptance commit. Full 30–60s cinematic
  creative/product acceptance remains **pending**, and auto-publication
  stays disabled.



## Modal A100 authorization check — October 25, 2026

The user reports that both backup workspaces were updated for A100 use.
We ran **actual minimal A100-80GB GPU probes**, without loading or rendering
Qwen or Wan models:

- [#40209029353](https://github.com/TheBayoumi/clipper/actions/runs/40209029353):
  Backup slot 2 (`mahmoud-bayoumi-mb`) failed **before GPU allocation**
  with Modal: `Workspace ... has exceeded its spend limit`. This is a
  workspace **spending-limit failure**, not a GPU quota exhaustion event,
  and should not trigger quota-only account failover.
- [#40209139919](https://github.com/TheBayoumi/clipper/actions/runs/40209139919):
  Backup slot 3 (`thebayoumi`) **successfully allocated** A100-80GB:
  `MODAL_A100_GPU_ACCESS_VERIFIED`,
  `MODAL_A100_PROBE_PASSED slot=3`. The check was isolated to slot 3
  and did not render scenes or install model weights.

Primary workspace stays excluded from Clipper; no fallback is permitted on
billing/spend-limit failures under the user's **quota-only switching policy**.
Full episode production should remain paused until backup slot 2's spending
limit is addressed, or the user explicitly authorizes a different initial
backup selection. GPU eligibility on slot 3 must not be misrepresented as
success of the entire 30–60 second finished video.



## Live correction: backup slot 2 was reauthenticated — October 25, 2026

The user changed the GitHub secret for slot 2 after the earlier billing
test. **Do not treat the previous `mahmoud-bayoumi-mb` mapping as current.**

- [Read-only credential audit #40209470334](https://github.com/TheBayoumi/clipper/actions/runs/40209470334)
  passed with the current secrets:
  `MODAL_SLOT_2_WORKSPACE=mahmoud-bayoumi-bb` and
  `MODAL_SLOT_3_WORKSPACE=thebayoumi`. Primary remains excluded.
- [Live A100 probe #40209548199](https://github.com/TheBayoumi/clipper/actions/runs/40209548199)
  selected slot 2, but Modal refused before GPU allocation with
  `Workspace ac-0LNag4aLq4g5YG1fg5GCTq has exceeded its spend limit`.
  Importantly, **the same internal workspace ID** was returned by
  [the earlier slot-2 A100 probe #40209029353](https://github.com/TheBayoumi/clipper/actions/runs/40209029353),
  which had identified the workspace by the earlier display name.
  Thus, current token-info metadata identifies a new workspace *name*,
  but the live authorization check still resolves to the same
  spending-limited Modal internal workspace. This is a verified
  discrepancy; **the cause has not been proven** (name change,
  account/workspace relationship, or credential routing).
- [Backup slot-3 A100 probe #40209139919](https://github.com/TheBayoumi/clipper/actions/runs/40209139919)
  succeeded earlier, independently of slot 2.

No new Qwen/Wan inference or full episode was launched during this
correction. Spend-limit errors remain excluded from **quota-only**
fallback. Before changing the initial GPU workspace or charging
other accounts, reconcile the internal workspace ID and spending
limit in Modal account/workspace settings or receive an explicit user
instruction to select slot 3 initially.


## Acceptance and regression policy

- Preserve existing archived 15-second storyboard tests for backwards
  compatibility, but production must fail closed for any duration shorter
  than 30 seconds or longer than 60 seconds.
- Use the actual synthesized, cached audio as authoritative timing, and
  have the script/editor rewrite the episode if narration is out of range.
  Never solve duration mismatch through silent padding or filler footage.
- Use source evidence for fact claims and require real visual review of the
  exact output video before any publishing action.
- No background tasks, scheduled polling or auto-publication are added.
