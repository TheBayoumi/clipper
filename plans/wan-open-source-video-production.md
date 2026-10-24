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
