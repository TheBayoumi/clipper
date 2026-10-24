# Page to Life — original animated book lessons (provisional identity)

Niche confirmed: actionable 30–60 second lessons from self-improvement,
psychology and business books. Not full summaries, not wholesale reproductions.
Every episode: relatable problem -> book-derived insight -> original practical
example -> one doable takeaway. Attribute the title and author. Use only approved,
source-backed claims and never fabricate quotations, statistics or endorsements.

## Recurring host

**Pip** is an original, bespectacled cartoon presenter with a mint sweater, a
small gold star and a friendly expressive face. The presenter is programmatically
drawn as reusable transparent PNG layers so gestures/bobbing and alternating
mouth-open/closed expressions work in GitHub CPU FFmpeg with no image API.
This is **not** accurate phoneme-level lip sync; no claim of precise alignment.
Working channel name: Page to Life. Brand/name and concept need uniqueness and
trademark review before public launch.

## Technical implementation

- `niches/book-lessons.yaml`: approved first episode, Atomic Habits / James Clear
  and author's official article. Renderer is enabled only for book_lesson config.
- `scripts/book_cartoon.py`: procedural artwork and two face poses, scene-specific
  cover card and labels, safe portrait typography and original book icon.
- `scripts/niche_video.py`: explicit book context in Qwen prompt, a source URL
  provenance field in manifest, existing strict script validation, Qwen cache and
  Kokoro voice cache retained. Four storyboard beats drive scene variation.
- `niches/book-lesson-backlog.yaml`: 20 story seeds, all research-required and
  **never auto-publishable** until source review and approved facts exist.
- Github CPU renders synthetic smoke on push; full Qwen/Kokoro draft requires
  explicit workflow dispatch, uploads media QC and source metadata.

## Acceptance gates

1. Render real 1080x1920 animated-character MP4 with Kokoro narration, full
   audio/video decode, source-attributed manifest, cache reuse; CI verified.
2. Human listening review for delivery, name pronunciation, cut-off endings.
3. Editorial assessment of book interpretation/attribution, no fabricated
   book claims, originality, actual usefulness, hook and takeaway.
4. Improve artwork, add properly aligned captions and audio-driven facial
   animation (current talking pose is approximate).
5. Build auditable original asset rights, automatic duplicate detection and
   niche episode selection based on approved facts.
6. Publishing requires channel/account authorisation and exact-MP4 human
   approval and idempotent receipts. Until then, all runs remain drafts.

## Source for pilot

James Clear, Four Laws of Behavior Change (excerpt from Atomic Habits):
https://jamesclear.com/three-steps-habit-change

## GitHub run recovery and book sentence selection v2

The October 24 run #40045892788 recorded a genuine rejected **260-word** Qwen3
candidate. The previous v1 editor could not keep both the opening and closing
full sentences in its artificial 30-word per-scene cap. Version 2 allocates up
to 45 words to the finale, keeps all scene opening sentences, selects complete
supporting sentences up to 120 words total, and preserves the existing hard
45–125 words global contract. No split/half-sentence trimming, no added claims.

To avoid expensive identical Qwen inference, dispatch with
`recover_from_run=40045892788` to retrieve that exact recorded GitHub audit
via the authorized GitHub CLI, validate and cache its original completion under
an updated content hash and append source-run provenance to the MP4 manifest.
This is source replay, not a new model generation and **does not certify the
book lesson's factual/editorial quality**. All artifacts remain unpublished.
