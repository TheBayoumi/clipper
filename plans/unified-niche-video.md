# Unified Clipper: original niche videos

## Branch consolidation decision (2026-10-23)

Development target: `feat/unified-niche-tts`, based on
`feat/contentrewards-campaign-factory`. Keep all source branches intact.

The main branch is a much older clipping-only baseline. The two major
development lines are not trivially mergeable: `feat/contentrewards-campaign-factory`
contains the general quality-moment/video architecture, while
`feat/tjr-weekly-hd-actions` diverged with specialized interview
editorial and cache-recovery contracts. Never mark a branch "merged" merely
because its name or features are documented. Review and transfer only
applicable components, with tests and exact-head proof.

## Goal

One extensible content factory with two sources:

* `source_type=licensed_clip`: historical rights-gated clip extraction.
* `source_type=original_niche`: original narration script -> TTS -> original
  vertical visual scenes -> media QC -> human review -> authorized publishing.

The original-niche mode never ingests or reformats third-party campaign media
and must not inherit Whop watermark, source-rights, or campaign selection rules.
Reusing cache, editorial QC, render utilities, and telemetry requires adapting
their contracts, not bypassing them.

## Implemented first vertical slice

* Niche/topic/audience/voice configuration in `niches/ai-workflows.yaml`.
* Schema-constrained script generation from approved facts; fail on incomplete
  generation, wrong scene counts, overly long text, missing credentials.
* Content-addressed script and TTS caching with atomic writes.
* TTS generation from OpenAI's speech API, then 1080x1920 H.264/AAC
  branded scene-card rendering on GitHub Actions CPU.
* Independent ffprobe stream/duration checks and full FFmpeg decode.
* Signed-off-by-machine draft manifest (source revision, request hash,
  media SHA256, AI voice disclosure). Publication is **never automatic**.
* GitHub-only preflight on branch pushes; paid generation only on an explicit
  manual dispatch with GitHub Secret `OPENAI_API_KEY`.

## Next acceptance gates (not implemented)

1. Adapt multimodal editorial reviewer to check AI-narrated story arc,
   fact provenance, hook/payoff, originality, and reject low-quality scripts.
2. Replace simple text-card scene design with licensed/original motion
   graphics, illustrative B-roll, editable visual plans, and actual
   scene-level caption timing. Do not pretend this prototype is viral-ready.
3. Add speech alignment (not proportional/guessed word timestamps),
   full spoken-word captions, timing QA and audio listening approval.
4. Add niche episode queue with episode IDs, topic diversity, cross-run
   duplicate prevention, source/fact citations, and editorial retries.
5. Add a publisher adapter (TikTok first, then Shorts/Reels) using
   account-authorized OAuth and official APIs. Respect account-specific
   platform rules; default to private/draft where supported.
6. Add independent human approval for an exact MP4 SHA256, an account
   allowlist, a per-video idempotency key and immutable publish receipts.
7. Once the original mode passes a real end-to-end test, review/cherry-pick
   reusable TJR checkpoints/reviewer logic and merge this branch via
   an independently reviewed PR. Archive old branch names only after
   evidence-based feature reconciliation, never destructively.

## Reproducibility and cost

One cached script request per exact niche/topic/model/contract and one cached
TTS request per scene/voice/model/prompt. The GitHub cache is best-effort,
not a durable database: expired or evicted cache entries may be regenerated
and incur API charges. The workflow uses no GPU. Model sampling is not
guaranteed deterministic on a cache miss. Source SHA and output SHA are
recorded. No paid calls are issued by PR CI or branch push.

## Acceptance truth

A successful draft run proves only that a playable, structurally valid MP4
was generated. It does **not** prove creative quality, exact word timings,
fact accuracy, commercial music/footage rights, subjective audio listening,
or public posting. The publish gate remains closed until those are proven.
