# Private source-bound chapter-film preproduction (no visuals)

**Not the one-lesson short.** This separate path accepts an existing
canonical, complete **section-scoped** mining ledger and generates a connected
chapter story in four **text** stages, with optional real CPU Kokoro WAV
narration. It neither uses `BookOrchestrator.plan/run` nor the legacy
catalog/Atomic Habits pilot, short `Episode` validator or sentence compactor.

## Stage map, authority and stop boundary

1. **Frozen original source**: `books ingest` or existing retained
   acquisition/ingest -> canonical `whole_book.plan_for(book, section_ids=...)`
   -> complete scoped mining progress and immutable export. The parent book
   fingerprint/edition, ingest identity, source URLs and SHA, section manifest,
   original evidence IDs, anchors, word offsets, excluded units, source warnings,
   recovery status and `authorization_status=unverified` are preserved.
   A complete section does **not** authenticate a book or cover the whole book.
   No new text, edition or mini-book is forged.
2. **ChapterBrief**: a **Qwen-generated** chapter argument, original example/
   assertion beats, essential flags, candid qualifications and an explicit
   include/omit reason for **every** mined candidate. Raw drafts are privately
   retained; deterministic source IDs/coverage are checked. Review state:
   `MINED_UNREVIEWED`, **semantic entailment not proved**.
3. **Human beat review checkpoint**: a private human reviews each generated
   beat and records support/rejection, reason and exact source qualifier
   (if applicable). All source beats need decisions; fewer than two supported
   beats = valid abstention. This is **assisted**, not automated semantic proof
   or publication approval. No adaptation is run before this checkpoint.
4. **AdaptationPlan**: generated causal through-line, explicit narrative form
   and its reason, and per-beat `keep/compress/omit` with qualification and
   reason. Essential supported beats cannot be silently omitted. Invented
   illustration remains labeled separately from author facts. A stable
   character/location/prop identity and initial-state registry is required;
   the story need not include a hoodie presenter or fictional dramatization
   where a documentary/conceptual sequence is more honest.
5. **ChapterScript**: generated `scene-01...` with exact predecessor/successor
   links, narrative purpose, action, consequence, transition, source-beat
   references, narration and optional **explicitly invented** dialogue.
   Uses chapter-specific `max_scenes/max_words`, not the short 3–5-scene,
   45–125-word `Episode` or its compactor. Limits and accepted raw output
   are independently validated. A human must still check dialogue and
   narrative statements for fidelity; qualifiers attested verbatim in
   the review are required in referenced scene narration.
6. **Narration**: disabled by default. On explicit `kokoro_cpu=true`, calls
   existing `narration.synthesize_scene` once per narration/dialogue
   utterance, validates mono 24 kHz WAV geometry and measures sample-accurate
   scene durations. `hexgrad/Kokoro-82M` has version **label** `v1.0`;
   its resolved immutable model revision remains unverified. One voice
   per run; fictional dialogue uses that same voice, not an invented actor
   voice. Without audio, timestamps are `NO_AUDIO_ESTIMATE`.
7. **ChapterDirection**: generated *text* shots with exact scene/shot IDs,
   causal purpose and action, camera and environment, registered entity IDs,
   before/after entity states and contiguous scene/narration timing. Reject
   missing scenes, orphan shots, unknown entities, impossible state
   continuity, timeline gaps/overlaps. This validates a **text representation**,
   not the visual appearance or continuity of any rendered frame.

**Hard stop:** No image, keyframe, Wan, Modal, renderer, ffmpeg video assembly,
publisher, asset upload or private/public book output is invoked. Final labels:

- `CHAPTER_PREPRODUCTION_READY_UNREVIEWED`: genuine opt-in model stages and
  Kokoro WAVs finished, but content is still **not editorially approved**.
- `CHAPTER_TEXT_PLAN_READY_NO_AUDIO_UNREVIEWED`: generated chapter text plan,
  **no measured audio**, so narrated preproduction is **incomplete**.
- `TEST_DOUBLE_CHAPTER_TEXT_READY_UNREVIEWED`: at least one injected
  synthetic producer; this proves only plumbing, **never** real generation.
- `AWAITING_*` / `ABSTAINED_*`: explicit fail/hold states.

A later independent human reviewer must judge faithful substantive source
coverage, invented fiction disclosure, story causality, dialogue, audio pacing
and coherent action/continuity. No automatic validator can establish semantic
entailment or film quality. `publish_eligible` stays false.

## Exact future invocation (NOT authorized to execute by this patch)

All paths below refer to *existing private files*, not public actions or
book uploads. Freeze source scope and mining progress using
`docs/BOOK_PREVISUAL_EVALUATION.md`. The user must first verify a genuinely
**free, private, memory-capable execution route**; metered CPU credits do not
count as free. Do not attempt this command until that separate checkpoint.

**Model-free checkpoint and identity verification:**

```sh
clipper books chapter-evaluate \
  --artifact-root "$PRIVATE_ARTIFACTS" \
  --candidate-ref "$PRIVATE_ANALYSIS_REF" \
  --section-manifest "$PRIVATE_SECTION_MANIFEST" \
  --progress "$PRIVATE_COMPLETE_SCOPED_PROGRESS"
```

A private **chapter plan** is a strict JSON object. The fields below enable
explicit stages for the **later** privately authorized real run; they are
*not* current execution permission:

```json
{
  "schema": "clipper-chapter-preproduction-run-v1",
  "brief_cpu": true,
  "adaptation_cpu": true,
  "script_cpu": true,
  "direction_cpu": true,
  "kokoro_cpu": true,
  "max_scenes": 8,
  "max_words": 1000,
  "max_beats": 20,
  "max_shots_per_scene": 3,
  "max_new_tokens": 4096,
  "max_attempts_per_stage": 2,
  "max_runtime_seconds": 900,
  "voice": "af_heart"
}
```

Omit `--run-manifest` to perform **zero inference**. With the manifest,
`brief_cpu=true` only produces a brief and stops for a *separate*
human beat review. The review file is:

```json
{
  "schema": "clipper-chapter-human-beat-review-v1",
  "brief_sha256": "<exact generated brief identity returned by the CLI>",
  "reviewer": "<private human identifier>",
  "decisions": [
    {
      "beat_id": "<every brief beat ID, once>",
      "decision": "support_attested",
      "reason": "<specific source-support rationale>",
      "qualifier_quote": "<exact original context quote or empty>"
    }
  ]
}
```

At least two source-supported beats must be retained, each explicit.
A `reject` decision prevents use as an adapted sourced fact. The real
chapter preproduction run (after review and private route verification) is:

```sh
clipper books chapter-evaluate \
  --artifact-root "$PRIVATE_ARTIFACTS" \
  --candidate-ref "$PRIVATE_ANALYSIS_REF" \
  --section-manifest "$PRIVATE_SECTION_MANIFEST" \
  --progress "$PRIVATE_COMPLETE_SCOPED_PROGRESS" \
  --review-file "$PRIVATE_CHAPTER_BEAT_REVIEW" \
  --run-manifest "$PRIVATE_CHAPTER_PLAN"
```

The text adapter is the existing exact
`Qwen/Qwen3-1.7B` revision
`8d4744f9e13072f4920c326350fa81eedb74eae9`,
seed 42 with unchanged sampling settings. The chapter's per-stage
`max_new_tokens` is separate from the existing miner's 960-token limit.
Configuration enforces **2–20 scenes**, **130–2400 total words**, **2–24
candidate beats**, **1–4 shots per scene**, **1024–8192 output tokens per
call**, **1–2 attempts per each of four text stages**, and **60–900 seconds
chapter runtime**. With example plan: max **8 Qwen generation calls** +
**24 Kokoro utterance calls** (8 scenes × narration plus up to 2 dialogue).
These are upper bounds, not guarantees of adequate context or timely finish.
An external private host must enforce CPU, offline snapshot cache, memory,
timeout and disk caps; these cannot be inferred from a model label.

Every model attempt creates an **immutable pre-call marker**, preserves the
actual raw output (including rejected output), records failure reason codes
and blocks silent restart after interrupted/failed attempts. Validated
stage outputs are independently rechecked on cache replay. Script and
direction cache identities are input+model+stage+limit+provenance bound;
fake test-producer output is isolated from actual model caches. Fake and
actual WAVs also have separate caches. All private traces and source
quotations remain under `book-candidates/` (private artifact root).
CLI stdout emits only allowed statuses, counts, hashes, source-auth
labels and runtime metrics. Never expose source text, book titles,
section IDs, character names, generated raw content or exception strings
in public GitHub logs.

## Explicitly unproven

No real private book chapter was generated by this implementation. Synthetic
injected producers and public CI prove validation and fail-stop plumbing,
not that Qwen will produce a coherent, complete, supported or dramatic
chapter script within finite attempts, and not Kokoro intelligibility.
Model availability, prompt/context fit, execution cost, CPU memory/runtime,
source rights, TTS voice immutable revision, full-book completeness and
rendered character/prop continuity remain unverified. Visual proof requires
separately authorized multi-shot generation later; this stage has none.
