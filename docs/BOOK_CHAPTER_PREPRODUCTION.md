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
2. **ChapterBrief**: a **Qwen-generated** chapter argument from **every original
   canonical work unit in the exact scoped source order** (including units
   that mined *zero* candidates), plus source-gap markers for skipped/empty/
   unrecovered images and candidate annotations. Each beat cites an exact
   original `unit_id + character span + verbatim quote`; mined candidate IDs
   are optional annotations, never the source boundary. Unit-level
   `keep/compress/omit + reason` covers **every** unit and gap; omission
   cannot be implicit. The source ledger retains parent ordinal, section,
   anchor, original source path, evidence origin, canonical text/hash,
   recovery and source-permission warnings. `MINED_UNREVIEWED` remains
   explicit; an exact quote is **not semantic proof**.
3. **Human source-unit and beat review checkpoint**: a private human must
   explicitly review every unit's **and skipped marker's** proposed coverage
   decision, including omissions, then attest or reject each generated beat,
   its exact span and source qualifications. Rejecting any unit coverage
   blocks adaptation and requires a revised brief; fewer than two supported
   beats is valid abstention. This is **assisted**, not semantic proof or
   publishing approval.
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
  "schema": "clipper-chapter-preproduction-run-v2",
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

Omit `--run-manifest` for a zero-inference source-scope checkpoint. The
**ordinary `clipper books chapter-evaluate` CLI remains blocked** for real
providers, even with `brief_cpu=true`: it emits fixed
`PRIVATE_CHAPTER_BOUNDED_HOST_UNVERIFIED`. The only prepared real CPU adapter
entrypoint is `python -m clipper.books.chapter_host`, guarded by
`scripts/private_chapter_codespace.sh` and the independently reviewed
**single Codespace** contract in
[BOOK_CHAPTER_CODESPACE_RUNBOOK.md](BOOK_CHAPTER_CODESPACE_RUNBOOK.md).
This has not yet run on a real host. Synthetic callback stages stay separate
from real Qwen/Kokoro provenance. A run manifest cannot bypass the ordinary
CLI host block. The following review file is a private artifact **only after**
an independently inspected real brief is produced:

```json
{
  "schema": "clipper-chapter-human-beat-review-v2",
  "brief_sha256": "<exact generated brief identity returned by the CLI>",
  "reviewer": "<private human identifier>",
  "decisions": [
    {
      "beat_id": "<every brief beat ID, once>",
      "decision": "support_attested",
      "reason": "<specific original unit/span support rationale>",
      "qualifier_quote": "<exact original source-unit text quote or empty>"
    }
  ],
  "unit_decisions": [
    {
      "source_id": "<every original source unit or gap, once>",
      "decision": "keep",
      "review": "accept_coverage",
      "reason": "<why this treatment or explicit omission is acceptable>",
      "qualifier_quote": "<exact original unit text quote or empty>"
    }
  ]
}
```

Every work unit and unrecovered source marker requires its own matching
`unit_decisions` entry and must use the brief's exact decision
(`keep/compress/omit`); skipped/unrecovered gaps must be `omit` with a
reason. A reviewer can mark `reject_coverage` to fail closed rather than
silently approve an omission. All beats must preserve original text span
offsets and exact quotes. At least two source-attested beats must be
retained; a rejected beat cannot become an adapted sourced fact. The real
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
source beats and original source-unit/gap coverage entries**, **1–4 shots
per scene**, **1024–16384 output tokens per call** (the active model profile's output bounds; or `"auto"`, see `BOOK_MODEL_PROFILE.md`), **1–2 attempts per
each of four text stages**, and **60–900 seconds chapter runtime**.
No source unit is truncated. The private source ledger has an absolute
**163,840-byte** bound derived from 4× the pinned 40,960-token model
context; it is **not a token-equivalence claim**. Up to **64 original
work units and unresolved source markers** are exposed to the reviewer.
For model-free tests the conservative full UTF-8 request plus 512-token
reserve and output budget still rejects excessive prompts. On the separately
approved real CPU host, the pinned Qwen tokenizer checks the **exact full
chat template and all output tokens before loading weights or reserving
the attempt**. Failure is a stop with no truncation, source-scope narrowing
or short Episode fallback. With example plan: max **8 Qwen generation calls** +
**24 Kokoro utterance calls** (8 scenes × narration plus up to 2 dialogue).
These are upper bounds, not guarantees of adequate context or timely finish.
An external private host must enforce CPU, offline snapshot cache, memory,
timeout and disk caps; these cannot be inferred from a model label. Direct
Python `ChapterPlan` construction enforces the same finite limits, attempts,
types and output timeline budgets as a strict JSON manifest before callbacks.

Every model attempt creates an **immutable pre-call marker**, preserves the
actual raw output (including rejected output), records failure reason codes
and blocks silent restart after interrupted/failed attempts. Validated
stage outputs are independently rechecked on cache replay. Stage request directories are input+model+stage+limit+provenance bound,
while the returned **text artifact identities** also bind exact raw and
validated output digests and an explicit validator contract. Replay rechecks
both artifacts and the exclusive attempt owner; changed raw, changed validated
payload or absent/partial records fail closed. Human review binds the exact
verified brief artifact, and the final result binds all four verified text
artifact identities. This does not protect a root controlled by a hostile
actor who can rewrite every artifact and its independent anchors. Fake test
producer output remains isolated from actual model caches. Fake and
actual WAVs also have separate caches. Repeated narration/dialogue text
is synthesized once per identical voice/text request and reused; every
utterance separately records scene ID, sequence, narrator/character ID,
fiction/narration kind, request hash, WAV hash and measured duration. All private traces and source
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

### Execution status and source provenance

Ordinary chapter CLI text/Kokoro provider calls remain **blocked**.
A separate explicitly guarded Codespace integration is prepared, not
provisioned or runtime-proven. Its shell wrapper obtains the live approved
Codespace metadata, enforces a single creation-time deadline with a hard
process timeout for inspection, mining, both briefs, human review and both
resumes, and reserves cleanup time. `max_runtime_seconds` still constrains
the **resulting story timeline**, not inference wall-clock. Injected
model-free producers remain available for synthetic tests; historical mining
artifacts without explicit producer origin are `LEGACY_LOCAL_CPU_UNKNOWN` or
`LEGACY_UNKNOWN`, not retrospective proof of CPU inference. Chapter final-result
identity binds the scoped source and mining provenance, all four stage keys,
human-review digest, exact measured WAV hashes, narration report reference and
per-utterance PCM sample offsets. An output marked `MEASURED_WAV` requires
validated readable PCM frames. Editorial truth and character continuity still
require human review; no images/video are generated here.

Reviewed KEEP/COMPRESS source units retain at least one beat through adaptation
and script coverage, regardless of whether their last beat was designated
essential; changing that disposition requires a new brief and human review.

## Argument stage input checks (model-free)

- The argument may cite at most 24 `source_links`. More is rejected with the code
  `TOO_MANY_LINKS` and the retry prompt names the limit.
- When a retry follows invalid citations, the links that already passed are sent back as
  bare id and role (never a quote) with the instruction to keep them. This is computed
  from the rejected reply alone, so replay stays deterministic.
- One outer wrapper object is removed when it is the only key and is named `argument`,
  `answer`, `response`, `result` or `output`. Double wrappers, wrappers with other keys and
  other names are still rejected. No text-level (code fence) stripping is done.
- The argument request text changed, so request hashes moved. Journals of failed attempts
  written before this change cannot be resumed under the new code.

## Batched argument input (optional, model-free)

A small model follows a long list limit badly when the whole chapter is in one prompt.
When the argument stage config sets `argument_batch_tokens` (an integer from 512 to 16384,
or 0 / absent for off, which is the default), the argument is built in two steps:

1. **Batches.** The text units are cut into contiguous windows by the exact token size of
   each rendered request, measured with the pinned tokenizer of the active model profile.
   A window grows until the next unit would pass the budget. No fixed unit count is used,
   so window edges follow the tokenizer and the budget. A unit that alone passes the
   budget gets its own window. Gap markers are not batched. At most 32 windows.
   Each window is its own cached stage (`hybrid_argument_batch`) and returns `points`
   (short sentences) and `source_links` (ids from that window only, checked in code).
2. **Merge in code, then one small argument call.** Each window contributes at most
   `ceil(24 / windows)` distinct ids (first seen), so a window that over-cites cannot crowd
   out another. If windows outnumber 24, an even spread is taken. The final argument
   call sees only the chosen units plus the batch points, and returns the same
   `hypothesis`, `questions`, `source_links` as before, validated by the same code
   (24-link cap, degeneracy gate, wrapper removal), except that it may cite only the
   chosen units it was shown. A batch note is kept only while one of its window's ids is
   still chosen, and all notes together stay under 12,000 characters. The notes are
   labelled in the prompt as unverified model text, never instructions. The summary
   reports `argument_batch_attempts` (model calls used by the batches). Audits and
   reconcile still see the whole chapter.

Replay: the partition and merge are fully determined by the ledger, the tokenizer and the
config, so a second run replays every batch from cache. The new request text changes
request hashes only for configs that turn batching on; configs without the field are
byte-identical to before. What it changes about the argument: it is a summary built from
per-part notes, so a thread that only shows up across parts can be missed in the first
pass; the audit and reconcile stages still check against the full chapter.

The script-phase config must carry the same `argument_batch_tokens` as the brief-phase
config (the argument settings already have to match for replay). If it does not, the
replay stops with a ChapterStop before any model call.
