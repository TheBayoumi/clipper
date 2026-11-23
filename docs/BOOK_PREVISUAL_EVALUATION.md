# Private book → previsual text evaluation (assisted, not production)

This is a **private evaluation-only path**, connected to the real candidate ingest,
faithful ordered-text projection, bounded whole-book miner, canonical episode
scripter, opt-in CPU Kokoro narration and a generated **TEXT-only** director.
It never invokes the legacy editorial catalog, prewritten script/recorded WAV
substitution, visual producers, keyframes, image/video generation, Modal, ffmpeg
assembly, review publishing or any public artifact upload.

## Required private inputs and invocation

Use the **existing private artifact root** and exact ingest analysis reference,
created by the ordinary `clipper books ingest` command (or its Python
`BookOrchestrator.ingest_candidate` equivalent). No uploads or new book sources.

**Default model-free inspection** of an already-complete, private, pinned mining
ledger and export:

```sh
clipper books evaluate \
  --artifact-root "$PRIVATE_ARTIFACTS" \
  --candidate-ref "$PRIVATE_ANALYSIS_REF" \
  --progress "$PRIVATE_COMPLETE_PROGRESS"
```

The source-bound evaluation stores a private `mined-candidates.json` in
`book-candidates/evaluations/<progress-fingerprint>/` and returns
`AWAITING_HUMAN_SOURCE_SUPPORT_REVIEW`. Stdout contains **only hashes, stage
status and counters**; it never outputs a source quote or generated creative
text. Progress must correspond to a **complete** mining run for the exact
candidate/edition, with original raw source, ingest ledger, candidate quote
occurrences and private unreviewed export still present. Missing, incomplete
or mismatched mining records fail closed. This is not yet a partial-book
editorial-completeness guarantee.

For a **fresh real CPU** evaluation, only after free/private execution
approval, omit `--progress` and provide a manifest with `mine_cpu: true`.
This explicitly invokes the existing `BookOrchestrator.mine_candidate`.
A bounded `max_new_units` caps *new* mining calls. **A cap is not a scope**:
unresolved included units fail-stop, and no complete export exists until
all included units finish.

To freeze one real-source section without forging a new mini-book, derive
the exact `section_id` and unit counts from the parent source's
`whole_book.plan_for(book)`. Freeze a **private immutable section scope**
manifest (`$PRIVATE_SECTION_MANIFEST`) with this exact JSON structure:

```json
{
  "schema": "clipper-book-section-evaluation-scope-v1",
  "book_fingerprint": "<original full BookText fingerprint>",
  "source_fingerprint": "<original source fingerprint>",
  "ingest_stage_key": "<original faithful ingest stage key>",
  "parent_plan_fingerprint": "<whole parent plan fingerprint>",
  "section_ids": ["<exact canonical section_id in source order>"],
  "parent_total_units": 27,
  "included_units": 5,
  "excluded_units": 22
}
```

The displayed counts are illustrative **only**; the actual manifest must
contain the exact verified parent and section counts. Supply
`--section-manifest "$PRIVATE_SECTION_MANIFEST"` on **both mining and
replay**. It is pinned by raw SHA-256 in the scoped private ledger, and
its original bytes are privately retained immutably. Plan/cache/review
identities include the selected parent work units without renumbering,
replacing source fingerprints or changing evidence anchors. Scoped
`complete` means **every included unit** completed; the ledger and
export explicitly set `parent_full_book_complete: false`. It does NOT
mean the full book is complete. Tampered or partial scope ledgers must fail.


For an **offline synthetic** integration regression only,
`--responses "$PRIVATE_SHA256_KEYED_RESPONSES"` feeds the existing recorded
whole-book miner. These are *recorded mining responses*, not manually
substituted scripts, direction outputs, or WAVs. This mode requires no model
weights or private API access.

## Human source-support checkpoint (no automatic semantic approval)

Provide a private JSON file `$PRIVATE_HUMAN_REVIEW`:

```json
{
  "schema": "clipper-book-human-source-support-review-v1",
  "progress_sha256": "<exact SHA256 of private progress JSON bytes>",
  "reviewer": "<human reviewer identifier>",
  "selected_candidate_id": "<exact mined candidate ID or null>",
  "decisions": [
    {
      "candidate_id": "<each mined candidate ID, once>",
      "decision": "accept_direct",
      "reason": "<why this exact quote supports this narrow claim>",
      "qualifier_quote": "<exact excerpt of retained source context, or empty>"
    }
  ]
}
```

Every mined candidate needs an explicit `accept_direct` or `reject` decision
and a substantive reason. Exactly one directly supported candidate may be
selected; selecting null is valid **only if all are rejected**, and stops
without script/audio/direction. The reviewer must examine duplicates, thin
material, caveats and counterexamples. This is **human-assisted evaluation**
and never promotes `MINED_UNREVIEWED` to editor-approved/publishable. The
claimed support is a **human assertion**, not proof of semantic entailment.

Private selected context includes original source SHA, exact quote, evidence
anchor/section/source path, occurrence offsets, origin, recovery and source
authorization status, surrounding excerpt, human reason and qualifiers. It
enters the **evaluation-specific Qwen script request and cache identity**, not
the legacy `approved_facts` field.

## Separate explicit CPU permissions

The optional `--run-manifest "$PRIVATE_MANIFEST"` is a separate private local
decision, **not provided or authorized by these code changes**:

```json
{
  "schema": "clipper-book-private-cpu-evaluation-manifest-v1",
  "mine_cpu": false,
  "script_cpu": true,
  "direction_cpu": true,
  "kokoro_cpu": false,
  "max_new_units": 1,
  "pillar": "self_improvement",
  "voice": "af_heart"
}
```

`false` is the default for every model flag when this manifest is omitted.
No CLI option silently invokes weights. To evaluate an already mined ledger,
keep `mine_cpu: false` and provide `--progress`. The **same canonical
scripting.generate_episode** permits at most **3 Qwen script attempts**.
Each provider call reserves a **durable started/attempt marker before**
invocation, including calls that subsequently throw, return nontext, are
interrupted or fail while saving raw evidence. Every textual completion
is privately saved before validation. The canonical script audit records
rejections and complete-sentence edits. A failed or ambiguous attempt
cannot be silently reinvoked on replay; a separately authorized recovery
process would be needed. Immutable script evidence is independent of
per-invocation attempts and audio mode; measured WAV reports and estimated
timing reports carry distinct content-addressed identities.

When `kokoro_cpu: true`, **actual Kokoro-generated WAVs** (one per scene)
are obtained by `narration.synthesize_scene` and independently validated.
For each, measured frames, 24 kHz rate, SHA-256, voice and duration are
recorded. With Kokoro off, estimated narration timing is clearly marked
`estimated_no_audio`, not a completed audio run. Kokoro identity is
`hexgrad/Kokoro-82M`, the package uses `v1.0` as a revision **label**:
**no immutable resolved model commit has been verified**. This remains a
reproducibility gap.

When `direction_cpu: true`, Qwen produces **new JSON text for each generated
script scene**: purpose, concrete subject/action/environment, visual
mechanism, continuity, and exact evidence candidate ID. The validator
checks scene count/order/IDs, nonempty bounded fields and narrator-aligned
timing (actual WAV when available; estimates otherwise). There are at most
**2 Qwen text-direction attempts**, each raw response stored privately.
Cache replay validates the previously generated shots and their timing
without any second inference. A new narration mode has a distinct
duration-specific shot plan key; it must not overwrite the other mode.
No image/video generator is called. Canonical Qwen script feedback and
provider stdout/stderr are retained **only in private diagnostics** on this
evaluation path; the public CLI prints safe statuses/hashes/counts.

Only a reviewer can judge whether the generated script conveys the specific
mechanism honestly, whether narration is intelligible, and whether actual
direction is concrete rather than vague. The terminal status
`PREVISUAL_TEXT_READY_UNREVIEWED` is **not** creative approval.

### Exact later private invocation (not authorized now)

```sh
clipper books evaluate \
  --artifact-root "$PRIVATE_ARTIFACTS" \
  --candidate-ref "$PRIVATE_ANALYSIS_REF" \
  --section-manifest "$PRIVATE_SECTION_MANIFEST" \
  --progress "$PRIVATE_COMPLETE_PROGRESS" \
  --review-file "$PRIVATE_HUMAN_REVIEW" \
  --run-manifest "$PRIVATE_MANIFEST"
```

The existing pinned text model is `Qwen/Qwen3-1.7B` at exact revision
`8d4744f9e13072f4920c326350fa81eedb74eae9`, using CPU, seed 42,
`max_new_tokens=960`, sampling temperature 0.7/top-p 0.8.
Worst-case new inference **caps** for one invocation: 0–8 mining Qwen calls
(as explicitly granted in the manifest), up to 3 Qwen script calls, up to
2 Qwen direction calls, and at most 5 Kokoro scene calls. With all flags
off: **0 model calls**. No automatic upgrades, account changes, GPU or paid
compute are supported. A valid *free*, memory-capable **private CPU runner**
with isolated caches and an externally enforced wall-clock/memory/disk
limit must be verified **before** any real-source execution. The API cannot
interrupt a stalled model call by itself. These tests do not validate its
availability, zero cost, resource limits, semantic quality or real-book
copyright/privacy constraints.

Public CI exercises only **synthetic texts and injected fake producers**.
Never check in private manifests, private review decisions, raw real-book
text, script/audio/direction outputs, job logs or source derivatives.
The original public synthetic CPU smoke trigger is already consumed and
must never be rerun by this evaluation workflow.
