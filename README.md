# Clipper

Clipper is a rights-gated autonomous multimodal editor for producing short-form clips from explicitly authorized long-form video. Production execution is quality-derived: the system may emit many clips, one clip, or zero clips. It does not fill a requested quota with weaker content.

## Production architecture

```text
explicit authorized target video(s)
        ↓
highest-available source media
        ↓
canonical grounding
(transcription → alignment → diarization)
        ↓
visual timeline / VLM evidence
        ↓
MultimodalEvent
        ↓
SemanticCore
        ↓
NarrativeEnvelope
        ↓
deterministic FeasibleDeliveryWindow[]
        ↓
QualityMoment
        ↓
VisualStrategy
        ↓
EditPlan
        ↓
1080x1920 render
        ↓
technical QC + multimodal editorial QC
        ↓
accepted production artifacts
```

The production graph has no editorial output quota, fixed shortlist count, fixed concept count, fixed variant count, hard-coded hook menu, or campaign scoring-weight table. A candidate reaches rendering only when the source evidence supports a complete, feasible, policy-compliant moment.

## Source targeting and rights

`clipper run` never performs implicit source discovery. Every production brief must use `targets.mode: explicit` and provide the exact authorized videos. Rights authorization is a separate gate and every target channel must be authorized. Production authorization is enforced by exact video ID; channel authorization constrains rights but never serves as a fallback that admits unlisted videos.

Example:

```yaml
campaign_id: example-campaign
title: Example clipping campaign
objective: Find every genuinely worthwhile, self-contained short-form moment.
language: en
region_code: US

targets:
  mode: explicit
  videos:
    - video_id: REPLACE_WITH_AUTHORIZED_VIDEO_ID
      url: https://www.youtube.com/watch?v=REPLACE_WITH_AUTHORIZED_VIDEO_ID
      channel_id: UC_REPLACE_WITH_AUTHORIZED_CHANNEL_ID

rights:
  confirmed: false
  authorized_channels:
    - UC_REPLACE_WITH_AUTHORIZED_CHANNEL_ID

content_constraints:
  min_clip_seconds: 20
  max_clip_seconds: 45

attribution_required: true
watermark_text: null
watermark_url: null
required_hashtags: []
posting_requirements: []

acceptance_policy:
  source_segments:
    allow: [editorial_content]
    forbid: [advertisement, sponsor_read, promo, intro, outro, housekeeping]
    unknown: escalate
    safety_buffer_seconds: 0.25
  branding:
    supplied_campaign_assets_allowed: true
    foreign_logos: forbid
    minimum_confidence: 0.75
  generated_media:
    synthetic_visuals: escalate
  portrayal:
    negative_creator_portrayal: escalate
  language:
    on_screen_text: en
  editorial:
    require_standalone_context: true
    require_resolved_ending: true
    minimum_boundary_confidence: 0.75
```

Set `rights.confirmed: true` only after the source is actually authorized. Clipper does not bypass DRM, authentication, paywalls, private-video permissions, or platform restrictions.

A separate `clipper discover` command exists only for source research outside production execution. Discovery output is not automatically promoted into a production run; selected videos must be written explicitly into the brief and authorized first.

## Autonomous editorial contract

The structured editorial model has four production task families:

- `source_hazards` — identify sponsor/promo/branding/policy hazards from source evidence.
- `semantic_cores` — identify independent semantic cores without a predetermined topic or vocabulary list.
- `narrative_envelope` — determine the context and resolution required for a core to stand alone.
- `quality_windows` — rank feasible delivery windows that preserve the required narrative envelope.

The prompt/schema contract prohibits predeclared topic, hook, emotion, numeric, or domain-specific lexical templates. Unknown production task families fail closed.

Duration feasibility is deterministic. The solver constructs valid delivery windows before editorial ranking, so an `EditPlan` cannot be accepted merely because a model asked for an under-duration span.

## Multimodal editing and rendering

- Full-frame `1080x1920` output with no blurred duplicate background.
- Source-resolution-first portrait crop; no unnecessary digital zoom.
- Speaker-aware framing uses face/speaker evidence rather than generic motion, so gestures and hands do not drive the crop.
- Source-camera cuts remain hard composition boundaries; virtual-camera movement is deliberately stabilized.
- Word-synchronized ASS reveal captions use the final clip-local word timeline.
- First-caption alignment is audited and fails QC on mismatch.
- Campaign watermark/branding policy is applied before release.
- Synthetic or generated visual media is policy-gated; campaigns may forbid it entirely.
- Technical QC validates decode, geometry, audio, captions, tracking evidence, and required sidecars.
- Final editorial QC is multimodal and can reject an otherwise technically valid render.

## Models and execution

The production model plan is resolved explicitly and recorded in run evidence. The open-model Modal runtime provides:

- transcription
- alignment
- diarization
- structured editorial inference
- vision / large-vision review
- schema and Hugging Face access smoke tests

Default production execution uses the deployed Modal pipeline when the resolved plan requires Modal. `--allow-local-lite` is an explicit opt-in to the smaller local profile; it is not silently substituted for production models.

## Content-addressed execution

Grounding and editorial stages are keyed by source, contract, model/revision,
decoding parameters, dependencies and relevant policy. Cache reuse requires a
**canonical, type-preserving identity match**: JSON `false`, integer `0` and
float `0.0` are distinct. A PASS also requires a complete terminal record and
matching output fingerprint; valid JSON null, false and empty outputs are
cacheable.

Local `DagStore` execution uses atomic records and OS-managed advisory write
locks. The production coordinated path uses a paired authoritative
`read_committed` / `publish_terminal` contract: the serialized lease owner
fences publication by owner, attempt and expiry. It does not treat a worker's
dirty Volume mount or an unacknowledged commit as confirmed durable evidence.
The older `reload` / `commit` coordinator callback path remains for
compatibility but does **not** independently guarantee fenced publication.
Lost acknowledgements and cancellations remain visible rather than being
silently returned as success. See [DAG execution and recovery](docs/dag-execution.md)
for responsibilities, locking errors and runtime limits.

`--resume RUN_ID` reuses matching content-addressed artifacts from an
interrupted run. `--fresh-inference` creates an empty cache root so live
model evidence cannot be satisfied by an existing cache entry.

Each run persists model identities, source hashes, cache events, funnel/yield
evidence, rejections, render attempts, QC evidence and output hashes in its
artifact directory.

## Bounded book-candidate and faithful text ingest

The separate book front half provides exact-title discovery from a **small,
explicitly reviewed candidate registry**, bounded acquisition, source-ordered
text extraction and versioned provenance/rejection reports. It is **not**
general internet book discovery, authentication of a publisher edition,
rights approval, a guarantee of complete text or title-to-video production.
The existing registry and source pins remain unchanged.

```bash
clipper books candidates --title "Atomic Habits"
clipper books ingest --title "Atomic Habits" --artifact-root artifacts/books
```

`books ingest` writes private, locally controlled evidence. It reports
`incomplete_recovery` with exit code **2** when recovery or unreviewed image
evidence is incomplete; this must not be treated as a full-text success.
Rights/provenance and human observation review are separate gates. Repository
fixtures are synthetic and do not contain private book sources or research
sidecars. Offline evidence import is a separate owner-controlled action, not
a prerequisite for inspecting the code.

An ingest result now exposes `analysis_ref_path`: a private versioned,
immutable handle to that exact extraction and dossier, not a request to copy
source text into the legacy `book-text/<book-id>.json` path. Whole-book analysis offers mutually exclusive **recorded** and explicitly
opted-in **local CPU** modes. Recorded mode uses a separately prepared local
response JSON for the exact work plan; it never initializes inference:

```bash
clipper books ingest --title "Atomic Habits" --offline --artifact-root artifacts/books
clipper books mine --candidate-ref artifacts/books/book-candidates/analysis-refs/atomic-habits/EXACT_STAGE_KEY.json \
  --book-id atomic-habits --responses /local/recorded-responses.json \
  --artifact-root artifacts/books
```

The first command can exit **2** for `incomplete_recovery`; it still emits
the exact `analysis_ref_path` on successful artifact creation. Substitute
the actual returned path, not a guessed or most-recent file. The recorded
`--responses` command never fetches content or loads a model. The additional
explicit CPU mode uses the **same exact reference** and pinned Qwen revision:

```bash
python -m pip install -e ".[niche-models]"
clipper books mine --candidate-ref artifacts/books/book-candidates/analysis-refs/atomic-habits/EXACT_STAGE_KEY.json \
  --book-id atomic-habits --local-cpu --max-new-units 8 \
  --artifact-root artifacts/books
```

`--local-cpu` reuses the exact pinned local Hugging Face snapshot if
complete, passing its local directory (not the repository ID) into
Transformers 4.57.3. To pre-cache explicitly on an owner's network-enabled
machine, call `huggingface_hub.snapshot_download` with model
`Qwen/Qwen3-1.7B` and full revision
`8d4744f9e13072f4920c326350fa81eedb74eae9`; then run the command
with `HF_HUB_OFFLINE=1`. Missing/incomplete offline weights become a
bounded failed unit, never another revision or remote inference. Without
offline mode the first uncached unit may download public pinned weights.
This remains **local CPU-only inference**, not a hosted API or GPU service.
A separate ~2 GB RAM environment could not mmap the first 3.44 GB shard;
allow substantially more memory (roughly 8–16 GB or more) and potentially
hours-to-days for a book. The limit counts **new producer
attempts per invocation**, not total planned units; repeat the same command
to resume without constructing a response-map file. Cache-only replays do not
initialize the model. Missing, failed and interrupted units remain incomplete
(exit 2); only actual valid empty model replies count as zero-candidate work.
The generated answers, including image/research provenance, are private,
bounded and unreviewed. This code milestone is tested with injected fake
completions only; actual Qwen execution and semantic lesson quality remain
separate acceptance gates. Per-unit recorded response
keys must match the canonical SHA-256 of the requested field/provenance.
The new path accounts for all ordered native and image evidence units across
bounded 32-unit windows, with separate statuses for zero candidates, gaps,
failed and unresolved work. Missing replies yield an incomplete progress
report (exit 2), not a partial catalog claimed complete. Successful unit
DAG records are reusable when later responses are supplied; new mappings
do not invalidate previously used identical completions. It produces
private, separate `MINED_UNREVIEWED` candidate and progress artifacts;
no editorial claim is approved, no lesson count is targeted, and no
downstream stage runs.

A separate [public synthetic CPU smoke staging guide](docs/book-local-cpu-smoke.md)
describes the dormant, path-gated GitHub runner test. It has **not** executed
the model and requires a separate one-file trigger commit after review.

The legacy explicit-owner-file ingest and `books mine --book-id ...`
interfaces remain available. See [whole-book recorded/local analysis and handoff](docs/book-whole-analysis.md),
[candidate acquisition and faithful ingest](docs/book-candidate-acquisition.md),
[offline ingest and legacy mining limits](docs/book-pipeline-phase3.md),
[recorded narration and unported stages](docs/book-pipeline-phase1b.md) and
[creative review trust boundaries](docs/book-creative-review.md).
Direction, visuals, assembly and review remain unported/approval-gated; no
book-video publication acceptance is implied.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp campaign.example.yaml campaign.yaml
```

Install only the inference extras required by the execution environment, or install `.[open-models]` for the complete open-model worker dependency set. Modal orchestration uses `.[modal]`.

## Commands

Validate an explicit production brief:

```bash
clipper validate --brief campaign.yaml
```

Run production:

```bash
clipper run --brief campaign.yaml --artifact-root artifacts
```

Plan without rendering:

```bash
clipper run --brief campaign.yaml --artifact-root artifacts --no-render
```

Force fresh inference:

```bash
clipper run --brief campaign.yaml --artifact-root artifacts --fresh-inference
```

Resume a compatible interrupted run:

```bash
clipper run --brief campaign.yaml --artifact-root artifacts --resume RUN_ID
```

Optional non-production source research:

```bash
clipper discover --query "podcast topic" --channel-id UC_CHANNEL --limit 10
```

Evaluate a private cross-domain acceptance corpus:

```bash
clipper benchmark --manifest acceptance/corpus.yaml --output benchmark.json
```

## Software quality gates

```bash
ruff check .
ruff format --check .
mypy
pytest
```

The **95% minimum applies to the `clipper` package**, not to all repository
scripts combined: `pytest` uses `--cov=clipper --cov-fail-under=95`, and CI
separately checks the measured package result with
`coverage report --fail-under=95`. GPU-driving scripts have a separate
CPU-fake coverage run and threshold; see
[GPU-driving script coverage](docs/gpu-worker-script-coverage.md).
CI runs these checks on Python 3.11 and 3.12. No measured coverage result
certifies live model execution or creative acceptance.

Production acceptance additionally requires exact-head deployment evidence, current model/schema smoke tests, a current end-to-end render, technical QC, multimodal final QC, and review of the actual MP4. Historical renders do not prove the current head.

## Original niche video (historical open-weight prototype)

**Historical prototype status — not acceptance of this cleanup branch.**
**Editorial acceptance: REJECTED for production visuals (October 24, 2026).**
The basic Pillow-designed Pip mascot, two-pose mouth animation, and static book
title cards are technical proofs **not professional cartoon content**. Successful
encoding or end-to-end Qwen3 + Kokoro generation must not be described as ready
for publishing. Book drafts now stop before model inference unless
`allow_rejected_prototype=true` is explicitly selected in Actions for archived
technical reproduction. See `plans/professional-book-video-creative-rebuild.md`.


Historical prototype branch: `feat/unified-niche-tts` (not the active
`review/unified-tts-cleanup` implementation branch). This independent
original-content workflow does not
use a Reach/Whop campaign or another creator's podcast. Configure
`niches/ai-workflows.yaml` (example English US Kokoro voice: `af_heart`).

```bash
python scripts/niche_video.py --config niches/ai-workflows.yaml --validate-only
```

Both the script generator (`Qwen/Qwen3-1.7B`, pinned revision) and the speech
generator (`hexgrad/Kokoro-82M` v1.0, Apache 2.0) run from downloaded open
weights, **without an OpenAI API key or a paid inference provider**.

The GitHub Actions workflow `Original niche video draft (open models)`
automatically checks the configuration and performs an FFmpeg smoke test. A
manual dispatch with `voice_smoke=true` generates a **real narrated Kokoro
MP4**, whereas `generate_draft=true` requests a full Qwen3-written, Kokoro
narrated episode. All outputs remain private Actions draft artifacts awaiting
human audio/visual and factual review.

Initial cold run requires downloading approximately 4 GB for Qwen3 and
approximately 360 MB for Kokoro, plus Python dependencies. GitHub's Ubuntu
CPU runner handles model inference; these models are not claimed to match
proprietary naturalness without an audio comparison. GPU acceleration and
more expressive models (Chatterbox/Qwen3-TTS) are potential subsequent
providers, not silently substituted for Kokoro. The content hash records
model IDs, revisions, prompts and voice parameters. Publication stays off.

## Publication boundary

Clipper stops at production artifacts and QC evidence. It does not automatically upload to TikTok, Instagram, YouTube Shorts, Whop, or another campaign platform. Publishing/submission is a separate explicitly authorized action after the final media and campaign/account requirements are reviewed.


## Modal infrastructure qualification (offline only)

The isolated CPU-only 64 KiB synthetic persistence and lifetime qualification
lives in `src/clipper/modal_infra_qualification.py`, with offline regressions
in `tests/test_modal_infra_qualification.py`. Read
[the scope and ownership limits](docs/modal_cpu_infra_offline_qualification.md).
This code is **not a live Modal/provider approval** and has no automatic CLI,
model, deployment or chapter execution entry point. Codespace chapter recovery
remains paused and blocked; it is not a prerequisite for this independent
infrastructure experiment.


## Offline Modal CPU infrastructure candidate (UNQUALIFIED)

The isolated, model-free 64 KiB Modal SDK 1.5.5 infrastructure harness is
documented in [offline qualification](docs/modal_cpu_infra_offline_qualification.md).
It has no live driver or account authorization. Candidate 2215 binds SDK
environment to intent, propagates public waiter cancellation to the owner STOP
latch, retains identified but not terminally observed sandboxes for reconciliation,
and treats terminal JSON as **unconfirmed file-only evidence**. Only a live
owner-returned acknowledgement after fsync may report local SUCCESS; provider
removal is always UNKNOWN. No CI or provider clearance is inferred.

## Explicit Modal CPU infrastructure driver (offline candidate)

`clipper.modal_infra_driver.run_infrastructure_experiment` is the optional
**synchronous Python injection boundary** for the model-free 64 KiB synthetic
exercise. It accepts only a separately supplied SDK/client and explicit Intent,
owns POSIX signal/ledger/event-loop cleanup, and has no CLI or account bootstrap.
The qualified owner module is unchanged; this driver is **UNQUALIFIED** until
independent fake-SDK E2E, interruptions, exact-SHA CI and final review.
See [driver prerequisites and limits](docs/modal_cpu_infra_offline_qualification.md).
No live provider invocation or cost authorization exists.


### Driver host-exception and per-signal handoff boundary (unqualified)

Unexpected host-handler exceptions while the Python Runner is active retain
the original owner for reconciliation on that **same Runner**, then propagate.
SIGINT/SIGTERM ownership transfers separately when each former disposition
is restored, not at one atomic joint handoff. See the qualification document
for inert subprocess interruption tests and remaining limitations.


### Explicit host signal contract (driver candidate UNQUALIFIED)

The driver now fails closed **before** ledger/SDKBackend/context/handler/Runner
work unless a POSIX main-thread host has inspectable dispositions and
unblocked SIGINT/SIGTERM. All *other catchable* signals must be SIG_DFL or
SIG_IGN (no custom callable, None or unknown values); owned SIGINT/SIGTERM
may additionally have restorable callable handlers. Inspection failures reject
without changing host handlers or masks. The host must keep this state stable.
The earlier arbitrary-host-handler exception/retry guarantee is **withdrawn**.
See [the exact host contract](docs/modal_cpu_infra_offline_qualification.md).
No live provider approval or account bootstrap is provided.


## Public synthetic book-chapter rehearsal (offline candidate)

An **opt-in, model-free, test-only** harness at
`tests/book_public_rehearsal.py` reuses existing public synthetic EPUB
fixtures and the real selected-source exporter/importer, chapter hybrid
evaluator, validators, stage evidence and replay. It is **not** the Modal
media/campaign pipeline, a chapter-host identity bypass, or real chapter
analysis. Two isolated tracks stop at the real human-review checkpoint and,
separately, use an explicitly **synthetic test review** to validate downstream
adaptation/script/fake PCM/direction. Neither track becomes publishable.

The CLI requires a checked-out exact Git SHA and a fresh absolute local output
root; the harness blocks dangerous imports, model/provider entrypoints,
network connections, subprocess launching during the scenario, source mining
and publication attempts. It does **not** initialize Modal, clients, accounts,
hosts, models or credentials, and never deploys or spends. It produces a
machine-readable synthetic report with provenance and artifact hashes, callback
counts, adversarial rejection results, wall time and peak RSS.

See [model-free chapter rehearsal contract](docs/BOOK_MODAL_PUBLIC_REHEARSAL.md)
for the explicit clean-checkout invocation (`PYTHONPATH=src:tests`),
application-level guard limitations, verified minimal local dependencies and
the **separate, unapproved** restricted CPU-only Modal Function proposal.
Independent external verification on `6162ca65333f67445fa0bb8003f8198fe95de376`:
12 new tests PASS, Ruff lint/format PASS, mypy PASS, and standalone Python
3.11 + PyYAML 6.0.3 CLI PASS (0.22127 s, Linux peak RSS 32,308 KiB).
No pytest, media stack or model/Modal library was needed for that standalone
run. The separate CI `45592640560` has not yet been confirmed fully
successful; the unchanged 95% gate and fresh read-only final review remain
pending. There is no remote launcher, resource allocation or spending
approval. The previously qualified offline Modal infrastructure owner/driver
remain unchanged. This harness is **UNQUALIFIED**.


### Synthetic rehearsal support isolation (UNQUALIFIED)

The standalone harness uses `tests/public_book_fixture_support.py`, not
broad test modules that transitively initialize the CLI, media pipeline or
download dependencies. A fresh-process check confirms its import graph;
the CLI rejects preloaded forbidden modules and reports **scoped** Python
guard evidence, not a kernel-level sandbox guarantee. No production source,
host authority, Modal runtime or deployment has been changed.
