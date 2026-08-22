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

Grounding and editorial stages are keyed from source/model/contract inputs. A changed source, model identity, schema contract, or relevant stage input invalidates the affected cache rather than reusing stale output.

`--resume RUN_ID` reuses matching content-addressed artifacts from an interrupted run. `--fresh-inference` creates an empty cache root so live model evidence cannot be satisfied by an existing cache entry.

Each run persists model identities, source hashes, cache events, funnel/yield evidence, rejections, render attempts, QC evidence, and output hashes in its artifact directory.

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

Pytest enforces a repository-wide coverage floor of **95%**. CI runs the quality gates on Python 3.11 and 3.12.

Production acceptance additionally requires exact-head deployment evidence, current model/schema smoke tests, a current end-to-end render, technical QC, multimodal final QC, and review of the actual MP4. Historical renders do not prove the current head.

## Publication boundary

Clipper stops at production artifacts and QC evidence. It does not automatically upload to TikTok, Instagram, YouTube Shorts, Whop, or another campaign platform. Publishing/submission is a separate explicitly authorized action after the final media and campaign/account requirements are reviewed.
