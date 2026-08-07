# Clipper

A rights-gated short-form production engine for Whop clipping campaigns. Clipper mines authorized long-form creator content, ranks distinct stories, plans legitimate hook variants, renders true 9:16 edits, and persists technical evidence for every production run.

## Production funnel

```text
Campaign brief + authorized source
        ↓
Full timed transcript
        ↓
StoryMoment discovery
        ↓
Multidimensional editorial scoring
        ↓
ClipConcept mining
        ↓
Semantic clustering / topic diversity
        ↓
HookVariant planning
        ↓
EditPlan + editorial beats
        ↓
Internal render budget
        ↓
1080x1920 speaker-aware MP4 batch
        ↓
Per-clip technical QC
        ↓
Submission shortlist
```

The campaign's final `clip_count` is **not** the analysis budget. A long podcast can produce dozens of raw moments and concepts while the campaign still requests only the best three submissions.

## Source and rights policy

The pipeline refuses unrestricted acquisition. Every brief must contain `source_channel_ids` or `allowed_video_ids`, and `rights_confirmed` must be `true`. Set that flag only after the campaign authorizes the exact source. Clipper does not bypass DRM, private-video access, paywalls, authentication, or platform permissions.

For public campaign-authorized YouTube sources, Clipper requests available captions/media with `yt-dlp`; a YouTube Data API key is preferred for discovery. The user remains responsible for campaign terms, attribution, copyright permissions, and platform rules.

Before production media download, Clipper inventories the authorized video's available streams and selects the highest practical MP4 video at or below `CLIPPER_SOURCE_MAX_HEIGHT` (default `1080`). The selected format ID, resolution, codec, bitrate, FPS and available alternatives are persisted beside the source media.

## V9 editorial model

The V9 pipeline preserves the V8 full-podcast editorial funnel while separating a source timestamp from the finished edit:

- `StoryMoment` — a conversational/story unit discovered from the full transcript.
- `ClipConcept` — a self-contained source story with setup, payoff, topic, duration, fingerprint, and editorial scores.
- `HookVariant` — a legitimate source-derived opening strategy. Supported modes include direct, curiosity text, question, number, conflict, and strong opinion. `payoff_first` is intentionally not emitted until a multi-span reorder renderer can preserve continuity safely.
- `EditPlan` — the exact source span, hook text, caption platform, ranking score, and meaning-driven visual beats sent to the renderer.

### Editorial scoring

Each concept records configurable 0–10 judgments for:

`hook_strength`, `curiosity`, `payoff_strength`, `standalone_clarity`, `emotional_energy`, `information_value`, `controversy_or_tension`, `quoteability`, `specificity`, `campaign_relevance`, `story_completeness`, and `retention_potential`.

Weights live under `editorial.score_weights` in the campaign brief. Start and end boundaries are separately scored; incomplete endings, filler openings, generic podcast housekeeping, and weak semantic closure are penalized.

### Semantic diversity

When no embedding service is configured, V9 uses a deterministic lexical-semantic fallback combining token-frequency cosine similarity and Jaccard overlap. The selected concept library enforces both semantic-cluster diversity and a configurable per-topic cap. This fallback is deterministic and intentionally does not pretend to be a learned embedding model.

## Rendering and editorial behavior

- True full-frame `1080x1920` output; no blurred duplicate background.
- Speaker-aware portrait framing starts from the maximum-resolution `9:16` source crop (`zoom_factor=1.0`) rather than globally enlarging faces.
- Camera changes are classified before rendering: source-camera cuts and large same-shot speaker switches are hard cuts, acceptable two-person/small-displacement compositions hold, and only genuine same-speaker subject movement may use an eased displacement-aware reframe.
- Source camera cuts are detected from the original footage and never receive a sliding crop transition. Transition evidence records reason, mode, distance, duration and source-cut timing for QC.
- Word-synchronized ASS captions preserve source word timestamps when available. If exact words are unavailable, the ASS evidence is explicitly marked `TimingMode: cue_interpolated`; QC never labels that mode word-exact.
- Platform-safe caption presets for TikTok, Instagram Reels, YouTube Shorts, and generic vertical output. Presets are conservative margins rather than claims about permanent app UI coordinates.
- Source-derived hook text can appear briefly above the dialogue.
- Digital punch-ins are disabled by default. The renderer rejects legacy post-upscale punch beats because they resample already-upscaled pixels and visibly soften podcast footage.
- Dialogue is normalized with FFmpeg loudness processing.
- Required campaign watermark assets are applied when configured.

## Production configuration

Example:

```yaml
production:
  candidate_pool_size: 36
  concept_count: 10
  variants_per_concept: 3
  final_render_budget: 6

diversity:
  semantic_similarity_threshold: 0.72
  max_concepts_per_topic: 2

hooks:
  enabled:
    - direct
    - curiosity_text
    - question
    - number
    - conflict
    - strong_opinion

editorial:
  platform: tiktok
  punch_ins_enabled: false
  max_punch_ins_per_clip: 0
  semantic_endings: true
  post_speech_tail_seconds: 0.25
  caption_max_lines: 2
  score_weights:
    hook_strength: 1.3
    payoff_strength: 1.2
    campaign_relevance: 1.6
    retention_potential: 1.3
```

Generic defaults remain intentionally cheap for CI/smoke runs. Real campaign YAML can opt into a larger internal production budget without changing the campaign's final submission target.

## Render profiles and image quality

Clipper separates encode speed from release quality:

- `smoke`: `libx264`, `ultrafast`, CRF 23 — CI/container smoke only.
- `review`: `libx264`, `medium`, CRF 18 — editorial review artifacts.
- `production`: `libx264`, `veryfast`, CRF 17 — default production master; selected from full-length 1080 acceptance tests to stay within the release runner memory envelope while preserving source pixels and one-generation rendering.

Production composition performs one source crop followed by one Lanczos scale to `1080x1920`; there is no scale-then-zoom-then-crop path. Render/tracking sidecars record source resolution, crop resolution, effective scale, resampling-stage count, digital-zoom usage, profile, preset and CRF.

## Caching and reproducibility

Transcript and editorial-analysis caches are schema-versioned and keyed from relevant inputs including source identity/hash, transcript/model identity, normalized production configuration, and cache schema version. A changed source or relevant config invalidates downstream analysis rather than silently reusing stale output.

Run manifests persist:

- Git commit SHA
- source hashes when media is acquired
- transcript hashes and transcript source
- normalized campaign configuration
- story moments, concept rankings, semantic clusters
- hook variants and edit plans
- source ranges and transcript fingerprints
- render hashes
- cache hit/miss events
- technical QC
- wall/CPU time, peak RAM, artifact disk use, and optional GPU-utilization samples

Set `CLIPPER_CACHE_ROOT` to keep reusable analysis outside the per-run artifact directory.

## Output

```text
artifacts/<campaign-id>-<UTC timestamp>/
├── brief.normalized.json
├── story-moments.json
├── concept-ranking.json
├── hook-variants.json
├── edit-plans/
│   └── plan-*.json
├── clips/
│   ├── 01-<topic>-<hook>.mp4
│   ├── 01-<topic>-<hook>.ass
│   └── 01-<topic>-<hook>.tracking.json
├── qc/
│   └── 01-<topic>-<hook>.json
├── work/<youtube-id>/
│   ├── transcript.json
│   ├── story-moments.json
│   └── clip-candidates.json
└── manifest.json
```

## Automated technical QC

Each production render is checked for:

- valid FFmpeg/ffprobe decode
- 1080x1920 geometry and ~30 fps
- H.264 video + AAC audio
- edit-plan duration agreement
- objective loudness / true peak / long silence
- caption safe-region margin and timing provenance
- tracking evidence confirming no filler and a valid 9:16 source crop
- transition QC rejecting source-cut sliding, excessive eased velocity, premature target reframes and short-window oscillation
- source/crop resolution, effective upscale factor, resampling-stage count and digital-zoom evidence
- render profile, encoder preset, CRF and output bitrate
- required watermark asset presence

Technical PASS does not imply editorial PASS. Actual final MP4 review remains a separate production gate.

## Environment

| Variable | Default | Purpose |
|---|---:|---|
| `YOUTUBE_API_KEY` | unset | Official YouTube search/metadata |
| `CLIPPER_ARTIFACT_ROOT` | `artifacts` | Per-run output root |
| `CLIPPER_CACHE_ROOT` | `<artifact-root>/_cache` | Persistent transcript/editorial cache |
| `CLIPPER_WHISPER_MODEL` | `small` | Faster-Whisper fallback model |
| `CLIPPER_WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `CLIPPER_WHISPER_COMPUTE_TYPE` | `int8` | ASR precision/performance |
| `CLIPPER_SOURCE_MAX_HEIGHT` | `1080` | Practical default source height; raise explicitly on higher-memory production runners |
| `CLIPPER_RENDER_PROFILE` | `production` | `smoke`, `review`, or `production` encoder profile |
| `CLIPPER_SPEAKER_FOCUS` | `true` | Enable speaker-aware portrait framing |
| `CLIPPER_SPEAKER_ZOOM` | `1.0` | Base portrait zoom; production default preserves maximum source pixels |
| `CLIPPER_SPEAKER_SAMPLE_FPS` | `4.0` | Face/mouth analysis rate |
| `CLIPPER_SPEAKER_SWITCH_MARGIN` | `1.35` | Speaker-switch hysteresis |
| `CLIPPER_SPEAKER_MIN_REFRAME_SECONDS` | `0.35` | Minimum eased same-speaker subject reframe duration |
| `CLIPPER_SPEAKER_MAX_REFRAME_SECONDS` | `0.9` | Maximum eased same-speaker subject reframe duration |
| `CLIPPER_SPEAKER_SECONDS_PER_CROP` | `0.75` | Additional ease duration per crop-width of movement |
| `CLIPPER_SPEAKER_HOLD_THRESHOLD` | `0.28` | Small same-shot speaker displacement that should hold instead of moving |
| `CLIPPER_SPEAKER_REVERSAL_GUARD_SECONDS` | `1.25` | Suppress short-lived speaker switches that would immediately reverse |
| `CLIPPER_SPEAKER_WINDOW_SECONDS` | `0.8` | Active-speaker decision window |
| `CLIPPER_SPEAKER_MIN_DETECTION_COVERAGE` | `0.35` | Reject sparse cut/graphic detections |

## Setup and commands

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,asr]"
cp campaign.example.yaml campaign.yaml

clipper validate --brief campaign.yaml
clipper discover --brief campaign.yaml
clipper run --brief campaign.yaml --artifact-root artifacts
```

Analysis-only mode:

```bash
clipper run --brief campaign.yaml --artifact-root artifacts --no-render
```

Quality gate:

```bash
make check
```

CI runs Ruff, strict mypy, and pytest with a 95% coverage floor on Python 3.11 and 3.12. Heavy podcast production rendering is intentionally kept out of CI; CI uses deterministic fixtures and a bounded container smoke render.

## Publication boundary

Clipper stops at production artifacts, QC evidence, and a submission shortlist. It does not automatically upload to TikTok, Instagram, YouTube Shorts, or Whop. Publishing must remain a separate explicitly authorized action after actual video review and account/campaign checks.
