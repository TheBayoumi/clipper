# Clipper

A production-oriented, rights-gated pipeline that turns a Whop Content Rewards brief into timestamped YouTube clip candidates and rendered 9:16 MP4 clips.

## What it automates

1. Parse and validate a campaign brief.
2. Search only the YouTube channels/video IDs authorized by that brief.
3. Retrieve timestamped captions when available; otherwise run local Faster-Whisper ASR.
4. Score sentence-aligned windows against campaign keywords, required phrases, hook strength, duration, and negative terms.
5. Select diverse clips across source videos.
6. Render 1080x1920 H.264 clips with blurred background fill, word-synced reveal captions, smoothed face-tracked zoom, and EBU-style loudness normalization.
7. Save a manifest containing source URLs, timestamps, scores, errors, and output paths.

## Non-negotiable source policy

The pipeline intentionally refuses unrestricted scraping. A brief must contain either `source_channel_ids` or `allowed_video_ids`, and `rights_confirmed` must be `true`. Set it only after confirming the Whop campaign permits clipping those exact sources. The pipeline does not bypass DRM, private-video access, paywalls, or platform permissions.

YouTube's official captions download API only supports videos the authenticated account can edit. For public campaign-authorized source videos, this project uses `yt-dlp` to request available subtitles and media. The user remains responsible for the campaign rules, attribution, platform terms, and copyright permissions.

## Architecture

```text
Whop brief YAML/JSON
        |
        v
Brief validator + rights allow-list
        |
        v
YouTube discovery (Data API key preferred; yt-dlp fallback)
        |
        v
Timed subtitles ---- unavailable ----> Faster-Whisper ASR
        |                                  |
        +----------------+-----------------+
                         v
             deterministic clip scorer
                         |
                         v
                diversity selector
                         |
                         v
      face tracker + word-timed ASS captions
                         |
                         v
              FFmpeg vertical renderer
                         |
                         v
 clips/*.mp4 + *.ass + *.tracking.json + manifest.json
```

## Requirements

- Python 3.11 or 3.12
- FFmpeg on `PATH`
- A YouTube Data API key is strongly recommended: `YOUTUBE_API_KEY`
- Optional GPU for Faster-Whisper. CPU works with the default `int8` compute mode.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,asr]"
cp campaign.example.yaml campaign.yaml
```

Edit `campaign.yaml` using the exact Whop brief. Do not set `rights_confirmed: true` until the allowed source channels/videos are verified.

```bash
clipper validate --brief campaign.yaml
clipper discover --brief campaign.yaml
clipper run --brief campaign.yaml --artifact-root artifacts
```

For transcript/timestamp planning without FFmpeg rendering:

```bash
clipper run --brief campaign.yaml --artifact-root artifacts --no-render
```

## Output

```text
artifacts/<campaign-id>-<UTC timestamp>/
├── brief.normalized.json
├── manifest.json
├── clips/
│   ├── 01-<youtube-id>.mp4
│   ├── 01-<youtube-id>.ass
│   └── 01-<youtube-id>.tracking.json
└── work/<youtube-id>/
    ├── transcript.json
    ├── clip-candidates.json
    └── source/subtitle files
```

## Environment tuning

| Variable | Default | Purpose |
|---|---:|---|
| `YOUTUBE_API_KEY` | unset | Official YouTube search and metadata |
| `CLIPPER_ARTIFACT_ROOT` | `artifacts` | Run output root |
| `CLIPPER_WHISPER_MODEL` | `small` | Faster-Whisper model |
| `CLIPPER_WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `CLIPPER_WHISPER_COMPUTE_TYPE` | `int8` | ASR precision/performance trade-off |
| `CLIPPER_FACE_TRACKING` | `true` | Enable sampled face tracking with center fallback |
| `CLIPPER_FACE_ZOOM` | `1.12` | Subtle tracked crop zoom; renderer accepts 1.0–1.35 |
| `CLIPPER_FACE_SAMPLE_FPS` | `4.0` | Face-detection sampling rate; crop motion is interpolated |

### Resource implications

- YouTube auto-caption word timestamps are preserved when available; otherwise caption words are time-distributed within each cue.
- Face detection runs on downscaled sampled frames and produces a smoothed crop path; missing faces fall back to a stable center crop.
- `small` Faster-Whisper is suitable for CPU fallback; `medium`/`large-v3` require substantially more RAM/VRAM and increase latency.
- Source media dominates storage. Each run is isolated under `artifacts/`; delete old `work/` directories after final QC.
- FFmpeg rendering is CPU-heavy. Parallel rendering is deliberately not enabled yet to avoid uncontrolled RAM and I/O spikes.

## Quality gates

```bash
make check
```

CI runs Ruff, strict mypy, and pytest with a 95% coverage floor on Python 3.11 and 3.12.

## Current boundary

This first production slice stops at reviewed MP4 artifacts. It does not auto-upload to TikTok, Instagram, YouTube Shorts, or Whop. Publishing should be added only after a visual/audio QC gate and per-platform account permissions are in place.
