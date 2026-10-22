# Canonical Clipper neural visual inference

The **Clipper engine** owns source identity, semantic validation, admissibility,
candidate search, allocation, rendering and acceptance. Modal is an optional
GPU compute provider. MW4 configures the capability; it supplies **no** neural
or clipping pipeline implementation.

## Current migration slice

`clipper mw4 analyze` calls the ordinary `clipper_engine.gameplay.candidates`
route. When `capabilities.neural_visual_events.enabled` is true it also invokes
`clipper_engine.inference.vision.analyze_source`, after verifying the certified
analysis source QA manifest. The output `neural_visual_evidence.json` holds
source-clock, model-pinned **PROPOSED** observations; it is not a list of
verified kills or semantic anchors. The existing deterministic detector,
independent-anchor search, allocator and original-source renderer remain the
sole production authorities. Source-wide frames are decoded into bounded JPEG
batches at the configured interval; original masters never go to Modal.

The default MW4 profile leaves neural scouting **disabled** until live GPU and
independent cinematic/combat benchmarking have been completed. To test the
capability on a certified source use a profile override with the following
properties, backed by a deployed isolated worker:

```json
{
  "capabilities": {
    "neural_visual_events": {
      "enabled": true,
      "backend": "modal",
      "app_name": "clipper-neural-qualification",
      "deployed_sha": "REPLACE_WITH_EXACT_40_HEX_DEPLOYMENT_SHA",
      "model_id": "Qwen/Qwen3-VL-8B-Instruct",
      "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
      "sample_interval_seconds": 4,
      "batch_frames": 8,
      "frame_max_edge": 640,
      "deadline_seconds": 180,
      "max_gpu_seconds": 900,
      "max_estimated_usd": 1
    }
  }
}
```

The profile must provide an **exact deployed SHA**, never `latest`. A checked-in
profile may instead specify `"deployed_sha": "current-run"`: Clipper resolves
that value from `GITHUB_SHA` / `CLIPPER_ACCEPTANCE_SHA` at execution time and
rejects conflicting identities. This avoids the impossible requirement to embed
its own ever-changing Git commit SHA in the source profile. The model request
and worker still compare immutable **40-character SHAs** before inference.

Neural scouting snaps frame samples to the **certified rational nominal frame
rate**, recording `source_clock.nominal_rate` and sampled source-frame indices.
For 60000/1001 timing, the nearest frames to 0, 4 and 8 seconds are frames
0, 240 and 480 at 0.000, 4.004 and 8.008 seconds respectively. This is source
clock evidence, not a claim that a sparse VLM sample proves an intermediate
hit or kill. Bounded high-rate interaction verification remains Clipper-owned.

Neural scouting may
only use a separate `clipper-*` Modal app, not the shared ContentRewards
`clipper-open-editor` or `clipper-production-pipeline` apps. The worker's app
name is selected with `CLIPPER_NEURAL_MODAL_APP`; its immutable commit identity
is set with `CLIPPER_DEPLOYED_GIT_SHA`. Deploying is an **explicit controlled
operation**, never source migration through GitHub Actions:

```bash
python -m pip install -e '.[modal]'
CLIPPER_NEURAL_MODAL_APP=clipper-neural-qualification \
CLIPPER_DEPLOYED_GIT_SHA=$(git rev-parse HEAD) \
  modal deploy src/clipper_engine/inference/modal_worker.py
```

The `VisionModel.ready()` method reports model revision, CUDA availability,
GPU count and source SHA; actual `inspect()` calls require the exact SHA and
same model contract. Unit tests use an injected provider and never incur GPU
charges. In live qualification, independently audit GPU consumption including
cold container/model loading; the worker's cost telemetry is a GPU-only
**estimate** of method execution, not a hard account-wide billing limit.

A budget exhausted mid-run, wrong source SHA, incomplete observations, invalid
clock, expired deployed SHA or unconfirmed cancellation is a **hard failure**;
no silent deterministic fallback may claim neural acceptance. Cache
checkpoints bind source identity, model, prompts, sampling, deployed SHA and
observation digest. Samples are proposals only: Clipper must subsequently
verify real direct interactions and run independent legal-window search for
every confirmed payoff before enabling neural-powered production allocation.

## Remaining acceptance gates

- Fresh, isolated, authorized MediaSilo frame inference and GPU/speech testing.
- Independently reviewed positive and negative evidence across all six MW4 reels.
- Integration of verified visual events into Clipper detector/timeline semantics.
- Exact final-SHA CI, Modal worker identity, full 4K master render, FFV1/NUT
  hashes, encoded 10-12 s, SSIM/PSNR, source/color QA, full clip playback.

## Non-disruptive bounded GPU qualification

`.github/workflows/modal-neural-qualification.yml` is a **permanent
validation-only workflow**. Its push trigger matches changes to the workflow
file on the MW4 feature branch; it runs a 25-minute bounded ephemeral
`modal run` (no shared worker deployment) after retrieving and re-certifying a
previously qualified `ghost_fight` MediaSilo analysis proxy. It transmits
only one 640-pixel source-clock JPEG to Modal and records exact source SHA,
worker SHA, revision, CUDA readiness and model observations. GPU-only estimated
cost must stay below $1. Actual Modal charges can include startup/CPU/memory
and must be separately audited. A failed or unavailable Modal workspace is
reported as an incomplete qualification, not silently deemed a pass.
