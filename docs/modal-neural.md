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

The default MW4 profile **enables** GPU visual scouting in the normal
`clipper mw4 analyze` production path. It launches an ephemeral pinned Modal
worker for each certified source via Clipper's `ModalVisionProvider`. No
persistent Modal app is deployed by CI and existing shared apps remain
untouched. Model output is **PROPOSED**, never directly verified combat.
A normal production run must fail closed if GPU inference is unavailable,
times out, loses source identity, exceeds the allowed budget, or does not
return complete source-stamped results. Successful activation is **not** proof
that six-source accuracy or the whole batch is qualified. To change GPU
sampling and spend calibration, use the gameplay profile override:

```json
{
  "capabilities": {
    "neural_visual_events": {
      "enabled": true,
      "backend": "modal",
      "app_name": "clipper-mw4-vision-run",
      "deployed_sha": "current-run",
      "model_id": "Qwen/Qwen3-VL-8B-Instruct",
      "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
      "sample_interval_seconds": 15,
      "batch_frames": 2,
      "frame_max_edge": 512,
      "max_refinement_anchors": 20,
      "max_sample_frames": 144,
      "deadline_seconds": 480,
      "max_gpu_seconds": 1800,
      "max_estimated_usd": 2
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

Neural scouting uses an isolated `clipper-*` app identity, not the shared
ContentRewards `clipper-open-editor` or `clipper-production-pipeline` apps.
The worker's app name comes from `CLIPPER_NEURAL_MODAL_APP`; its immutable
commit identity from `CLIPPER_DEPLOYED_GIT_SHA`. The production Actions job
sets these to `clipper-mw4-vision-run` and `${{ github.sha }}` and runs
`clipper mw4 analyze`, which launches an ephemeral worker for the duration
of source inference. No persistent deploy is needed. An independently
operated persistent deployment remains an explicit administrative operation:

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

## Source-wide isolated GPU qualification (six-source, opt-in)

The permanent validation-only workflow `.github/workflows/modal-neural-qualification.yml`
has an optional `workflow_dispatch` input `source_wide=true`. It uses one of six
hash-certified original MediaSilo **analysis proxies** as inputs per job, rechecks
each certificate, and invokes the existing `VisionModel` via Modal's ephemeral
app context with `modal run ...::source_scout`. The orchestration is in the same
canonical Clipper inference provider (`ModalVisionProvider`) and normal Clipper
CPU visual-semantic timeline (`gameplay.analysis.analyze_source`); no duplicate
analyzer, planner, allocator or renderer is instantiated.

An isolated source-wide run produces `neural_visual_evidence.json`,
`neural_review_queue.json` and `source_wide_gpu_qualification.json`. The
expected and observed number of source-frame samples must match. Every
observation remains `PROPOSED` or a negative observation; the reviewer queue
may link to independently verified pre-existing anchors but never turns a
model claim into a verified event. Every sample is tied to exact certified
source bytes, rational time grid, pinned model revision, prompt and deployment
SHA. Explicit per-source GPU and sampling budgets fail closed, and terminal
cancellation remains mandatory. Recorded dollar figures are **estimates**, not
account-wide hard Modal spending caps.

The validation job may reuse archived **certified** analysis proxies for
repeatable experiments. Original-master renders, complete independent video and
audio ground-truth review, player-death evidence, GPU speech/alignment, actual
10–12-second encoded clip and final same-SHA production qualification remain
separate mandatory acceptance gates. No MediaSilo credential or original
uncompressed master is uploaded to the Modal worker.

## Bounded adaptive production analysis and reproducible qualification

Clipper keeps sparse, rational-frame-clock coverage of the whole video and
prioritizes every independently verified outcome and campaign-reviewed cue.
Remaining local refinement slots now go to strong audiovisual activity or
uncertain changes **across the entire source**, using the canonical CPU
features, not just earliest combat detections. Fixed 15-second VLM interval
alone is not enough to discover every 2-second interaction; no such claim is
made. The VLM remains a context proposal generator; the direct CPU verifier
and independently verified scene events are the only authorities for outcomes.

MW4 calibrates this Clipper functionality to 2-frame GPU batches, a 512-pixel
source image edge and at most 144 samples per source, with an estimated 1800
GPU-second/$2 inference budget. Large model and multi-batch latency must be
measured in an actual six-source run; exceeding budget fails closed rather than
silently omitting independently verified anchors. Modal timeout cancellation
also requests remote GPU container termination. Unknown cancellation terminal
state remains a hard failure; no speculative retry or success receipt.

Source-wide inference saves source/model/deployment-SHA-bound batch checkpoints.
The production workflow restores and retains checkpoints for retries on the
**same exact commit**, and uploads partial diagnostics on failure. The model
cache can never substitute for independent reviewed positives and negatives.
The canonical analysis manifest provides candidate edit-type counts, inferred
source-wide observations, cache reuse, GPU seconds, estimated GPU cost and
reviewable new-vs-corroborated proposals. These are throughput/coverage metrics,
not proof of precision, recall, or editorial quality.

Full MW4 qualification is manually dispatched on a frozen SHA; normal CI remains
push-triggered. Source analyses (2 independent runners), global allocation (1),
rendering (2 independent runners) and final batch QA have explicit dependency
order. Independent six-source A/V gold continues to reject incomplete review;
it is never bypassed to produce a green build. A qualifying acceptance record
requires same-SHA CI, source-wide GPU/CPU review, all verified-anchor coverage
and actual original-master render QA, plus source and final-clip playback.

## Failure-analysis correction: source-wide deadline must match MW4 production

The six-source GPU qualification previously **overrode** MW4's calibrated
480-second source-inference deadline with a hard-coded 180-second value in
`source_scout`. The first Convenience Store model batch timed out and the
30-second post-cancel terminal poll also timed out. That was a real
qualification/configuration defect, not evidence of a source semantic
rejection and not an authorization to ignore timeouts.

`source_scout` now retains the canonical gameplay profile's inference deadline
and only overrides bounded sampling/compute parameters supplied by the
validation command. A regression confirms the production deadline is
preserved. Clipper also writes `neural_inference_failure.json` whenever the
worker or batch parser fails, with exact certified source/model/Git identity,
failed batch/times, complete versus expected sample counts, previous verified
checkpoint hits and consumed budget telemetry. This is **a failure artifact**,
never a neural qualification receipt, and cannot promote VLM observations into
confirmed gameplay. An unconfirmed canceled GPU call remains fail-closed.

The independent six-source human review file is still marked
`benchmark_annotation_seed_NOT_model_ground_truth` with
`independent_review_complete=false`. Its acceptance failure is correct until
actual independent full A/V playback, signed annotations, negative examples,
source hashes, rational frame-rate evidence and cinematic rereviews exist. No
synthetic test fixture, GPU proposal or contact sheet substitutes for this.

## Cold-start readiness and GPU quota isolation (2026-10-23)

At `fc92620`, both the one-frame smoke and the Convenience Store source-wide
qualification started Modal apps but never produced readiness receipts. They
ran concurrently, each requesting **two L4 GPUs**, and were eventually
cancelled by their GitHub job limits. This does not establish whether GPU
quota, cold-start model loading, or a control-plane failure was responsible.

All three canonical Clipper GPU entrypoints now bound `ready.spawn().get()`:
900 seconds for the smoke and source-wide qualification, and the configured
480 seconds for MW4 production. A timeout requests GPU container termination
and refuses unknown terminal states. Source-wide and smoke retain source-bound
readiness failure artifacts. The one-frame smoke also uses Clipper's existing
bounded inference provider instead of an unbounded `inspect.remote()` call.
The source-wide matrix **requires a successful smoke job first**, avoiding
simultaneous 2×L4 allocation and expensive qualification after a failed smoke.
None of these changes certify a source, model observation, or human gold.
