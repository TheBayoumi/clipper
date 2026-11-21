# Book pipeline Phase 1B — Narration stage and legacy boundary

## Status

The canonical book stages in `src/clipper/books/` are selection, scripting,
and narration, all backed by the existing `DagStore` for stage identity,
durable evidence, locking and content-addressed resume. Direction, visuals,
assembly and review remain registered but `NOT_PORTED` and never launch a
producer. The first unported failure is now `NOT_PORTED: books.direction`.

## CPU-only recorded run

A deterministic run using the committed Atomic Habits script and owner-supplied
synthetic or previously recorded audio takes this form:

```sh
clipper books run --book-id atomic-habits --lesson-id two-minute-rule \
  --script-brief niches/atomic-habits-episode-02.json \
  --narration-dir artifacts/recorded-narration \
  --artifact-root artifacts/book-dag-test
```

For an episode with four scenes, the narration directory must contain exactly
`scene-00.wav`, `scene-01.wav`, `scene-02.wav` and `scene-03.wav`.
Each file must be a valid mono, 24 kHz WAV with at least 8,000 frames. The CLI
requires `--narration-dir`, so it never silently invokes Kokoro. Use
`--script-brief` as well to avoid the existing optional Qwen script-generation
path. Both inputs are local, explicit and content-addressed; this is **not**
evidence that any real Kokoro model generated them.

The recorded-WAV mode hashes bytes for stage identity, validates each recording
before a narration PASS, and does not seed the normal Kokoro output cache. If
a recording changes, the stage gets a new cache identity. A repeated identical
run resumes the selection, script and narration stages and re-records the
expected failure at direction, without producer calls.

## Kokoro legacy fidelity

`scripts/niche_video.py` now delegates `_kokoro_pipeline`, `_wav_metadata`,
`_kokoro_wav` and `synthesize_scene` to `clipper.books.narration`.
The wrapper retains its existing public/private symbol aliases, mock injection
points, CLI flags and exit behavior. The package uses the same pinned
`VOICE_MODEL = hexgrad/Kokoro-82M`, `VOICE_REVISION = v1.0`,
`VOICE_CONTRACT = kokoro-24khz-wav-v1`, voice, text, rate, language and speed
in the original request fingerprint. The WAV cache remains
`<cache-root>/tts/<request-sha256>.wav`, with the same validation errors.
Golden tests assert request hash, path, cache bytes, generated-audio rejection
and cached-file handling, without model downloads or audio hardware.

The existing Kokoro `KPipeline` constructor accepts the repository name and
language code; the old runtime did not pass `VOICE_REVISION` as an explicit
constructor argument. This phase preserves that historical invocation instead
of guessing a revision flag or silently changing a provider integration.
The revision remains in the legacy request fingerprint and new DAG identity;
this fact alone does **not** prove the identity of weights in a live process.

For the DagStore-backed Kokoro path, a resumed PASS also requires the original
WAV to exist, match the recorded SHA-256 and retain valid geometry. Missing
or replaced audio fails closed without silently regenerating it.

## Legacy prototype boundary

The *creatively rejected* two-pose Pillow character renderer, FFmpeg utilities,
archival `produce` function and its CLI remain **only** in
`scripts/niche_video.py`. They are not duplicated or introduced into the
canonical books domain package. Moving an explicitly rejected renderer into
the reusable production package would expand the production surface and
require a separate creative/visual approval and compatibility migration;
neither is authorized in Phase 1B. The historic
`--allow-prototype-visuals` opt-in, default creative BLOCK, and
`DRAFT_REQUIRES_HUMAN_REVIEW` / `published: false` manifest semantics are
unaltered.

This is the remaining exception to a completely thin legacy script.
The selection and scripting code lives exclusively in their canonical modules;
the wrapper contains only small compatibility delegates for owned stages.

No production workflow YAML, W1/W3 policies, reviewer-identity decision, review
gate, endpoint pins, Qwen CPU-cache behavior, model invocation, deployment,
publication or coverage thresholds were changed. Independent adversarial review
and correlated live evidence remain prerequisites to production acceptance.
