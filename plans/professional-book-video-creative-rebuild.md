# Creative rebuild — professional illustrated book lessons

**Decision 2026-10-24:** The user reviewed the real 1080×1920 Pip output and
rejected the current art direction as unprofessional. This is a **creative
failure**, not an FFmpeg failure. The exact-head full Qwen3→Kokoro→Pip video
from [GitHub run 40047215101](https://github.com/TheBayoumi/clipper/actions/runs/40047215101)
is encoded and playable but **editorially rejected**. The two-pose mascot
and repeated poster-like scenes are deprecated for production.

## What failed

- Geometry made in Pillow, flat color planes, round eyes and glasses:
  inconsistent with an adult-facing psychology/business reading channel.
- Immobile presenter, two alternating mouths unaligned to phonemes, and no
  meaningful acting or microexpressions.
- Static, unvarying title-card composition; book context does not create
  action, narrative or scene-specific illustration.
- Structural QC (codec, 9:16 resolution, decode) mistook technical success
  for editorial readiness. No subjective visual review had been performed.
- More FFmpeg drawing tricks and character bobbing cannot fix the art direction.

## Replacement visual language — provisional creative reference

**Editorial-quality 2.5D/3D animated explainer, grown-up, cinematic and
story-driven**, not flat clipart. Keep one expressive and recognizable narrator,
but use the narrator selectively. Most runtime should *show* the lesson with
rich environments, illustrated scenarios, objects and dynamic motion graphics.

Brand/channel title and character identity are still working concepts, not
cleared for public use. Avoid imitating the copyrighted illustration style of
a specific studio or reproducing book cover artwork. Book title and author
may be attributed textually.

### Required reusable character pack

1. Approved neutral, three-quarter and profile turnarounds; consistent
   identity, hair, clothing, skin texture and lighting under all poses.
2. Approved expressions: curious, skeptical, focused, surprised, empathetic,
   relieved, confident; include high-resolution close-ups and full-body frames.
3. Storyboard acting: present a book, look at an action, interact with a
   desk/phone/calendar, react to a result, return for an ending.
4. Realistic shot staging: camera crop, depth, shadow, highlights, props,
   color grading and lighting directions. No full-screen idle mascot by default.
5. Speech visuals only with aligned mouth/visemes validated against actual TTS;
   until then, narrate over action/B-roll or use off-camera character shots.

### Pilot storyboard — Atomic Habits example (40–55 sec)

| Beat | Visual direction | Purpose |
|---|---|---|
| 0–3 sec | Human-scale relatable problem: tired character facing a pile of abandoned plans, close camera movement | Visual hook, not a giant text title |
| 3–8 sec | Dynamic book-title and author attribution using original graphic shapes; character enters frame naturally | Source attribution, no copyrighted cover |
| 8–18 sec | Desk environment transforms; phone alarm, habit cue, visual flow; overlay one clear short principle | *Show* the concept |
| 18–36 sec | Before/after mini-story: chaotic start contrasted with a tiny, repeatable two-minute action; motivated acting | Relatable demonstration |
| 36–50 sec | Character returns for a concise, specific takeaway; warm branded end slate | Complete payoff and shareable identity |

Timing depends on the actual Kokoro waveform, not fixed guesses. The first
reference is an **illustrative moodboard**, not usable rig assets, a source
book cover, or proof the clips exist.

## Engineering architecture

- **Keep:** source/claim validation, Qwen script caching, Kokoro TTS caching,
  media QC, provenance manifest, human-review approval, draft-only safety.
- **Replace:** `book_cartoon.py` vector mascot render with an
  `illustrated_story` backend consuming an **approved versioned asset pack**,
  source-validated episode beat list, camera/motion cues and audio timing.
- **Production approach:** art-direct coherent character/scene designs first,
  make a riggable master character (Blender 3D or layered 2D with licensed
  animation tooling). Use deterministic animation/compositing for repeatable
  poses, and GPU-assisted scene/motion generation when it materially improves
  realism. Pin model, seed/reference and source asset identity. Never use a
  generic image generator as a face-consistency guarantee.
- **Speech:** Kokoro WAV→time-aligned speech/phoneme or viseme analysis.
  Animate the correct mouth shapes only if independently verified. Exact
  word-aligned subtitles, safe-area design and readable phone-sized text.
- **Editorial:** a meaningful storyboard for each concept rather than one
  generic layout for every scene. Minimum two actual example/action shots and
  visibly differentiated camera setups for the pilot, judged by human review.
- **Acceptance:** character identity consistency, scene relevance, natural
  movement/pacing, readable captions, audio sync, no cutoffs, no misleading
  book claims, source attribution, commercial-use rights and a human reviewer
  explicitly approving the **exact MP4 SHA256**.
- **Workflow:** render one representative shot and a 10–15 sec approved sample
  before investing in longer GPU output. Keep work on
  `feat/unified-niche-tts` and [PR #10](https://github.com/TheBayoumi/clipper/pull/10).
  Do not re-run long Qwen inference just to test graphic style: use
  cached/recovered editorial timelines.

## Enforcement

The rejected `Pip` prototype is accessible only for **explicit manual
technical reproduction**, using `--allow-prototype-visuals` or
`allow_rejected_prototype=true` in GitHub Actions. It is labeled
`creative_review_status=REJECTED_PROTOTYPE` and `publish_eligible=false`.
The default book-production command fails closed before running Qwen3 or Kokoro.
The new professional renderer is *not yet implemented or accepted*.

The AI illustrated moodboard shown in chat is a visual direction reference
only, not approved artwork, a character rig, a reproducible multi-shot video,
or a substitute for human creative assessment.
