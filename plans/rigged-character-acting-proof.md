# 3D animated character proof – replace discarded Pip cutout

This proves physical human skeleton movement independently from camera motion.
The original Pip vector character, 2-mouth-pose animation and still-image pan
**remain rejected and publish-blocked**. This is NOT an announcement of
creative acceptance.

## Technical and licensing source

[Ara 3D Gen](https://github.com/olatunde85/3d-gen) provides a genuine
MakeHuman mesh, skin weights, armature, facial joints, expressions and reusable
animation clips. Its source code is MIT; core MakeHuman human data are CC0.
Some walking/running/cry clips derive from CMU mocap and may be used
commercially, but cannot be resold **as motion data on their own**.
Pin exact source commit: 8aa993d7141b6f37e5d16f611264e6f1ab520f30.
The actual origin and license notes are copied from upstream rather than
pretending all external motion is individually CC0.

## What is implemented

- One versioned persistent character specification:
  niches/character-host-v1.json (adult human, casual outfit and neutral
  expression, fixed seed, face bones enabled, GLB skin rig and animations).
- scripts/blender_character_probe.py loads the actual GLB in Blender. It
  rejects a GLB lacking skinned meshes, a meaningful Wave action, or a
  measured arm/hand joint change of at least 0.015 in pose-matrix elements.
  It renders consecutive body-performance frames under studio lighting
  and records a source model SHA256 and actual joint movement.
- .github/workflows/rigged-character-proof.yml uses the pinned MIT code,
  builds the CC0 MakeHuman asset source in GitHub Actions, exports one
  versioned GLB, renders actual armature animation, and combines it with
  **existing Kokoro audio** from full original Clipper run 40047215101.
  No Qwen3 rerun, no paid AI video model, no social publication.
- Workflow outputs model GLB, a real movement video, proof.json, media
  full-decode and a rendered thumbnail. Human review is required.
- tests/test_rigged_character_proof.py checks fixed identity and real-motion
  gate.

## Exact acceptance contract

**Stage 1:** GitHub runner must show the "Wave" skeletal action loaded,
mesh skin weighted to armature, joint delta greater than 0.015 across
rendered action frames, media decode, and GLB source evidence. First proof is
2–4 sec at 360x640 12fps for cheap reproducibility; low resolution is
purposely **not** a professional deliverable.

**Stage 2:** After visible proof and a professional character-art decision,
render 3 different 10–15s 1080x1920 shots with on-screen acting and props.

**Stage 3:** Match Kokoro word/phoneme timing to *jaw/facial rigs* or keep
the character off-camera during narration. Audio envelope alone is not
phoneme-accurate lip sync.

**Stage 4:** Human editorial review of actual output, character identity,
lighting/pacing, full book evidence, audio cutoffs, rights, and exact MP4
SHA256. Publication is disabled, PR remains draft.

Do NOT raise machine-coverage stats as evidence of visual quality.
