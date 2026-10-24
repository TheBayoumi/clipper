# Image-to-3D Option A: storyboard-approved host

## Locked appearance

Reference artwork: `character_turnaround_and_expression_sheet.png`
SHA256 `4220e0b76add8cbf35d5233b173a800808f2e1b43f70363f4409e9c1170a9d76`

Clean full-body front-view input: `a_clean_studio_neutral_background_scene_a_single.png`
SHA256 `3f7853775cae4b20569f74830e944a6c1b57c9129fda35312dfdf031e1020869`

Maintain dark wavy/curly hair, black rectangular glasses, dark hoodie,
expressive young adult facial proportions, dark trousers and white trainers.
A full-body neutral pose avoids occluded arms and improves auto-rigging.

**Do not feed the storyboard collage** to the image-to-3D model.
A clean portrait front input and optional alpha-masked reference are provided
in the conversation download package. The original, unmasked PNG is retained.

## Implementation

`scripts/trellis_image_to_3d.py` orchestrates the official MIT-licensed
`microsoft/TRELLIS.2` public Gradio ZeroGPU Space:
`/preprocess_image` → `/image_to_3d` → `/extract_glb`.

`.github/workflows/approved-host-image-to-3d.yml` runs an API-health probe
on changed files, and a real model generation only after **explicit**
`workflow_dispatch` with `probe_only=false` and either the full character
picture at a reachable HTTPS `reference_url`, or a base64 reference
`reference_base64`. It supports `secrets.HF_TOKEN` when the user has it.
Anonymous Hugging Face ZeroGPU quotas can reject image generation.

Any GLB candidate must pass geometry, positive dimensions, material and
SHA256 checks. The output is named `page-to-life-host-candidate.glb`.
A clean GLB is **not automatically rigged** and is not editorial approval.

## Current evidence

GH API probe has passed on the feature branch, establishing the external
Gradio endpoint is reachable. That is **not** a model-generation test.
Actual conversion from the new source image is still pending transport of
the local reference image to an authorized HTTPS endpoint or GH dispatch
payload. We deliberately do not substitute a stock model for our approved
character or claim that a real GLB exists.

Once the actual output exists:
1. Compare front/side/back proportions to the approved turnaround.
2. Inspect hair, glasses, hood, hands, feet and actual rear geometry.
3. Repair topology and split or retopologize parts if necessary.
4. Rig body and build expression/viseme controls. Feed existing Kokoro.
5. Produce the actual 15-second story-matched pilot with three physical
   prop actions, 1080×1920 video and full audio/media QC.
6. Review the exact MP4 before PR merge or any social publication.

PR #10 remains draft and publication stays disabled.
