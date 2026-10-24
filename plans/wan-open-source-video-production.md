# Clipper open-source Wan video production: implementation evidence

This is the current implementation status on feat/unified-niche-tts, PR #10.
Existing licensed-clip workflows and Qwen/Kokoro cached work stay intact.
Auto-publication remains disabled.

## Verified (2026-10-24)

- scripts/wan_shot_contract.py: SHA256 shot fingerprint based on model, approved
  reference image, narration WAV, prompt, seed and storyboard scene; image
  minimum dimensions 1024x1024; full FFmpeg decode of video candidates.
  Technical success never approves creative or publishing quality.
- scripts/modal_wan_s2v.py: official Wan2.2-S2V-14B renderer for Modal A100 80GB;
  original upstream Git commit 1ea34ff48f87168174e12956e200b1d908b1c5ff;
  Hugging Face checkpoint dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b;
  persistent independent Wan weights volume. Inference NOT YET RUN.
- .github/workflows/wan-s2v-qualification.yml: cheap on-push preflight
  and explicit mode=generate behind approved original image SHA256, cached
  Kokoro audio restoration, and Modal GPU credential preflight.
- Five passing tests; CI run at commit 4c82b567 passed.
- The real model metadata/config preflight passed without 49 GB weights or
  GPU spend: https://github.com/TheBayoumi/clipper/actions/runs/40118032178
  This is NOT a proven video-render execution.

## Honest blocker

The branch currently contains only a heavily compressed 512x768 WebP
source (6,980 bytes). It cannot pass the new 1024x1024 minimum input quality
gate. The prior high-resolution approved clean A-pose original PNG is not
committed to the feature branch.

Restore the exact originally approved high-resolution image under
assets/character/page-to-life-host-approved-master.png and manually pin
its SHA256 as the workflow input. Do not silently upscale the old WebP.
This image is a user-approved creative asset, not a generic model fallback.

## Next stage: production

1. Run explicit S2V generation of one short speaking character using
   the approved full-resolution image and the previously cached Kokoro
   draft audio (run 40047215101). That recovered audio excerpt is for
   runtime qualification, not a claim of final matching storyboard narration.
2. Inspect the real generated MP4 for visual identity, glasses, hair,
   mouth motion, audio, expressions and shot composition.
3. Add and verify the official Wan I2V model for action shots. Reject
   implausible book/phone/notebook interactions. Introduce Wan-Animate-2
   only with validated driving motion assets when I2V cannot satisfy them.
4. Assemble the final 15s Atomic Habits pilot with the original storyboard
   beat timings, 1080x1920 and 24 fps, actual narration alignment and
   realistic cinematic action. Review the exact MP4.
5. Preserve the old GLB-specific acceptance as a separate 3D track;
   pixel-generation QA must not forge 3D skeleton and contact evidence.
6. Human creative approval and exact artifact hash required before
   allowing PR merge or any publication.

The model is open-weight and Apache 2.0. Remote inference on Modal still
costs GPU time. A green workflow is only technical evidence, not a
professional animation acceptance certificate.
