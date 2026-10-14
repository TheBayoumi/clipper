# Clipper production audit — open editorial and rendering issues

**Date:** 2026-10-14  
**Reference production:** [Double Coverage run #38469818017](https://github.com/TheBayoumi/clipper/actions/runs/38469818017)  
**Reference source:** YouTube `3S-fL9rNPp0`; pinned source SHA-256 `db13e67b993ff1e56d33721ce64a0349b695b71165ab6b36791b9a5e95b9e080`.  
**Status:** Investigation list, not a claim that the problems have been fixed. The production run delivered 16 real MP4 review drafts and technically passed. **None is automatically approved for publication.**

## Verified facts

- The 111 initial candidate proposals were assessed: 44 preserved from archived checkpoint #38364505410 and 67 completed in nine parallel jobs.
- Every candidate received an initial `keep=True` from the selector. The stricter review outcomes were **16 accepted**, **47 headline-review rejected**, **45 span/story rejected**, and **3 invalid source-evidence references**.
- The final render queue contained 16 clips: **12 from accepted reviewed candidates and four recovered verbatim-quote drafts** (the recovery mechanism explicitly marks these *unapproved*).
- Clean technical production: **16/16** H.264/AAC portrait files, 1080 × 1920, 24 fps. The automated audit marked `TECHNICAL_QA_PASSED__HUMAN_REVIEW_REQUIRED`, *not* editorial approval.
- Source video and full editorial decisions were restored from pinned caches during rendering; no 111-candidate model rerun occurred in the renderer.

## Issues to address

### P0 — Editorial selection and factual integrity

1. **Selector over-accepts proposals:** `keep=True` on all 111 candidates although only 16 survive the independent review. Investigate permissive selector rubric, context interpretation, schema, and insufficient separation of quality confidence from basic eligibility. Do not impose a fixed output quota as a shortcut.
2. **Headline review inconsistent with delivery:** 47 headline-review rejections. Four quote-recovery candidates still reached the render queue as unapproved drafts. Retain this recovery only with a distinct unapproved label and a mandatory factual/headline QA gate before any publication.
3. **Invalid source evidence:** Three reviewed candidates failed evidence verification, including source span pointers that were reversed, absent, or outside the allowed namespace. Correct interface and evidence provenance; never weaken the evidence validator or invent missing claims.
4. **Malformed or low-context hooks reached MP4 output:** Examples observed: clip #2, `WE'RE IN WASHINGTON, .C.`; clip #10, `FANS DO. THEY, FANS WANT US AS PLAYERS TO DO THE SAME.`; clip #11, `I WANT THE UNIVERSITY TO DO GREAT.`; clip #15, `BEARS' QB SITUATION HAS BEEN GENERATIONS BAD`. Provide final, source-grounded language-quality validation for reviewed and recovered headlines. Do not silently rewrite spoken evidence or hallucinate facts.
5. **Editorial score does not imply approval:** Some recovered drafts receive scores as high as 100/100 despite prior headline rejection. Separate eligibility, source-evidence validity, editorial scoring and publication approval. Emotional delivery remains unscored; rubric coverage is only 90%.

### P1 — Viewer-facing subtitle and montage quality

6. **Word-by-word caption alignment incomplete:** Across 16 clips, 1,663 spoken-word highlight events and 91 unaligned/static fallback events were recorded. Diagnose root causes and decide whether each fallback is acceptable, repaired or flagged; do not claim perfect synchronized captions.
7. **Weak creative montage:** Nine clips had caption-led editing with no extra attention beat; seven had a small 2.5% zoom. The output lacks strong context-motivated B-roll, purposeful shot changes, and bespoke transitions. Improve the *editorial plan* rather than injecting random effects.
8. **Hook scoring blind spots:** Automated feedback reported zero generic hooks even with clearly awkward wording. Add a separate headline copy-edit and self-containedness review based on the full source context, and distinguish mechanical overlay checks from semantic acceptance.
9. **Visual framing requires human review:** Automated face-presence and SSIM checks cannot verify important on-screen action, cut timing, creator presence, logo compliance or viewer comprehension. Evaluate full clips, not just contact sheets.
10. **Creative text may obscure footage:** Verify persistent hooks, caption plates and important speaking faces across the full duration and all device-safe zones. Technical 1080 × 1920 compliance does not demonstrate readable, professional composition.

### P1 — Reproducibility, caching and trustworthy reporting

11. **Misleading reviewer cache metrics:** `reviewer_cache_hits=108` on rendering reflects existing stored review objects, not proof that 108 expensive assessments were avoided by the nine worker jobs. Report **reused selector decisions**, **reused model requests**, **fresh model calls**, **recovered draft count** and **model versions** separately.
12. **Do not reassess for rendering:** A finished 111-candidate checkpoint must be reusable without transcription, semantic models or fresh reviewer calls. A failed render should never trigger a full editorial restart.
13. **Preserve candidate-level checkpoints across interruptions:** Maintain identity-bound 44 + 67 shard results and immutable source provenance. Parallel jobs cannot assume their single runner persists after cancellation. Audit missing/duplicate shard intervals before rendering.
14. **Keep clean renders separate from stylized renders:** A separate workflow should use only verified original video bytes plus the already-selected 16 source ranges. It must not invoke ASR, selector/reviewer models, subtitle generation, hooks, watermark or text-overlay stages. Save its own QA report and distinguish it from the 16 edited MP4s. Source-embedded visuals/text cannot be removed by this requirement.
15. **Do not equate green CI or green media checks with creator-grade delivery:** Gate production states as *source verified → editorial decisions complete → MP4 rendered → media technically verified → manual factual/creative review → publication approval*. Keep publication disabled without the final explicit review.

## Acceptance conditions for the new clean-only render path

- Exactly the selected source ranges are used, in original selected order; no AI/editorial recomputation.
- Original file matches the archived source SHA-256; video/channel IDs agree.
- Every generated output is **1080 × 1920**, portrait **H.264 video + AAC audio**, with no subtitle stream and no newly added on-screen text or graphic overlay.
- The renderer and workflow perform **zero ASR, zero transcript parsing, zero model invocations**. Existing embedded source text, if any, is left unchanged.
- Produce a separate clean MP4 artifact and per-clip duration/hash/decode verification report.
- All results are **private review drafts only**, with manual editorial and campaign-policy approval still required.

## Evidence

- [Rendered media and editable reports](https://github.com/TheBayoumi/clipper/actions/runs/38469818017/artifacts/11861880015)
- [Full reconciled editorial decisions](https://github.com/TheBayoumi/clipper/actions/runs/38469818017/artifacts/11860880858)
- [Independent technical audit](https://github.com/TheBayoumi/clipper/actions/runs/38469818017/artifacts/11861302435)
- [Archived 44-candidate checkpoint](https://github.com/TheBayoumi/clipper/actions/runs/38364505410/artifacts/11819910219)


## Verified full-view vs. center-crop quality comparison (2026-10-14)

**Assessment workflow:** [Run #38552680233](https://github.com/TheBayoumi/clipper/actions/runs/38552680233). Source SHA-256 and 16 clean-clip identities were verified; **no new ASR/model work or clip rendering occurred**.

| Measure | SHA-verified original | Existing clean source-only render |
| --- | --- | --- |
| Geometry | 1920 × 1080 (16:9) | 1080 × 1920 (9:16) |
| Video codec | AV1 | H.264 |
| Average encoded/container bitrate | 0.779 Mb/s (whole original) | 4.548 Mb/s (weighted mean of 16 clips) |
| Unique source image region visible | Full 1920 × 1080 | Central 606 × 1080 |
| Original frame area retained | 100% | **31.5625%** |
| Original frame area discarded | 0% | **68.4375%** |
| Horizontal and vertical image scaling | Native dimensions | 1.782× × 1.778× interpolation |
| Real scene content added by upscaling | N/A | **None** |

The 16 clean outputs total 479.59 seconds and 272,643,327 MP4 bytes. Their output bitrates range approximately 3.82–5.07 Mb/s. These bitrates are **not** comparable as perceptual quality metrics across AV1 and H.264 with different frame regions. A larger H.264 output is not evidence of recovered original detail.

**Cause:** `scripts/tjr_clean_render.py` uses a fixed center-crop to 9:16 and scales the remaining 606×1080 image to 1080×1920. This is **not a motion/AI zoom**, but it looks like zooming and permanently removes both sides of every shot.

**Frame-based review evidence:** [Side-by-side contact sheet and quality JSON](https://github.com/TheBayoumi/clipper/actions/runs/38552680233/artifacts/11887895531), comparing the actual full source and clean crop at representative moments from clips 01, 04, 08 and 14. This provides framing evidence, **not** an SSIM/VMAF perceptual verdict. The current job does not compute frame-aligned PSNR/SSIM/VMAF; do not claim that those scores exist.

**Recommended frame policy:**
1. **Raw clean-review master:** retain the complete original 1920×1080 scene in source aspect ratio, without overlays, ASR, AI, or unnecessary upscaling.
2. **Portrait clean-review alternative:** fit the complete 16:9 frame inside 1080×1920 with neutral padding or a clearly identified background fill; never silently discard source pixels. The fitted picture would occupy approximately 1080×608 pixels, leaving vertical space to fill.
3. **Edited TikTok production:** choose scene-specific face/speaker tracking, split composition or justified selective reframing *only when the full-frame reference confirms nothing important is lost*. No blanket center-crop default.

**Open issue 16:** Separate clean review framing from creative crop decisions. Expose `source_full_frame` and `portrait_fit` as non-destructive modes, make the default preserve all footage, and report native-vs-output geometry, retained source area, and interpolation ratio in QA. Keep automated publication disabled.
