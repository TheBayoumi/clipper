# Issue #8: offline audio-to-text cross-check (diagnostic)

On 2026-10-04, the three held-out MP4s from Actions run `36854533220` were present locally and matched the SHA-256 values in `tests/fixtures/issue8_heldout_claims.json`. Reconstructing `work/issue8-blind-audio-review.json` from those MP4s, the pinned transcript and provenance produced the same twelve-case manifest. All twelve `audible_claim_verdict` fields remain null.

An independent ASR pass used the locally cached `Systran/faster-whisper-base` snapshot `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`, `faster-whisper==1.2.1`, CPU/int8, `language=en`, `beam_size=5`, no VAD, and no conditioning on previous text. The model was loaded with `local_files_only=True`; no remote API or new media acquisition was used.

| Clip and pinned SHA-256 | ASR observations relative to clip start |
| --- | --- |
| `05-double-coverage-_kDrxucOx9g.mp4` — `77ec8ea92da940aa48e5bd2c46d1baec3a5f7eb3eb2ed62cec5596bd76cc65ea` | At 12.80–15.84 seconds, the ASR heard “he says I spar too much”; at 15.84–18.64 seconds, “going to be doing some reassessing in the future.” It garbled an earlier phrase about the podcast. |
| `07-double-coverage-_kDrxucOx9g.mp4` — `d525f3f5ef36029ff4fe2dd79b929ede41081b52a843880d6e33ed55664e2ed6` | At 1.36–3.40 seconds, “you guys are independent contractors”; at 8.28–12.64 seconds, “you can be a great fighter and get cut” and “they'll cut you if you're too boring.” The modal “can” matters: this is not evidence of an actual past cut. |
| `15-double-coverage-_kDrxucOx9g.mp4` — `829ff0bd5038fe10865d051703c63a67733b28249fcb55bdc7613777a4d61ff4` | At 12.96–18.36 seconds, the ASR heard a need for “an actual career plan”; at 23.04–26.72 seconds, an NFL/NBA contrast. Several connecting words and speaker boundaries are unclear in the ASR output. |

This is **not** independent semantic annotation or audio gold. ASR can mishear words and cannot establish who speaks, what is quoted, or whether every part of each proposed headline is true. It must not be used to fill the blind review fields, set a production threshold, or authorize rendering. A separate audio-first review remains pending for qualification; runtime factual approval must still be fully automated after qualification.
