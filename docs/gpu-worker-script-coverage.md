# GPU-driving script coverage

The main Python package gate remains `pytest` with `--cov=clipper --cov-fail-under=95`.
This **separate** run uses `.coveragerc.scripts`, its own `.coverage.scripts`
data file, and targeted CPU-only tests with fake Modal/model boundaries.

The scope intentionally covers the expensive Wan S2V and Qwen keyframe workers,
their versioned input contracts, and the editorial endpoint bootstrap. It is
not an aggregate of the package and script denominators, nor a substitute
for live Modal/GPU deadline evidence.

GitHub Actions run 44850065774 at commit d67a30c6c4689ac56d16d6f197596426bae0b317
measured 303 covered statements of 498 (60.8433%) across the five include-list
scripts, using 35 targeted, mocked tests on both Python 3.11 and Python 3.12.
The regression floor is `--fail-under=60.84`: losing even one covered
statement without a compensating test fails this exact measured denominator.
Coverage additions should test real failure paths, not trivial constants.

W1's pending deadline design means deadline/termination tests are not yet claimed.
All GPU/model inference remains unverified; no live Modal worker was invoked.

## Rollout — CPU-first Qwen shared-cache lookup

`scripts/modal_qwen_keyframes.py` now calls the new **CPU Modal function**
`reuse_cached_keyframe` (same shared Volume) before any A100 GPU dispatch.
The function must be present in the deployed Modal App for the new local script
to work. Deploying that app requires **separate, explicit owner approval**.
**No Modal deployment was performed for this work.**

When a verified shared cache hit is found, the GPU function is not invoked;
the CPU lookup and Volume read still cost time/resources. A miss adds one CPU
round-trip before the unchanged GPU path; savings are not measured.

When the **new script** is run against a deployment missing this function, or
the CPU lookup raises any error, it persists `FRAME_GENERATION_FAILED`
with an actionable preflight error and **never silently falls through to GPU**.
A separately deployed **older script** may still follow the original GPU-first
path; this change does not retroactively replace deployed code. All runtime
behavior here is inferred from source and synthetic tests, not live execution.

Known residual: the worker validates the supplied request fingerprint as a
64-character string but does not independently recompute it. Fingerprinting
is done by `wan_keyframe_contract.prepare` before dispatch. Re-derivation was
not added because it would require coordinated changes to the test fixtures
and request validation assumptions; no cache schema was changed.
