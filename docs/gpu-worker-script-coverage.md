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
