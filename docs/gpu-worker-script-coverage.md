# GPU-driving script coverage

The main Python package gate remains `pytest` with `--cov=clipper --cov-fail-under=95`.
This **separate** run uses `.coveragerc.scripts`, its own `.coverage.scripts`
data file, and targeted CPU-only tests with fake Modal/model boundaries.

The scope intentionally covers the expensive Wan S2V and Qwen keyframe workers,
their versioned input contracts, and the editorial endpoint bootstrap. It is
not an aggregate of the package and script denominators, nor a substitute
for live Modal/GPU deadline evidence.

The first CI run records an honest measured baseline. Its follow-up change
will use the observed script coverage as a regression floor. W1's pending
deadline design means deadline/termination tests are not yet claimed.
