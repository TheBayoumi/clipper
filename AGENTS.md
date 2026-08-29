# Repository Guidelines

## Project Structure & Module Organization

Clipper is a Python 3.11-3.13 application using a `src/` layout. Production code lives in `src/clipper/`; provider integrations are isolated in `src/clipper/providers/`. Tests in `tests/` generally mirror package modules (`src/clipper/render.py` is covered by `tests/test_render.py`). Put operational utilities in `scripts/`, reusable campaign definitions in `campaigns/`, and CI/deployment automation in `.github/workflows/`. Treat `artifacts/`, `work/`, caches, and virtual environments as generated data; do not commit them.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install an editable development build:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,asr]"
```

- `make check` runs the full local gate: Ruff lint/format checks, strict mypy, and pytest.
- `pytest tests/test_cli.py -k validate` runs a focused test selection.
- `ruff check .` and `ruff format --check .` reproduce CI style checks.
- `clipper validate --brief campaign.yaml` validates campaign configuration.
- `clipper run --brief campaign.yaml --artifact-root artifacts --no-render` exercises analysis without rendering media.

## Coding Style & Naming Conventions

Use four-space indentation, complete type annotations, and a 100-character line limit. Ruff enforces imports and the configured `E`, `F`, `I`, `B`, `UP`, `S`, `SIM`, and `RUF` rules; mypy runs in strict mode. Name modules, functions, and variables `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Prefer small provider adapters over embedding service-specific logic in the pipeline.

## Testing Guidelines

Write pytest files as `test_<module>.py` and test functions as `test_<behavior>`. Use fixtures such as `tmp_path` and mock network, Modal, model, and filesystem boundaries so core tests remain deterministic. The suite must maintain at least 95% package coverage, configured in `pyproject.toml`. Add regression tests with every behavioral fix.

## Commit & Pull Request Guidelines

Follow the repository's concise Conventional Commit subjects: `feat:`, `fix:`, `test:`, `style:`, `perf:`, `refactor:`, or `ci:` followed by an imperative summary. Pull requests should explain behavior and risk, link relevant issues, list verification commands, and call out campaign or workflow changes. Include representative screenshots or rendered samples for visual-output changes, but never commit generated media.

## Security & Configuration

Copy `campaign.example.yaml` for local configuration. Keep API keys (for example, `YOUTUBE_API_KEY`) in the environment or `.env`, never YAML or source control. Preserve source-rights checks and the explicit publication boundary when modifying acquisition or release behavior.

## Expert Reviewer Contract

When acting as a reviewer, operate independently and adversarially rather than as the implementation
agent. Review the repository in execution-path context, not only the changed lines. Do not infer
runtime correctness from green CI, comments, README claims, acceptance markers, or historical
artifacts.

Reviewer findings must be evidence-based and prioritized P0-P3. For each actionable finding, identify
the exact file/range, concrete failure mode, reachable execution path or reproduction, why existing
tests or checks miss it, and the smallest safe corrective direction. Label concerns that require live
evidence as `NEEDS_RUNTIME_EVIDENCE` instead of guessing.

Always verify these production invariants when they are affected:

- exact-head acceptance must prove the immutable SHA embedded in the deployed Modal workers;
- Modal spy evidence must be scoped to the exact spawned production execution and its correlated
  descendants, never global app logs;
- source authorization and explicit-target rights gates must fail closed;
- publication must remain behind the explicit human review boundary;
- production open-model execution must not silently fall back to a weaker/local path;
- fresh inference must not reuse grounding or editorial inference caches;
- content-addressed resume must reuse only contract-compatible artifacts;
- evidence projection must reduce LLM-facing evidence without changing canonical source truth;
- capacity rejection must happen before unsafe generation;
- runtime-safe token guards must be enforced in addition to raw model context limits;
- timeout, OOM, and context failures must either repartition with measurable forward progress or fail
  closed;
- repartition ranges must terminate, remain contiguous and ordered, and contain no gaps or overlap;
- a timed-out or stalled generation must never count toward a passing acceptance result;
- watchdogs must cancel only the exact offending production call;
- compute budgets must be enforced during execution, not only audited after completion;
- persistent DAG writes must remain valid under concurrent writers;
- tests must preserve the repository-wide >=95% coverage floor while asserting behavior rather than
  implementation text alone.

For review-only requests, do not push commits, update refs, dispatch/rerun/cancel GitHub Actions, call
production endpoints, deploy Modal workers, or start paid compute unless the user explicitly asks for
implementation or execution after the review.

## Mandatory Codex PR Gate

Every pull-request head is reviewable only after an independent Codex review of that exact 40-character
head SHA. The repository CI must fail closed when the exact-head review has not completed or while any
Codex P0, P1, or P2 review thread remains unresolved. A Codex P1/P2 marked `NEEDS_RUNTIME_EVIDENCE`
remains blocking until the required runtime evidence is produced and the thread is resolved; static
reasoning alone must not clear it.

The required check is `codex-review-gate`. It must verify both of these conditions against live GitHub
PR state:

1. Codex (`chatgpt-codex-connector`) submitted a review whose `commit_id` equals the current PR head,
   or reacted with `+1` to an explicit `@codex review` request containing that exact full head SHA.
2. No unresolved review thread authored by Codex contains a P0, P1, or P2 finding.

A new commit invalidates the previous Codex gate result and requires a new exact-head review. Do not
resolve blocking Codex threads merely because implementation changed; resolve them only when the fix
is present and verified, and keep runtime-evidence findings open until the evidence exists.

Modal deployment, production/editorial acceptance, HILP, rendering, and publication must never be used
to bypass this gate. Workflows that require successful exact-head CI therefore inherit the Codex gate.
Repository branch protection/rulesets should additionally mark `codex-review-gate` as a required status
check before merge wherever repository administration permits it.
