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

## Mandatory Independent Reviewer Contract

Codex is the mandatory independent adversarial reviewer for the current repository workflow. The repository custom agent `Clipper Adversarial Reviewer` is optional and dormant while GitHub Copilot agent access is unavailable; its absence must not block review or acceptance. Every material PR head must receive a fresh Codex review after implementation and deterministic verification are complete. A review of an older SHA is historical evidence only and must never be represented as approval of a newer head.

The reviewer must inspect the repository in execution-path context, including relevant callers, callees, workflows, state transitions, persistence boundaries, failure handling, and tests. Do not limit review to the changed lines. Do not infer correctness from green CI, comments, README claims, acceptance markers, prior reviews, or implementation-agent assertions.

### Review independence and anti-gaming rules

- Never modify this file, reviewer prompts, tests, comments, workflow text, or acceptance criteria merely to steer Codex toward a clean verdict.
- Never weaken, hide, reclassify, suppress, auto-resolve, or omit a finding to satisfy an acceptance gate.
- Never treat a reviewer reaction, acknowledgement, absence of comments, or stale review as proof of correctness.
- Never ask Codex to implement its own findings during the independent review pass. Implementation and review are separate phases.
- Never resolve a blocking review thread solely because code changed. Verify the corrective behavior first; runtime-evidence findings remain open until the required live evidence exists.
- A new material commit after review invalidates the prior exact-head acceptance verdict and requires another independent review.
- If reviewer infrastructure is unavailable, fail the acceptance process closed rather than replacing independent review with self-review.

### Required review output

Classify findings P0-P3. Every actionable P0/P1/P2 finding must identify the exact file/range, concrete failure mode, reachable execution path or reproduction, affected invariant, why current tests/checks fail to prove safety, and the smallest safe corrective direction. Distinguish static defects from questions that genuinely require deployment or live execution. Mark the latter `NEEDS_RUNTIME_EVIDENCE`; do not guess either success or failure.

A review is not clean while any actionable P0/P1/P2 remains. P3 findings may be non-blocking only when they cannot violate correctness, safety, acceptance integrity, cost bounds, or publication boundaries. Conflicting evidence must be called out explicitly rather than averaged into a verdict.

### Mandatory production invariants

When affected by a change, independently verify all applicable invariants:

- exact-head acceptance proves the immutable source SHA embedded in every deployed Modal worker involved in the execution;
- runtime evidence is correlated to the exact spawned root execution and its descendants, never inferred from global or temporally adjacent logs;
- every tracked producer has a one-to-one lifecycle with terminal evidence, and PASS is impossible while a producer remains active, lost, ambiguous, or uncorrelated;
- source authorization and explicit-target rights gates fail closed;
- publication remains behind the explicit human review boundary;
- production open-model execution cannot silently fall back to a weaker, local, cached, or untracked path;
- fresh-inference mode cannot reuse grounding or editorial inference caches that would invalidate the proof;
- content-addressed resume reuses only artifacts whose complete contract and source identity remain compatible;
- editorial evidence projection reduces LLM-facing evidence without mutating, dropping, duplicating, reordering, or fabricating canonical source truth;
- generation arithmetic accounts for prompt tokens, requested output, reserved capacity, model context, runtime-safe limits, and late candidates consistently;
- capacity rejection occurs before unsafe generation whenever the unsafe condition is knowable before invocation;
- timeout, OOM, context, and capacity failures repartition only with measurable forward progress or fail closed;
- repartition ranges terminate, strictly reduce unsafe work, preserve contiguous ordered source coverage, and introduce no gaps, overlap, duplication, or unbounded work amplification;
- minimum-span failure is deterministic when further safe repartition is impossible;
- timed-out, cancelled, stalled, over-budget, or late generation can never become accepted editorial evidence or a passing acceptance result;
- primary generation deadlines are enforced by the real deployed generation path; watchdog cancellation is a correlated fallback, not evidence that the primary deadline works;
- watchdogs and abort handlers cancel only the exact offending Modal function call and cannot terminate unrelated executions;
- compute/GPU/cost budgets are finite, validated before execution, enforced while work is in flight, and checked again before PASS;
- persistent capacity/DAG/cache state remains valid under concurrent writers, atomic replacement, retries, and partial failures;
- a successful inference cannot be converted into an unrelated failure solely by best-effort persistence or telemetry cleanup;
- deployment workflows use the minimum GitHub permissions required for every API they invoke and fail closed on identity mismatches;
- editorial-only acceptance cannot cross the render, HILP, publication, or other paid-compute boundary that the run did not explicitly authorize;
- tests maintain the repository-wide >=95% coverage floor and assert observable behavior and failure paths rather than implementation text alone.

### Runtime evidence standard

Static source inspection can prove wiring and invariants that are deterministic from code, but it cannot substitute for properties of the deployed runtime. For `NEEDS_RUNTIME_EVIDENCE`, require correlated evidence from the exact accepted SHA and execution. Evidence must identify relevant deployed model/package versions and runtime identity, invocation IDs, start/terminal lifecycle, timing/deadline behavior, failure classification, repartition decisions, budget state, and final terminal barrier as applicable.

For generation deadlines specifically, adapter/source inspection may prove that a parameter is forwarded, but acceptance requires live evidence that the deployed generation call actually stops at the configured boundary, rejects any late candidate, closes the producer with the correct non-success terminal state, and either makes strictly smaller repartition progress or fails deterministically at minimum span. A larger watchdog timeout is only fallback evidence.

### Review and execution separation

For review-only requests, Codex must remain read-only: it must not push commits, update refs, create branches, edit files, dispatch/rerun/cancel GitHub Actions, resolve review threads, call production endpoints, deploy Modal workers, start paid compute, alter acceptance markers, or publish artifacts. It may inspect existing repository state and already-produced evidence. Implementation or live execution begins only after an explicit user request outside the independent review pass.

The implementation agent must not claim the PR is acceptance-ready until deterministic tests and exact-head CI are green, a fresh independent Codex review has no unresolved static P0/P1/P2 findings, and every blocking `NEEDS_RUNTIME_EVIDENCE` item has been satisfied by the required correlated runtime proof. HILP/render/full production acceptance remains disabled until those prerequisites are satisfied.
