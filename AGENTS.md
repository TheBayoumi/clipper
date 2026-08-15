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
