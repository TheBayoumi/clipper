"""Immutable Git revision contract shared by Clipper analysis, render and QA."""

from __future__ import annotations

import os
import re

_SHA = re.compile(r"[0-9a-f]{40}\Z")


def current_execution_sha() -> str:
    """Return the exact runner SHA; reject conflicting or invalid authorities."""
    acceptance = os.getenv("CLIPPER_ACCEPTANCE_SHA", "").strip().lower()
    github = os.getenv("GITHUB_SHA", "").strip().lower()
    if acceptance and github and acceptance != github:
        raise RuntimeError("Clipper acceptance SHA conflicts with GitHub checkout SHA")
    result = acceptance or github
    if result and not _SHA.fullmatch(result):
        raise RuntimeError("Clipper execution SHA must be exactly 40 lowercase hex characters")
    return result


def validate_artifact_sha(observed: object, *, stage: str) -> str:
    """In CI a missing or stale artifact revision is a hard production failure."""
    expected = current_execution_sha()
    observed_sha = str(observed or "").strip().lower()
    if expected and observed_sha != expected:
        raise RuntimeError(
            f"{stage}: execution SHA mismatch (expected {expected}, "
            f"artifact {observed_sha or '<missing>'})"
        )
    if observed_sha and not _SHA.fullmatch(observed_sha):
        raise RuntimeError(f"{stage}: malformed artifact execution SHA")
    return observed_sha
