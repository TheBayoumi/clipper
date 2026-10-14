"""Content identity for hosted editorial qualification, separate from selector caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def qualification_code_hash(project_root: Path) -> str:
    """Bind qualification to the shipped package and both orchestration scripts.

    A qualification may depend on any package helper. Hash the package tree
    rather than maintaining a partial list that becomes stale after refactors.
    Relative paths make local and remote checkout locations interchangeable.
    This identity is not used to invalidate production selector decisions.
    """
    package = project_root / "src" / "clipper"
    if not (package / "editorial_engine.py").is_file():
        raise ValueError("qualification requires the package-owned editorial engine")
    files = [
        *package.rglob("*.py"),
        project_root / "scripts" / "tjr_semantic_editor.py",
        project_root / "scripts" / "tjr_modal_probe.py",
    ]
    manifest = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
