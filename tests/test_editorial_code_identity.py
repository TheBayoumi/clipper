from pathlib import Path

import pytest

from clipper.editorial_code_identity import qualification_code_hash


def checkout(root):
    for name in (
        "src/clipper/editorial_engine.py",
        "src/clipper/editorial_source_scope.py",
        "scripts/tjr_semantic_editor.py",
        "scripts/tjr_modal_probe.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# identical source\n")
    return root


def test_qualification_identity_is_independent_of_checkout_location(tmp_path):
    left, right = checkout(tmp_path / "local"), checkout(tmp_path / "remote")
    assert qualification_code_hash(left) == qualification_code_hash(right)


@pytest.mark.parametrize(
    "name",
    [
        "src/clipper/editorial_engine.py",
        "src/clipper/editorial_source_scope.py",
        "src/clipper/new_dependency.py",
        "scripts/tjr_semantic_editor.py",
        "scripts/tjr_modal_probe.py",
    ],
)
def test_changed_or_added_dependency_invalidates_qualification(tmp_path, name):
    root = checkout(tmp_path)
    previous = qualification_code_hash(root)
    (root / name).write_bytes(b"# changed implementation\n")
    assert qualification_code_hash(root) != previous


def test_removed_dependency_invalidates_qualification(tmp_path):
    root = checkout(tmp_path)
    previous = qualification_code_hash(root)
    (root / "src/clipper/editorial_source_scope.py").unlink()
    assert qualification_code_hash(root) != previous


def test_missing_engine_cannot_have_a_qualification_identity(tmp_path):
    with pytest.raises(ValueError, match="package-owned editorial engine"):
        qualification_code_hash(tmp_path)


def test_worker_and_orchestrator_use_shared_identity():
    root = Path(__file__).resolve().parents[1]
    for name in ("tjr_semantic_editor.py", "tjr_modal_probe.py"):
        source = (root / "scripts" / name).read_text()
        assert "from clipper.editorial_code_identity import qualification_code_hash" in source
        assert "= qualification_code_hash(" in source
