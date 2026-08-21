from __future__ import annotations

import re
from pathlib import Path


path = Path("tests/test_open_models.py")
text = path.read_text(encoding="utf-8")
pattern = (
    r"^def test_open_analysis_rejects_bad_proposal_without_discarding_valid_moment\(.*?"
    r"(?=^def test_story_moment_alias_disambiguates_with_grounded_word_overlap\(\) -> None:)"
)
replacement = '''def test_legacy_open_analysis_method_is_absent(tmp_path: Path) -> None:
    planner = AutonomousEditorialPlanner(Mock(), Mock(), FileCache(tmp_path / "legacy-analysis"))
    assert not hasattr(planner, "analyze_video")


'''
updated, count = re.subn(pattern, replacement, text, count=1, flags=re.M | re.S)
if count != 1:
    raise SystemExit("final legacy open-analysis test anchor failed")
path.write_text(updated, encoding="utf-8")
