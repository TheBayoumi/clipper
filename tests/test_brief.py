import json
from pathlib import Path

import pytest

from clipper.brief import load_brief
from clipper.models import BriefValidationError


DATA = {
    "campaign_id": "c1",
    "title": "Title",
    "objective": "Goal",
    "keywords": ["one"],
    "allowed_video_ids": ["abc"],
    "rights_confirmed": True,
}


def test_load_json_and_yaml(tmp_path: Path) -> None:
    json_path = tmp_path / "brief.json"
    json_path.write_text(json.dumps(DATA), encoding="utf-8")
    assert load_brief(json_path).campaign_id == "c1"

    yaml_path = tmp_path / "brief.yaml"
    yaml_path.write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in DATA.items()),
        encoding="utf-8",
    )
    assert load_brief(yaml_path).title == "Title"


def test_load_unknown_extension_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "brief.txt"
    path.write_text(json.dumps(DATA), encoding="utf-8")
    assert load_brief(path).objective == "Goal"


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_brief("does-not-exist.json")


def test_unknown_extension_yaml_and_bad_yaml_root(tmp_path: Path) -> None:
    path = tmp_path / "brief.conf"
    path.write_text(
        "campaign_id: c1\n"
        "title: T\n"
        "objective: G\n"
        "keywords: [one]\n"
        "allowed_video_ids: [v]\n"
        "rights_confirmed: true\n",
        encoding="utf-8",
    )
    assert load_brief(path).campaign_id == "c1"

    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- an\n- object\n", encoding="utf-8")
    with pytest.raises(BriefValidationError, match="root"):
        load_brief(bad)
