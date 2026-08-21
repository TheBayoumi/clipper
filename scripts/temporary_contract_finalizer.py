from __future__ import annotations

import re
import textwrap
from pathlib import Path


def exact(path_name: str, old: str, new: str) -> None:
    path = Path(path_name)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"migration guard failed {path_name}: {old[:100]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def source_cleanup() -> None:
    workflow = Path(".github/workflows/temporary-mypy-contract-cleanup.yml").read_text(
        encoding="utf-8"
    )
    marker = "          python - <<'PY'\n"
    start = workflow.index(marker) + len(marker)
    end = workflow.index("          PY\n", start)
    script = textwrap.dedent(workflow[start:end])
    exec(compile(script, "<guarded-source-cleanup>", "exec"), {})

    pipeline = Path("src/clipper/pipeline.py")
    text = pipeline.read_text(encoding="utf-8")
    old = "quality: BatchQualityPlanningResult,"
    count = text.count(old)
    if count != 2:
        raise SystemExit(f"expected exactly two stale quality type annotations, found {count}")
    pipeline.write_text(text.replace(old, "quality: QualityBatchResult,"), encoding="utf-8")

    Path("src/clipper/_temporary_cleanup_trigger.py").unlink(missing_ok=True)
    Path("acceptance/temporary-contract-cleanup-trigger.json").unlink(missing_ok=True)
    Path("acceptance/temporary-source-finalizer-trigger.json").unlink(missing_ok=True)


def migrate_editorial_tests() -> None:
    Path("tests/test_editorial.py").write_text(
        '''import pytest

from clipper.editorial import (
    LegacyEditorialRemovedError,
    build_edit_plan,
    discover_story_moments,
    generate_hook_variants,
    mine_clip_concepts,
    select_distinct_concepts,
)
from clipper.models import CampaignBrief, ClipConcept, HookVariant


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="c",
        title="Campaign",
        objective="Autonomous quality",
        allowed_video_ids=["v"],
        rights_confirmed=True,
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: discover_story_moments(_brief(), "v", ()),
        lambda: mine_clip_concepts(_brief(), "v", (), ()),
        lambda: select_distinct_concepts(_brief(), ()),
        lambda: generate_hook_variants(_brief(), ClipConcept.__new__(ClipConcept), ()),
        lambda: build_edit_plan(
            _brief(),
            ClipConcept.__new__(ClipConcept),
            HookVariant.__new__(HookVariant),
            (),
        ),
    ],
)
def test_removed_lexical_editor_fails_closed(call) -> None:
    with pytest.raises(LegacyEditorialRemovedError, match="deterministic lexical editorial engine"):
        call()


def test_removed_editor_does_not_expose_legacy_scoring_or_hook_taxonomy() -> None:
    import clipper.editorial as editorial

    for name in (
        "_find_hook_sentence",
        "score_editorial_text",
        "cluster_concepts",
        "start_boundary_score",
        "end_boundary_score",
    ):
        assert not hasattr(editorial, name)
''',
        encoding="utf-8",
    )

    exact(
        "tests/test_editorial_integrity.py",
        "from clipper.models import AcceptancePolicy, CampaignBrief, ProductionConfig\n",
        "from clipper.models import AcceptancePolicy, CampaignBrief\n",
    )
    exact("tests/test_editorial_integrity.py", '        keywords=["interview"],\n', "")
    exact(
        "tests/test_editorial_integrity.py",
        "        production=ProductionConfig(final_render_budget=6, minimum_distinct_finalist_concepts=3),\n",
        "",
    )


def migrate_fixture_tests() -> None:
    Path("tests/test_fixture.py").write_text(
        '''import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from clipper.fixture import FixtureError, FixtureSourceClient, SpanMedia
from clipper.models import CampaignBrief


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, full_media: bool = False) -> Path:
    transcript = tmp_path / "source.en.vtt"
    transcript.write_text("WEBVTT\\n", encoding="utf-8")
    watermark = tmp_path / "watermark.png"
    watermark.write_bytes(b"watermark")
    media = tmp_path / "span.mp4"
    media.write_bytes(b"source-media")
    payload = {
        "video": {
            "video_id": "v1",
            "title": "Podcast",
            "channel_id": "UC1",
            "channel_title": "Channel",
            "url": "https://www.youtube.com/watch?v=v1",
            "duration_seconds": 100.0,
        },
        "transcript": {"file": transcript.name, "sha256": _hash(transcript)},
        "watermark": {
            "file": watermark.name,
            "sha256": _hash(watermark),
            "source_url": "https://example.test/watermark.png",
        },
        "spans": [
            {
                "file": media.name,
                "sha256": _hash(media),
                "source_origin": 8.0,
                "source_end": 25.0,
            }
        ],
    }
    if full_media:
        master = tmp_path / "full.mkv"
        master.write_bytes(b"full-authorized-media")
        payload["full_media"] = {
            "file": master.name,
            "sha256": _hash(master),
            "quality_policy": "highest_available_no_transcode",
        }
    (tmp_path / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="c",
        title="Campaign",
        objective="Clip",
        source_channel_ids=["UC1"],
        allowed_video_ids=["v1"],
        rights_confirmed=True,
        watermark_url="https://example.test/watermark.png",
    )


def test_fixture_source_verifies_identity_files_watermark_and_span(tmp_path: Path) -> None:
    client = FixtureSourceClient(_fixture(tmp_path))
    video = client.discover(_brief())[0]
    assert client.download_subtitles(video, tmp_path / "work", "en") == tmp_path / "source.en.vtt"
    span = client.download_media_span(video, 10.0, 20.0, tmp_path / "work")
    assert span == SpanMedia(tmp_path / "span.mp4", 8.0, 25.0, _hash(tmp_path / "span.mp4"))
    assert client.campaign_watermark(_brief()) == tmp_path / "watermark.png"
    with pytest.raises(FixtureError, match="no full media"):
        client.download_media(video, tmp_path / "work")
    with pytest.raises(FixtureError, match="no source span"):
        client.download_media_span(video, 1.0, 7.0, tmp_path / "work")


def test_fixture_rejects_unauthorized_identity_and_checksum(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    client = FixtureSourceClient(root)
    with pytest.raises(FixtureError, match="channel"):
        client.discover(replace(_brief(), source_channel_ids=["UC2"]))
    with pytest.raises(FixtureError, match="video"):
        client.discover(replace(_brief(), allowed_video_ids=["other"]))
    (root / "span.mp4").write_bytes(b"changed")
    with pytest.raises(FixtureError, match="checksum"):
        FixtureSourceClient(root)


def test_fixture_validates_paths_and_requests(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    client = FixtureSourceClient(root)
    wrong_video = replace(client.video, video_id="other")
    with pytest.raises(FixtureError, match="subtitle request"):
        client.download_subtitles(wrong_video, root, "en")
    with pytest.raises(FixtureError, match="media request"):
        client.download_media_span(wrong_video, 10, 12, root)
    assert client.campaign_watermark(replace(_brief(), watermark_url=None)) is None

    payload = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    payload["transcript"]["file"] = "../outside.vtt"
    outside = tmp_path.parent / "outside.vtt"
    outside.write_text("WEBVTT", encoding="utf-8")
    payload["transcript"]["sha256"] = _hash(outside)
    (root / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FixtureError, match="escapes"):
        FixtureSourceClient(root)


def test_fixture_manifest_failures_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="invalid fixture manifest"):
        FixtureSourceClient(tmp_path)
    (tmp_path / "fixture.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FixtureError, match="JSON object"):
        FixtureSourceClient(tmp_path)
    (tmp_path / "fixture.json").write_text(
        json.dumps({"video": {}, "spans": "bad"}), encoding="utf-8"
    )
    with pytest.raises(FixtureError, match="video and spans"):
        FixtureSourceClient(tmp_path)


def test_fixture_full_media_is_checksum_verified_and_covers_source_window(tmp_path: Path) -> None:
    client = FixtureSourceClient(_fixture(tmp_path, full_media=True))
    video = client.discover(_brief())[0]
    full = tmp_path / "full.mkv"
    assert client.download_media(video, tmp_path / "work") == full
    assert client.download_media_span(video, 10.0, 20.0, tmp_path / "work") == SpanMedia(
        full, 0.0, 100.0, _hash(full)
    )


def test_fixture_missing_or_mismatched_watermark_fails(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    client = FixtureSourceClient(root)
    with pytest.raises(FixtureError, match="does not match"):
        client.campaign_watermark(
            replace(_brief(), watermark_url="https://example.test/other.png")
        )
    payload = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    payload.pop("watermark")
    (root / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    without = FixtureSourceClient(root)
    with pytest.raises(FixtureError, match="does not provide"):
        without.campaign_watermark(_brief())
''',
        encoding="utf-8",
    )


def migrate_modal_recovery_tests() -> None:
    Path("tests/test_modal_editorial_recovery.py").write_text(
        '''from typing import Any

import pytest

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.editorial_prompt import (
    EDITORIAL_IDENTITY,
    EDITORIAL_SCHEMA_IDENTITY,
    editorial_contract,
    editorial_contract_fingerprint,
    editorial_json_schema,
    editorial_output_budget,
    editorial_task_family,
)
from clipper.providers.modal import ModalEditorialProvider, ModalRemoteError


class SequenceEditorialProvider(ModalEditorialProvider):
    def __init__(self, outcomes: list[object]) -> None:
        identity = ModelIdentity(
            "test/editorial", "rev", "none", "test", "editor", "structured-json"
        )
        super().__init__(app_name="test", function_name="editorial", identity=identity)
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
        self.requests.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, ProviderResult)
        return outcome


def _result(value: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
    identity = ModelIdentity(
        "test/editorial", "rev", "none", "test", "editor", "structured-json"
    )
    return ProviderResult(
        value,
        identity,
        InferenceUsage(provider="modal", started_at="now", duration_seconds=0.0),
    )


def test_editorial_provider_recovers_from_truncation_then_json_error() -> None:
    provider = SequenceEditorialProvider(
        [
            ModalRemoteError(
                function_name="editorial",
                error_type="EditorialOutputTruncated",
                message="cut",
            ),
            ModalRemoteError(
                function_name="editorial", error_type="JSONDecodeError", message="bad json"
            ),
            _result({"cores": []}),
        ]
    )
    result = provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert result.value == {"cores": []}
    assert [item.get("generation_recovery_attempt") for item in provider.requests] == [None, 2, 3]


def test_editorial_provider_does_not_retry_non_contract_remote_errors() -> None:
    provider = SequenceEditorialProvider(
        [ModalRemoteError(function_name="editorial", error_type="CUDAError", message="oom")]
    )
    with pytest.raises(ModalRemoteError, match="CUDAError"):
        provider.complete_json(task="semantic_cores:0", payload={})
    assert len(provider.requests) == 1


def test_active_editorial_contracts_are_structured_content_addressed_and_quota_free() -> None:
    tasks = (
        "source_hazards:0",
        "semantic_cores:0",
        "narrative_envelope:core-1",
        "quality_windows:core-1",
    )
    fingerprints = set()
    for task in tasks:
        assert editorial_task_family(task)
        contract = editorial_contract(task)
        schema = editorial_json_schema(task)
        assert schema["additionalProperties"] is False
        assert "never manufacture moments to satisfy a count" in contract
        assert "predeclared vocabulary" in contract
        fingerprints.add(editorial_contract_fingerprint(task))
    assert len(fingerprints) == 4
    assert EDITORIAL_IDENTITY == "editor"
    assert EDITORIAL_SCHEMA_IDENTITY == "structured-json"
    assert editorial_output_budget({"task": "narrative_envelope:core-1"}) == 1536
    assert editorial_output_budget({"task": "semantic_cores:0"}) == 2048


def test_legacy_editorial_task_families_are_rejected() -> None:
    for task in ("story_moments:0", "clip_concepts:0", "hook_variants:c", "edit_plans:c"):
        with pytest.raises(ValueError, match="unsupported production editorial task"):
            editorial_task_family(task)
''',
        encoding="utf-8",
    )


def migrate_import_only_tests() -> None:
    exact(
        "tests/test_open_models.py",
        "from clipper.autonomous_editor import AutonomousEditorialPlanner, OpenVideoAnalysis, _cosine\n",
        "from clipper.autonomous_editor import AutonomousEditorialPlanner, _cosine\n",
    )
    exact("tests/test_open_models.py", "    ProductionConfig,\n", "")
    exact("tests/test_open_models.py", "    EDITORIAL_PROMPT_VERSION,\n", "")
    exact("tests/test_open_models.py", "    EDITORIAL_SCHEMA_VERSION,\n", "")
    exact("tests/test_phase_b_contract_edges.py", "    editorial_legacy_cache_compatible,\n", "")
    exact("tests/test_pipeline.py", "    _campaign_media_candidates,\n", "")


def migrate_youtube_tests() -> None:
    exact(
        "tests/test_youtube.py",
        "from clipper.models import CampaignBrief, VideoCandidate\n",
        "from clipper.models import VideoCandidate\n",
    )
    exact(
        "tests/test_youtube.py",
        "from clipper.youtube import YouTubeClient, YouTubeError, _run\n",
        "from clipper.youtube import DiscoveryQuery, YouTubeClient, YouTubeError, _run\n",
    )
    path = Path("tests/test_youtube.py")
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(
        r"def brief\(\) -> CampaignBrief:\n.*?\n\n(?=def test_api_discovery_maps_results)",
        'def brief() -> DiscoveryQuery:\n    return DiscoveryQuery(query="agents", channel_ids=("UC1",), limit=20)\n\n\n',
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("youtube discovery helper migration failed")
    path.write_text(text, encoding="utf-8")


def migrate_cli_tests() -> None:
    path = Path("tests/test_cli.py")
    text = path.read_text(encoding="utf-8")
    old_payload = '''            {
                "campaign_id": "c",
                "title": "AI",
                "objective": "Goal",
                "keywords": ["automation"],
                "allowed_video_ids": ["v1"],
                "rights_confirmed": True,
            }
'''
    new_payload = '''            {
                "campaign_id": "c",
                "title": "AI",
                "objective": "Goal",
                "targets": {
                    "mode": "explicit",
                    "videos": [
                        {
                            "video_id": "v1",
                            "url": "https://www.youtube.com/watch?v=v1",
                            "channel_id": "UC1",
                        }
                    ],
                },
                "rights": {"confirmed": True, "authorized_channels": ["UC1"]},
            }
'''
    if text.count(old_payload) != 1:
        raise SystemExit("CLI brief payload migration failed")
    text = text.replace(old_payload, new_payload)
    text, count = re.subn(
        r"def test_cli_discover\(.*?(?=def test_cli_run_defaults_to_audited_open_v10)",
        '''def test_cli_discover_is_separate_from_production_targets(capsys) -> None:
    video = VideoCandidate("v2", "Title", "UC2", "Channel", "https://youtu.be/v2")
    with patch("clipper.cli.YouTubeClient") as client_cls:
        client_cls.return_value.discover.return_value = [video]
        assert main(["discover", "--query", "agents", "--channel-id", "UC2", "--limit", "7"]) == 0
        request = client_cls.return_value.discover.call_args.args[0]
    assert request.query == "agents"
    assert request.channel_ids == ("UC2",)
    assert request.limit == 7
    assert '"video_id": "v2"' in capsys.readouterr().out


''',
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("CLI discover test migration failed")
    text = text.replace('    monkeypatch.setenv("CLIPPER_WHISPER_MODEL", "base.en")\n', "")
    text = text.replace('    monkeypatch.delenv("CLIPPER_EDITORIAL_ENGINE", raising=False)\n', "")
    text = text.replace('    monkeypatch.delenv("CLIPPER_GROUNDING_ENGINE", raising=False)\n', "")
    text = text.replace('        assert settings.whisper_model == "base.en"\n', "")
    text = text.replace('        assert settings.editorial_engine == "open"\n', "")
    text = text.replace('        assert settings.grounding_engine == "open"\n', "")
    text, count = re.subn(
        r"def test_cli_refuses_accidental_legacy_and_local_lite\(.*?(?=def test_cli_refuses_local_lite_open_without_explicit_opt_in)",
        '''def test_cli_allows_local_lite_only_with_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    path = make_brief(tmp_path)
    monkeypatch.setenv("CLIPPER_COMPUTE_PROFILE", "local-lite")
    run_dir = tmp_path / "local-lite-run"
    write_open_manifest(run_dir)
    with (
        patch("clipper.cli._resolved_model_plan", return_value=open_plan()),
        patch("clipper.cli.run_pipeline", return_value=run_dir) as run,
    ):
        assert main(["run", "--brief", str(path), "--no-render", "--allow-local-lite"]) == 0
    assert run.call_args.kwargs["settings"].compute_profile == "local-lite"


''',
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("CLI local-lite migration failed")
    text = text.replace(
        '    settings = PipelineSettings(\n        editorial_engine="open", grounding_engine="open", compute_profile="balanced"\n    )\n',
        "",
    )
    text = text.replace(
        "_audit_model_evidence(run_dir, settings, open_plan())",
        "_audit_model_evidence(run_dir, open_plan())",
    )
    text = text.replace(
        "_audit_model_evidence(missing, settings, open_plan())",
        "_audit_model_evidence(missing, open_plan())",
    )
    text = text.replace(
        "_audit_model_evidence(no_manifest, settings, open_plan())",
        "_audit_model_evidence(no_manifest, open_plan())",
    )
    text = text.replace(
        "_audit_model_evidence(run_dir, settings, plan)",
        "_audit_model_evidence(run_dir, plan)",
    )
    path.write_text(text, encoding="utf-8")


def migrate_modal_execution_tests() -> None:
    path = Path("tests/test_modal_execution.py")
    text = path.read_text(encoding="utf-8")
    text = text.replace("    _authorized_candidates,\n", "    _explicit_candidates,\n")
    text, count = re.subn(
        r"def _brief\(.*?\n\n(?=def _write_brief)", "", text, count=1, flags=re.S
    )
    if count != 1:
        raise SystemExit("modal obsolete brief helper migration failed")
    text, count = re.subn(
        r"def _write_brief\(path: Path\) -> None:\n.*?\n\n(?=def test_function_hydrates_deployed_handle)",
        '''def _write_brief(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign",
                "title": "Podcast",
                "objective": "Find clips",
                "targets": {
                    "mode": "explicit",
                    "videos": [
                        {
                            "video_id": "v1",
                            "url": "https://www.youtube.com/watch?v=v1",
                            "channel_id": "UC1",
                        }
                    ],
                },
                "rights": {"confirmed": True, "authorized_channels": ["UC1"]},
            }
        ),
        encoding="utf-8",
    )


''',
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("modal brief migration failed")
    text, count = re.subn(
        r"def test_authorized_candidates_build_direct_youtube_requests_without_download\(\) -> None:.*?(?=def test_acquire_remote_source_uses_modal_egress_and_validates_quality)",
        '''def test_explicit_candidates_build_exact_requests_without_discovery(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidates = _explicit_candidates(brief_path)
    assert [item.video_id for item in candidates] == ["v1"]
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"
    assert candidates[0].channel_id == "UC1"


''',
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("modal candidate tests migration failed")
    text = text.replace(
        "clipper.modal_execution._authorized_candidates",
        "clipper.modal_execution._explicit_candidates",
    )
    text = text.replace('match="no authorized source"', 'match="no explicit authorized targets"')
    text, count = re.subn(
        r'''\n    with \(\n        patch\("clipper\.modal_execution\.ensure_modal_runtime"\),\n        patch\("clipper\.modal_execution\._function", return_value=Mock\(\)\),\n        patch\(\n            "clipper\.modal_execution\._explicit_candidates",\n            return_value=\[candidate, candidate\],\n        \),\n        pytest\.raises\(RuntimeError, match="source_limit=1"\),\n    \):\n        run_modal_pipeline\(.*?\n        \)\n''',
        "\n",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("modal single-source quota test migration failed")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    source_cleanup()
    migrate_editorial_tests()
    migrate_fixture_tests()
    migrate_modal_recovery_tests()
    migrate_import_only_tests()
    migrate_youtube_tests()
    migrate_cli_tests()
    migrate_modal_execution_tests()


if __name__ == "__main__":
    main()
