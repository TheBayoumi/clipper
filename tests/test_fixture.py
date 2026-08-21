import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from clipper.fixture import FixtureError, FixtureSourceClient, SpanMedia
from clipper.models import CampaignBrief
from clipper.pipeline import PipelineSettings, run_pipeline


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    transcript = tmp_path / "source.en.vtt"
    transcript.write_text("WEBVTT\n")
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
    (tmp_path / "fixture.json").write_text(json.dumps(payload))
    return tmp_path


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="c",
        title="Campaign",
        objective="Clip",
        source_channel_ids=("UC1",),
        allowed_video_ids=("v1",),
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


def test_fixture_source_rejects_unauthorized_identity_and_checksum(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    client = FixtureSourceClient(root)
    unauthorized = CampaignBrief(
        campaign_id="c",
        title="Campaign",
        objective="Clip",
        source_channel_ids=("UC2",),
        allowed_video_ids=("v1",),
        rights_confirmed=True,
    )
    with pytest.raises(FixtureError, match="channel"):
        client.discover(unauthorized)
    (root / "span.mp4").write_bytes(b"changed")
    with pytest.raises(FixtureError, match="checksum"):
        FixtureSourceClient(root)


def test_fixture_source_rejects_watermark_mismatch_and_path_escape(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    client = FixtureSourceClient(root)
    wrong = replace(_brief(), watermark_url="https://example.test/other.png")
    with pytest.raises(FixtureError, match="watermark"):
        client.campaign_watermark(wrong)
    payload = json.loads((root / "fixture.json").read_text())
    payload["transcript"]["file"] = "../outside.vtt"
    outside = tmp_path.parent / "outside.vtt"
    outside.write_text("WEBVTT")
    payload["transcript"]["sha256"] = _hash(outside)
    (root / "fixture.json").write_text(json.dumps(payload))
    with pytest.raises(FixtureError, match="escapes"):
        FixtureSourceClient(root)


def test_fixture_manifest_validation_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FixtureError, match="invalid fixture manifest"):
        FixtureSourceClient(missing)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "fixture.json").write_text("[]")
    with pytest.raises(FixtureError, match="JSON object"):
        FixtureSourceClient(malformed)

    no_parts = tmp_path / "no-parts"
    no_parts.mkdir()
    (no_parts / "fixture.json").write_text(json.dumps({"video": {}, "spans": "wrong"}))
    with pytest.raises(FixtureError, match="video and spans"):
        FixtureSourceClient(no_parts)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "fixture.json").write_text(
        json.dumps({"video": {"video_id": "v"}, "spans": [{}]})
    )
    with pytest.raises(FixtureError, match="identity"):
        FixtureSourceClient(incomplete)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "fixture.json").write_text(
        json.dumps(
            {
                "video": {"video_id": "v", "channel_id": "UC", "url": "https://x"},
                "spans": ["not-an-object"],
            }
        )
    )
    with pytest.raises(FixtureError, match="no source spans"):
        FixtureSourceClient(empty)


def test_fixture_file_and_request_validation_errors(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    payload = json.loads((root / "fixture.json").read_text())
    payload["transcript"] = "bad"
    (root / "fixture.json").write_text(json.dumps(payload))
    with pytest.raises(FixtureError, match="entry is invalid"):
        FixtureSourceClient(root)

    root = _fixture(tmp_path)
    payload = json.loads((root / "fixture.json").read_text())
    payload["transcript"]["file"] = ""
    (root / "fixture.json").write_text(json.dumps(payload))
    with pytest.raises(FixtureError, match="path is missing"):
        FixtureSourceClient(root)

    root = _fixture(tmp_path)
    (root / "source.en.vtt").unlink()
    with pytest.raises(FixtureError, match="file is missing"):
        FixtureSourceClient(root)

    root = _fixture(tmp_path)
    client = FixtureSourceClient(root)
    wrong_video = replace(client.video, video_id="other")
    with pytest.raises(FixtureError, match="subtitle request"):
        client.download_subtitles(wrong_video, root, "en")
    with pytest.raises(FixtureError, match="media request"):
        client.download_media_span(wrong_video, 10, 12, root)
    no_mark = replace(_brief(), watermark_url=None)
    assert client.campaign_watermark(no_mark) is None
    payload = json.loads((root / "fixture.json").read_text())
    payload.pop("watermark")
    (root / "fixture.json").write_text(json.dumps(payload))
    client_without_mark = FixtureSourceClient(root)
    with pytest.raises(FixtureError, match="does not provide"):
        client_without_mark.campaign_watermark(_brief())


def _brief_video():
    from clipper.models import VideoCandidate

    return VideoCandidate("v1", "Podcast", "UC1", "Channel", "https://www.youtube.com/watch?v=v1")


class _FixtureRenderer:
    def render(self, source_path, output_path, clip, segments, watermark_path=None, edit_plan=None):
        assert source_path.name == "span.mp4"
        assert clip.start >= 0
        assert edit_plan is not None and edit_plan.source_spans[0].start >= 0
        assert segments
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return output_path


def test_pipeline_uses_span_fixture_and_records_span_hash(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    transcript = fixture / "source.en.vtt"
    transcript.write_text(
        "WEBVTT\n\n"
        "00:00:02.000 --> 00:00:12.000\n"
        "How did this automation story make one million dollars?\n\n"
        "00:00:12.100 --> 00:00:24.000\n"
        "It worked because the system saved time and the result changed everything.\n"
    )
    media = fixture / "span.mp4"
    media.write_bytes(b"real-source-window")
    manifest = {
        "video": {
            "video_id": "v1",
            "title": "Podcast",
            "channel_id": "UC1",
            "channel_title": "Channel",
            "url": "https://www.youtube.com/watch?v=v1",
            "duration_seconds": 30,
        },
        "transcript": {"file": "source.en.vtt", "sha256": _hash(transcript)},
        "spans": [
            {"file": "span.mp4", "sha256": _hash(media), "source_origin": 0, "source_end": 30}
        ],
    }
    (fixture / "fixture.json").write_text(json.dumps(manifest))
    brief = tmp_path / "brief.json"
    brief.write_text(
        json.dumps(
            {
                "campaign_id": "fixture-campaign",
                "title": "Podcast",
                "objective": "Clip",
                "keywords": ["automation", "money"],
                "source_channel_ids": ["UC1"],
                "allowed_video_ids": ["v1"],
                "rights_confirmed": True,
                "min_clip_seconds": 20,
                "max_clip_seconds": 30,
                "clip_count": 1,
                "production": {
                    "candidate_pool_size": 10,
                    "concept_count": 1,
                    "variants_per_concept": 1,
                    "final_render_budget": 1,
                },
                "hooks": {"enabled": ["direct"]},
            }
        )
    )
    client = FixtureSourceClient(fixture)
    from unittest.mock import patch

    with patch(
        "clipper.pipeline.run_technical_qc",
        return_value={"status": "PASS", "issues": [], "captions": {"alignment": "PASS"}},
    ):
        run_dir = run_pipeline(
            brief,
            settings=PipelineSettings(artifact_root=tmp_path / "artifacts"),
            source_client=client,
            renderer=_FixtureRenderer(),
        )
    result = json.loads((run_dir / "manifest.json").read_text())
    assert result["status"] == "SUCCESS"
    assert result["run_metadata"]["source_span_hashes"]["v1"]
    assert result["actual"]["rendered_finalists"] == 1


def test_fixture_source_can_supply_checksum_verified_full_media_for_open_grounding(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    full = tmp_path / "full.mkv"
    full.write_bytes(b"full-authorized-media")
    manifest_path = root / "fixture.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["full_media"] = {
        "file": "full.mkv",
        "sha256": _hash(full),
        "quality_policy": "highest_available_no_transcode",
    }
    manifest_path.write_text(json.dumps(manifest))
    client = FixtureSourceClient(root)
    video = client.discover(_brief())[0]
    assert client.download_media(video, tmp_path / "work") == full
    assert client.download_media_span(video, 10.0, 20.0, tmp_path / "work") == SpanMedia(
        full,
        0.0,
        100.0,
        _hash(full),
    )
