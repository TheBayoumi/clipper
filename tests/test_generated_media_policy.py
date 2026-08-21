from pathlib import Path

import pytest

from clipper.brief import load_brief
from clipper.generated_media import (
    GeneratedMediaBlocked,
    GeneratedMediaRequest,
    generate_policy_gated_media,
)
from clipper.models import CampaignBrief


class FakeGenerator:
    provider_name = "fake"
    model_id = "fake/model"
    model_revision = "rev"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GeneratedMediaRequest, output_path: Path) -> Path:
        self.calls += 1
        output_path.write_bytes(request.prompt.encode())
        return output_path


def _brief(policy: str) -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "campaign",
            "title": "Campaign",
            "objective": "Test",
            "keywords": ["test"],
            "rights_confirmed": True,
            "acceptance_policy": {
                "generated_media": {"ai_generated_source_video": policy},
            },
        }
    )


def _request() -> GeneratedMediaRequest:
    return GeneratedMediaRequest("quality:1", "illustrate source-supported idea", ("evidence-1",))


def test_double_coverage_forbid_policy_results_in_zero_generator_calls(tmp_path: Path) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    provider = FakeGenerator()
    with pytest.raises(GeneratedMediaBlocked, match="forbids"):
        generate_policy_gated_media(brief, provider, _request(), tmp_path / "asset.mp4")
    assert provider.calls == 0
    assert not (tmp_path / "asset.mp4").exists()


def test_escalate_policy_also_blocks_before_provider_invocation(tmp_path: Path) -> None:
    provider = FakeGenerator()
    with pytest.raises(GeneratedMediaBlocked, match="escalation"):
        generate_policy_gated_media(_brief("escalate"), provider, _request(), tmp_path / "asset.mp4")
    assert provider.calls == 0


def test_allow_policy_records_generated_asset_provenance(tmp_path: Path) -> None:
    provider = FakeGenerator()
    output = tmp_path / "asset.mp4"
    asset = generate_policy_gated_media(_brief("allow"), provider, _request(), output)
    assert provider.calls == 1
    assert asset.path == output
    assert asset.sha256
    assert asset.to_dict()["provider"] == "fake"


def test_generated_media_request_must_be_silent_and_source_grounded() -> None:
    with pytest.raises(ValueError, match="source evidence"):
        GeneratedMediaRequest("quality:1", "prompt", ())
    with pytest.raises(ValueError, match="silent"):
        GeneratedMediaRequest("quality:1", "prompt", ("evidence",), silent=False)
