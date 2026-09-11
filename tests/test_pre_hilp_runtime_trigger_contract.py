from pathlib import Path

import yaml


def test_lovable_bootstrap_uses_immutable_tag_for_deploy_and_owner_only_hilp_trigger() -> None:
    workflow = Path(".github/workflows/lovable-production-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)
    assert "pull_request" in parsed[True]
    assert "edited" in parsed[True]["pull_request"]["types"]
    assert "contents: write" in workflow
    assert "actions: write" in workflow
    assert "github.actor == github.repository_owner" in workflow
    assert "CLIPPER_HILP_TRIGGER sha=" in workflow
    assert 'tag="clipper-hilp-${EXPECTED_SHA}"' in workflow
    assert '"ref": sys.argv[1], "sha": sys.argv[2]' in workflow
    assert "modal-workers-deploy.yml/dispatches" in workflow
    assert '"deployment_sha": sys.argv[2]' in workflow
    assert "production-pipeline.yml/dispatches" in workflow
    assert '"ref": sys.argv[1]' in workflow
    assert 'test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"' in workflow


def test_bootstrap_waits_for_the_specific_new_deployment_before_production() -> None:
    workflow = Path(".github/workflows/lovable-production-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    dispatch = workflow.index("modal-workers-deploy.yml/dispatches")
    resolve_new_run = workflow.index("selected_deployment_run_id")
    wait_for_success = workflow.index('deployment_conclusion" != "success"')
    production = workflow.index("production-pipeline.yml/dispatches")

    assert dispatch < resolve_new_run < wait_for_success < production
    assert "deploy-run-ids-before.json" in workflow
    assert 'int(item["id"]) not in before' in workflow
    assert "actions/runs/${selected_deployment_run_id}" in workflow
    assert "head_sha != sys.argv[1]" in workflow
    assert "head_branch != sys.argv[2]" in workflow
    assert "only after fresh deploy run" in workflow


def test_runtime_tag_name_is_content_addressed_to_full_sha() -> None:
    workflow = Path(".github/workflows/lovable-production-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    assert 'tag="clipper-hilp-${EXPECTED_SHA}"' in workflow
    assert "refs/tags/${tag}" in workflow
    assert "immutable HILP tag points at the wrong SHA" in workflow
