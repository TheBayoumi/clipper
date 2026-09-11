from pathlib import Path

import yaml


def test_modal_deploy_executes_exact_head_speech_cuda_smoke_before_production() -> None:
    workflow = Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)

    handles = workflow.index("Verify required deployed model handles resolve")
    smoke = workflow.index("Verify immutable deployed SHA identities and speech CUDA runtime")
    editorial = workflow.index("Verify editorial generation runtime contract")
    endpoint = workflow.index("Verify production endpoint resolves without invoking paid work")
    assert handles < smoke < editorial < endpoint

    smoke_block = workflow[smoke:editorial]
    assert '"speech_cuda_smoke"' in workflow
    assert 'os.environ["CLIPPER_MODAL_APP"], "speech_cuda_smoke"' in smoke_block
    assert ".spawn()" in smoke_block
    assert "call.get(timeout=180)" in smoke_block
    assert 'speech.get("ok") is not True' in smoke_block
    assert 'value.get("deployed_git_sha")' in smoke_block
    assert 'speech.get("cuda_available") is not True' in smoke_block
    assert 'speech.get("device_count")' in smoke_block
    assert 'speech.get("cublas") != "libcublas.so.12"' in smoke_block
    assert 'speech.get("cudnn") != "libcudnn.so.9"' in smoke_block
    assert "modal-deployed-sha-identities.json" in smoke_block

    evidence = workflow[workflow.index("Record deployment evidence") :]
    assert '"speech_cuda_smoke"' in evidence
    assert "modal-deployed-sha-identities.json" in evidence
