from types import SimpleNamespace

import pytest

from clipper.editorial_recovery import recover_gpu_evidence


def test_recovery_reads_only_submitted_job_files_and_weight_metadata(tmp_path):
    calls = []
    key = "a" * 64

    def read(path):
        calls.append(path)
        if path.endswith("manifest.json"):
            raise FileNotFoundError("Worker timed out before manifest")
        return iter([b'{"partial": ', b"true}"])

    volume = SimpleNamespace(
        read_file=read,
        listdir=lambda path: [SimpleNamespace(path="reviewer/weights/model.partial", size=123)],
    )
    result = recover_gpu_evidence(
        {"experiment": "source_position_gpu_qualification", "profiles": [{"job_key": key}]},
        tmp_path / "report.json",
        volume,
    )
    assert len(result["jobs"][0]["files"]) == 2
    assert len(result["jobs"][0]["errors"]) == 1
    assert result["weight_files"][0]["bytes"] == 123
    assert result["inference_started"] is False
    assert result["production_approved"] is False
    assert all(path.startswith(f"reviewer/qualification/{key}/") for path in calls)


@pytest.mark.parametrize("key", ["../other", "", None, "A" * 64])
def test_recovery_rejects_invalid_job_before_remote_access(tmp_path, key):
    with pytest.raises(ValueError, match="SHA-256"):
        recover_gpu_evidence(
            {"experiment": "source_position_gpu_qualification", "profiles": [{"job_key": key}]},
            tmp_path / "report.json",
            object(),
        )


def test_recovery_reports_unavailable_checkpoints_without_approving(tmp_path):
    result = recover_gpu_evidence(
        {"experiment": "source_position_gpu_qualification", "profiles": [{"job_key": "b" * 64}]},
        tmp_path / "report.json",
        object(),
    )
    assert len(result["jobs"][0]["errors"]) == 3
    assert result["jobs"][0]["files"] == []
    assert "weight_listing_error" in result
