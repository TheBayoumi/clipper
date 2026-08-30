from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch

from clipper.performance import (
    RunTelemetry,
    cpu_seconds,
    directory_size_bytes,
    gpu_utilization_pct,
    peak_rss_mb,
)


def test_directory_size_and_peak_rss(tmp_path: Path) -> None:
    assert directory_size_bytes(tmp_path / "missing") == 0
    (tmp_path / "a").write_bytes(b"abc")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b").write_bytes(b"12345")
    assert directory_size_bytes(tmp_path) == 8 and peak_rss_mb() > 0


def test_windows_telemetry_fallback_does_not_require_resource() -> None:
    fake_times = SimpleNamespace(
        user=1.25,
        system=0.75,
        children_user=0.0,
        children_system=0.0,
    )
    with (
        patch("clipper.performance._resource", None),
        patch("clipper.performance.os.times", return_value=fake_times),
    ):
        assert cpu_seconds() == 2.0
        assert peak_rss_mb() > 0.0


def test_gpu_utilization_handles_unavailable_failure_and_values() -> None:
    with patch("clipper.performance.shutil.which", return_value=None):
        assert gpu_utilization_pct() is None
    with (
        patch("clipper.performance.shutil.which", return_value="nvidia-smi"),
        patch(
            "clipper.performance.subprocess.run",
            return_value=CompletedProcess(["nvidia-smi"], 1, "", "err"),
        ),
    ):
        assert gpu_utilization_pct() is None
    with (
        patch("clipper.performance.shutil.which", return_value="nvidia-smi"),
        patch(
            "clipper.performance.subprocess.run",
            return_value=CompletedProcess(["nvidia-smi"], 0, "20\nbad\n40\n", ""),
        ),
    ):
        assert gpu_utilization_pct() == 30.0


def test_run_telemetry_records_stages_and_final_metrics(tmp_path: Path) -> None:
    (tmp_path / "artifact").write_bytes(b"x")
    with patch("clipper.performance.gpu_utilization_pct", side_effect=[10.0, 20.0]):
        telemetry = RunTelemetry()
        telemetry.start("analysis")
        assert telemetry.stop("analysis") >= 0
        assert telemetry.stop("missing") == 0
        result = telemetry.finish(tmp_path)
    assert result["wall_seconds"] >= 0 and result["cpu_seconds"] >= 0 and result["peak_rss_mb"] > 0
    assert result["artifact_bytes"] == 1 and result["gpu_available"] is True
    assert (
        result["gpu_utilization_samples_pct"] == [10.0, 20.0]
        and "analysis" in result["stages_seconds"]
    )
