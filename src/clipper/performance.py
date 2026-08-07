from __future__ import annotations

import resource
import shutil
import subprocess
import time
from pathlib import Path


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / 1024.0, 2)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def gpu_utilization_pct() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if process.returncode != 0:
        return None
    values: list[float] = []
    for line in process.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return round(sum(values) / len(values), 2) if values else None


class RunTelemetry:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.cpu_started = cpu_seconds()
        self.gpu_samples: list[float] = []
        self.stages: dict[str, float] = {}
        self._stage_started: dict[str, float] = {}
        initial_gpu = gpu_utilization_pct()
        if initial_gpu is not None:
            self.gpu_samples.append(initial_gpu)

    def start(self, name: str) -> None:
        self._stage_started[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        started = self._stage_started.pop(name, None)
        if started is None:
            return 0.0
        elapsed = time.perf_counter() - started
        self.stages[name] = round(self.stages.get(name, 0.0) + elapsed, 4)
        return elapsed

    def sample_gpu(self) -> None:
        value = gpu_utilization_pct()
        if value is not None:
            self.gpu_samples.append(value)

    def finish(self, artifact_root: str | Path) -> dict[str, object]:
        wall = max(1e-9, time.perf_counter() - self.started)
        cpu = max(0.0, cpu_seconds() - self.cpu_started)
        self.sample_gpu()
        return {
            "wall_seconds": round(wall, 4),
            "cpu_seconds": round(cpu, 4),
            "cpu_percent_equivalent": round(cpu / wall * 100.0, 2),
            "peak_rss_mb": peak_rss_mb(),
            "artifact_bytes": directory_size_bytes(artifact_root),
            "gpu_available": bool(self.gpu_samples),
            "gpu_utilization_samples_pct": self.gpu_samples,
            "stages_seconds": dict(self.stages),
        }
