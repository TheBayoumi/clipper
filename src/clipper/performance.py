from __future__ import annotations

import ctypes
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_resource: Any | None
try:
    _resource = importlib.import_module("resource")
except ModuleNotFoundError:
    _resource = None


def cpu_seconds() -> float:
    if _resource is not None:
        own = _resource.getrusage(_resource.RUSAGE_SELF)
        children = _resource.getrusage(_resource.RUSAGE_CHILDREN)
        total = own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
        return float(total)
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


def _proc_peak_rss_mb() -> float:
    """Best-effort Linux RSS fallback when the stdlib resource module is unavailable."""

    status = Path("/proc/self/status")
    if not status.is_file():
        return 0.0
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    return round(float(fields[1]) / 1024.0, 2)
    except (OSError, ValueError):
        return 0.0
    return 0.0


def peak_rss_mb() -> float:
    if _resource is None:
        if sys.platform != "win32":
            return _proc_peak_rss_mb()

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        windows_api: Any = ctypes.windll
        windows_api.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        windows_api.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        windows_api.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = windows_api.kernel32.GetCurrentProcess()
        ok = windows_api.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if not ok:
            return 0.0
        peak_bytes = int(counters.peak_working_set_size)
        return float(round(peak_bytes / (1024.0 * 1024.0), 2))
    usage = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(usage / (1024.0 * 1024.0), 2)
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
