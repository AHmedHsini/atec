"""Time the onboard stages. Does not train and does not replace weights."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from veil.model import WEIGHTS, load
from veil.perceive import detect
from veil.sim import render_pose
from veil.tracks import Tracker

ROOT = Path(__file__).resolve().parents[1]


def _rss_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage if os.name == "nt" else usage * 1024)
    except (ImportError, AttributeError):
        return None


def _cpu() -> str:
    return platform.processor() or platform.machine()


def _percentiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float) * 1000.0
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
    }


def run_bench(frames: int = 80, one_thread: bool = True) -> dict:
    if one_thread:
        cv2.setNumThreads(1)
    affinity = "not available on this platform"
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0})
            affinity = "pinned to cpu 0"
        except OSError as exc:
            affinity = f"sched_setaffinity failed: {exc}"
    net = load()
    params = int(net.w1.size + net.b1.size + net.w2.size + net.b2.size + net.mean.size + net.std.size)
    render_t, propose_t, total_t = [], [], []
    tracker = Tracker()
    rng = np.random.default_rng(0)
    cpu_start = time.process_time()
    wall = time.perf_counter()
    for frame in range(frames):
        t0 = time.perf_counter()
        image = render_pose((0.0, frame * 0.05), rng)
        t1 = time.perf_counter()
        found = detect(image, (0.0, frame * 0.05))
        t2 = time.perf_counter()
        tracker.update(found, frame * 0.05, frame * 0.1)
        t3 = time.perf_counter()
        render_t.append(t1 - t0)
        propose_t.append(t2 - t1)
        total_t.append(t3 - t0)
    elapsed = time.perf_counter() - wall
    return {
        "parameters": params,
        "weight_file_kb": round(WEIGHTS.stat().st_size / 1024, 2),
        "frames": frames,
        "render_ms": _percentiles(render_t),
        "perceive_ms": _percentiles(propose_t),
        "frame_ms": _percentiles(total_t),
        "fps_excluding_radio": round(frames / elapsed, 2),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_rss_bytes": _rss_bytes(),
        "affinity": affinity,
        "opencv_threads": cv2.getNumThreads(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cpu": _cpu(),
        "flight_weights": str(WEIGHTS),
        "note": "perceive_ms includes proposal, the NumPy network, projection, and geometric sigma. The flight weights were not retrained.",
    }


def main() -> None:
    payload = run_bench()
    out = ROOT / "artifacts" / "bench" / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"fps": payload["fps_excluding_radio"], "kb": payload["weight_file_kb"]}, indent=2))


if __name__ == "__main__":
    main()
