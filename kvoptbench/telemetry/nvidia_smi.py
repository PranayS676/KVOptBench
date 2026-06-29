"""NVIDIA telemetry placeholder for later milestones."""

from __future__ import annotations


def collect_gpu_metrics() -> dict:
    return {
        "gpu_memory_used_gb": None,
        "gpu_memory_peak_gb": None,
        "reason": "GPU telemetry is not collected in Milestone 1 mock mode.",
    }

