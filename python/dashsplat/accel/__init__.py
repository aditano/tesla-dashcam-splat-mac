"""Apple Silicon acceleration helpers (device detect, decode, tune)."""

from dashsplat.accel.device import ComputeDevice, HostInfo, detect_host
from dashsplat.accel.tune import TunedBudget, tune_for_host

__all__ = [
    "ComputeDevice",
    "HostInfo",
    "TunedBudget",
    "detect_host",
    "tune_for_host",
]
