"""Per-camera time offset table (seconds) applied when pairing frames to SEI poses.

Defaults are zeros; calibration (Ceres / Ryan-style) can overwrite this table.
Rolling shutter: skip high yaw-rate frames in keyframes.py instead of Metal RS.
"""

from __future__ import annotations

import json
from pathlib import Path

from dashsplat.cameras import CameraId

# Seconds to add to each camera's wall time when sampling the trajectory.
DEFAULT_OFFSETS_SEC: dict[CameraId, float] = {
    "front": 0.0,
    "back": 0.0,
    "left_repeater": 0.0,
    "right_repeater": 0.0,
}


def load_offsets(path: Path | None = None) -> dict[str, float]:
    offsets = dict(DEFAULT_OFFSETS_SEC)
    if path and path.exists():
        data = json.loads(path.read_text())
        for k, v in data.items():
            offsets[k] = float(v)
    return offsets


def apply_offset(time_sec: float, camera: str, offsets: dict[str, float] | None = None) -> float:
    table = offsets or DEFAULT_OFFSETS_SEC
    return time_sec + float(table.get(camera, 0.0))


def save_offsets(path: Path, offsets: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(offsets, indent=2))
