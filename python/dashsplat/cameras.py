"""Tesla dashcam rig configs (HW3 first-class; HW4 when pillars present).

Extrinsics / HFOV below are **approximations** for init — not factory calibration.
Cite / refine with:
- Community dashcam tooling + SEI: https://github.com/teslamotors/dashcam
- R44VC0RP/dashsplat (Apache-2.0) self-calibrates lenses/mounts from HW4 footage;
  upstream notes HW3 four-cam is untested there:
  https://github.com/R44VC0RP/dashsplat
- Encode size/fps (probe preferred): 1280×960 @ ~36 fps reported for HW3 dashcam
  encodes (conscientsystems/teslacam, kandotrun/teslacam-viewer, nasrtech repair notes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CameraId = Literal[
    "front",
    "back",
    "left_repeater",
    "right_repeater",
    "left_pillar",
    "right_pillar",
]

ALL_CAMERA_IDS: tuple[CameraId, ...] = (
    "front",
    "back",
    "left_repeater",
    "right_repeater",
    "left_pillar",
    "right_pillar",
)


@dataclass(frozen=True)
class CameraExtrinsics:
    position: tuple[float, float, float]  # vehicle frame: +X fwd, +Y left, +Z up
    rotation_deg: tuple[float, float, float]  # yaw-ish about Z / body
    hfov_deg: float
    width: int = 1280
    height: int = 960
    fps: float = 36.0
    fisheye: float = 0.0
    notes: str = ""


# Approximate HW3 dashcam rig (4 cameras). Positions/HFOV are estimates for MapAnything init.
HW3_CAMERA_RIG: dict[CameraId, CameraExtrinsics] = {
    "front": CameraExtrinsics(
        position=(1.85, 0.0, 1.25),
        rotation_deg=(0, 0, 0),
        hfov_deg=50,
        fisheye=0.05,
        notes="Approx. windshield main dashcam encode; ~50° HFOV estimate (not factory).",
    ),
    "back": CameraExtrinsics(
        position=(-1.05, 0.0, 1.05),
        rotation_deg=(0, 180, 0),
        hfov_deg=130,
        fisheye=0.35,
        notes="Approx. hatch/rear camera; wide FOV estimate.",
    ),
    "left_repeater": CameraExtrinsics(
        position=(1.55, 0.95, 0.85),
        rotation_deg=(0, -155, 0),
        hfov_deg=80,
        fisheye=0.45,
        notes="Approx. left fender repeater, rear-biased.",
    ),
    "right_repeater": CameraExtrinsics(
        position=(1.55, -0.95, 0.85),
        rotation_deg=(0, 155, 0),
        hfov_deg=80,
        fisheye=0.45,
        notes="Approx. right fender repeater, rear-biased.",
    ),
}

# HW4 adds B-pillar cameras (dashcam/Sentry). Positions are approximations.
HW4_PILLAR_RIG: dict[CameraId, CameraExtrinsics] = {
    "left_pillar": CameraExtrinsics(
        position=(0.55, 0.55, 1.15),
        rotation_deg=(0, -90, 0),
        hfov_deg=90,
        fisheye=0.25,
        notes="Approx. left B-pillar (HW4). Prefer self-calibration when available.",
    ),
    "right_pillar": CameraExtrinsics(
        position=(0.55, -0.55, 1.15),
        rotation_deg=(0, 90, 0),
        hfov_deg=90,
        fisheye=0.25,
        notes="Approx. right B-pillar (HW4). Prefer self-calibration when available.",
    ),
}

CAMERA_RIG: dict[CameraId, CameraExtrinsics] = {
    **HW3_CAMERA_RIG,
    **HW4_PILLAR_RIG,
}


def focal_length_px(hfov_deg: float, width: int) -> float:
    import math

    return (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)


def intrinsics_matrix(cam: CameraExtrinsics) -> list[list[float]]:
    fx = focal_length_px(cam.hfov_deg, cam.width)
    fy = fx
    cx = cam.width / 2.0
    cy = cam.height / 2.0
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def rig_for_cameras(cameras: list[str] | tuple[str, ...]) -> dict[CameraId, CameraExtrinsics]:
    return {c: CAMERA_RIG[c] for c in cameras if c in CAMERA_RIG}  # type: ignore[misc]
