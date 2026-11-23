"""HW3 dashcam rig config — ported from aditano/tesla-dashcam-splat src/config/hw3Cameras.ts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CameraId = Literal["front", "back", "left_repeater", "right_repeater"]


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


HW3_CAMERA_RIG: dict[CameraId, CameraExtrinsics] = {
    "front": CameraExtrinsics(
        position=(1.85, 0.0, 1.25),
        rotation_deg=(0, 0, 0),
        hfov_deg=50,
        fisheye=0.05,
        notes="Windshield main/narrow-ish dashcam encode; ~50° HFOV estimate.",
    ),
    "back": CameraExtrinsics(
        position=(-1.05, 0.0, 1.05),
        rotation_deg=(0, 180, 0),
        hfov_deg=130,
        fisheye=0.35,
        notes="License-plate / hatch rear camera; wide FOV.",
    ),
    "left_repeater": CameraExtrinsics(
        position=(1.55, 0.95, 0.85),
        rotation_deg=(0, -155, 0),
        hfov_deg=80,
        fisheye=0.45,
        notes="Left fender repeater, rear-biased; fisheye-ish.",
    ),
    "right_repeater": CameraExtrinsics(
        position=(1.55, -0.95, 0.85),
        rotation_deg=(0, 155, 0),
        hfov_deg=80,
        fisheye=0.45,
        notes="Right fender repeater, rear-biased; fisheye-ish.",
    ),
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
