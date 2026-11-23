"""Fixed per-camera HW3 ego-car masks (hood / bumper / body in frame).

Defaults are hand-authored polygons in normalized image coords, refined from
temporal variance on real footage when available. White = keep, black = ignore
(Brush convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dashsplat.cameras import CameraId

# Normalized polygons (x,y in 0..1) covering typical ego-car pixels per camera.
EGO_POLYGONS: dict[CameraId, list[list[tuple[float, float]]]] = {
    "front": [
        # Hood / dashboard bottom
        [(0.0, 0.72), (1.0, 0.72), (1.0, 1.0), (0.0, 1.0)],
    ],
    "back": [
        # Bumper / license plate area
        [(0.15, 0.78), (0.85, 0.78), (0.95, 1.0), (0.05, 1.0)],
    ],
    "left_repeater": [
        # Body along right side of left-repeater view
        [(0.55, 0.35), (1.0, 0.25), (1.0, 1.0), (0.45, 1.0)],
    ],
    "right_repeater": [
        # Body along left side of right-repeater view
        [(0.0, 0.25), (0.45, 0.35), (0.55, 1.0), (0.0, 1.0)],
    ],
}


def ego_mask_array(camera: CameraId, width: int, height: int) -> np.ndarray:
    """Return HxW uint8 mask: 255=keep, 0=ego (ignore)."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for poly in EGO_POLYGONS.get(camera, []):
        pts = [(x * width, y * height) for x, y in poly]
        draw.polygon(pts, fill=0)
    return np.asarray(img, dtype=np.uint8)


def write_ego_mask(path: Path, camera: CameraId, width: int, height: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ego_mask_array(camera, width, height)
    Image.fromarray(arr, mode="L").save(path)
    return path
