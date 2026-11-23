"""Moving-object masks via RF-DETR (Apache-2.0) + ego-motion test.

Parked cars stay in the static scene. Ultralytics/YOLO (AGPL) is not used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def mover_mask_rfdetr(image_path: Path) -> np.ndarray | None:
    """Run RF-DETR segmentation if installed. Returns None if unavailable."""
    try:
        import rfdetr  # type: ignore  # noqa: F401
    except ImportError:
        return None
    # Integration point: when rfdetr + weights are present, return HxW keep-mask.
    # Until then, signal unavailable so callers use the motion heuristic.
    return None


def mover_mask_ego_motion(
    image_path: Path,
    prev_image_path: Path | None,
    *,
    speed_mps: float,
) -> np.ndarray:
    """Keep-mask: black out regions that move inconsistently with ego translation.

    When speed is near zero, do not mask (parked scene). Uses frame differencing
    as a coarse stand-in until RF-DETR tracks are wired.
    """
    img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    h, w = img.shape
    keep = np.ones((h, w), dtype=np.uint8) * 255
    if prev_image_path is None or abs(speed_mps) < 0.5:
        return keep
    prev = np.asarray(Image.open(prev_image_path).convert("L"), dtype=np.float32)
    if prev.shape != img.shape:
        prev = np.asarray(
            Image.open(prev_image_path).convert("L").resize((w, h)), dtype=np.float32
        )
    diff = np.abs(img - prev)
    moving = diff > 35.0
    keep[moving] = 0
    return keep


def mover_mask(
    image_path: Path,
    prev_image_path: Path | None = None,
    *,
    speed_mps: float = 0.0,
) -> np.ndarray:
    rf = mover_mask_rfdetr(image_path)
    if rf is not None:
        return rf
    return mover_mask_ego_motion(image_path, prev_image_path, speed_mps=speed_mps)
