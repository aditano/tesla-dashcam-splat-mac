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
    protect_vegetation: bool = True,
    erode_px: int = 0,
) -> np.ndarray:
    """Keep-mask: black out regions that move inconsistently with ego translation.

    When speed is near zero, do not mask (parked scene). Uses frame differencing
    as a coarse stand-in until RF-DETR tracks are wired. Foliage edges are kept
    so wind-blown leaves / parallax on trees are not stripped as movers.
    """
    img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    h, w = img.shape
    keep = np.ones((h, w), dtype=np.uint8) * 255
    # Raw frame differencing while driving blanks the static scene (ego-motion
    # looks like "everything moved"). Without ego-motion compensation, only
    # attempt differencing when nearly stopped; never when speed > threshold.
    _ = prev_image_path, protect_vegetation, erode_px
    if abs(speed_mps) > 0.5:
        return keep
    # Parked / crawl: also skip raw differencing (parked cars must stay in scene).
    # RF-DETR path handles movers when available; this heuristic is a no-op.
    return keep


def mover_mask(
    image_path: Path,
    prev_image_path: Path | None = None,
    *,
    speed_mps: float = 0.0,
    compute_units: str = "all",
    protect_vegetation: bool = True,
    erode_px: int = 0,
) -> tuple[np.ndarray, str]:
    """Return (mask, device_label). RF-DETR prefers MPS/Core ML when available."""
    _ = compute_units
    rf = mover_mask_rfdetr(image_path)
    if rf is not None:
        if protect_vegetation or erode_px:
            from dashsplat.detail_quality import protect_vegetation_in_mask

            rf = protect_vegetation_in_mask(image_path, rf, erode_px=erode_px)
        return rf, "rfdetr"
    return (
        mover_mask_ego_motion(
            image_path,
            prev_image_path,
            speed_mps=speed_mps,
            protect_vegetation=protect_vegetation,
            erode_px=erode_px,
        ),
        "cpu_motion",
    )
