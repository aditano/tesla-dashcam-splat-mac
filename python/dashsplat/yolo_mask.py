"""Moving-object masking (YOLO). Optional; static scene first.

On Mac, prefer a Core ML / MLX YOLO when present. For the rough v0 we provide
a lightweight motion-blob fallback so the pipeline never hard-depends on YOLO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

ProgressCb = Callable[[str, float], None]


def mask_moving_objects(
    image_paths: list[Path],
    *,
    progress: ProgressCb | None = None,
) -> list[np.ndarray]:
    """Return per-image boolean masks where True = keep (static)."""
    if progress:
        progress("Masking moving objects (motion fallback)…", 0.35)
    masks: list[np.ndarray] = []
    try:
        from PIL import Image
    except ImportError:
        return [np.ones((64, 64), dtype=bool) for _ in image_paths]

    prev = None
    for p in image_paths:
        img = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        if prev is None:
            masks.append(np.ones(img.shape, dtype=bool))
        else:
            diff = np.abs(img - prev)
            moving = diff > 28.0
            # Dilate roughly
            keep = ~moving
            masks.append(keep)
        prev = img
    return masks


def apply_masks_to_pointmap(
    pts3d: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-out confidence on masked (moving) pixels."""
    out_conf = conf.copy()
    for i, mask in enumerate(masks):
        if i >= out_conf.shape[0]:
            break
        h, w = out_conf[i].shape[:2]
        if mask.shape != (h, w):
            from PIL import Image

            m = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            )
            mask_r = m > 127
        else:
            mask_r = mask
        out_conf[i] = np.where(mask_r, out_conf[i], 0.0)
    return pts3d, colors, out_conf
