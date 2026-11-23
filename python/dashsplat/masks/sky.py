"""Sky segmentation via skyseg-coreml (MIT) when available; heuristic fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def sky_mask_heuristic(image_path: Path) -> np.ndarray:
    """Simple top-of-frame blue/bright sky heuristic → 255 keep, 0 sky (ignore)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    h, w, _ = arr.shape
    # Upper third: bright + bluish → sky
    sky = np.ones((h, w), dtype=np.uint8) * 255
    top = arr[: h // 3]
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    is_sky = (b > 140) & (b >= g) & (b >= r * 0.9) & ((r + g + b) > 400)
    # Also very bright near-white
    is_sky |= (r > 200) & (g > 200) & (b > 200)
    sky[: h // 3][is_sky] = 0
    return sky


def sky_mask(image_path: Path, *, use_coreml: bool = True) -> np.ndarray:
    if use_coreml:
        try:
            return _sky_coreml(image_path)
        except Exception:
            pass
    return sky_mask_heuristic(image_path)


def _sky_coreml(image_path: Path) -> np.ndarray:
    """Optional Core ML skyseg. Raises if unavailable (Linux CI / no model)."""
    # Placeholder hook — real Core ML inference lands when model is bootstrapped.
    raise RuntimeError("skyseg-coreml not installed in this environment")
