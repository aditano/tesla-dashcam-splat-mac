"""Sky segmentation via skyseg-coreml (MIT) when available; heuristic fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def sky_mask_heuristic(image_path: Path, *, protect_vegetation: bool = True) -> np.ndarray:
    """Simple top-of-frame blue/bright sky heuristic → 255 keep, 0 sky (ignore).

    Does not mark green foliage as sky (trees against bright sky).
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    h, w, _ = arr.shape
    sky = np.ones((h, w), dtype=np.uint8) * 255
    top = arr[: h // 3]
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    is_sky = (b > 140) & (b >= g) & (b >= r * 0.9) & ((r + g + b) > 400)
    is_sky |= (r > 200) & (g > 200) & (b > 200)
    if protect_vegetation:
        foliage = (g > r * 1.05) & (g > b * 1.05) & (g > 40)
        is_sky &= ~foliage
    sky[: h // 3][is_sky] = 0
    return sky


def sky_mask(
    image_path: Path,
    *,
    use_coreml: bool = True,
    compute_units: str = "all",
    protect_vegetation: bool = True,
    erode_px: int = 0,
) -> tuple[np.ndarray, str]:
    """Return (mask, device_label). Prefer Core ML with computeUnits=.all on Mac."""
    if use_coreml:
        try:
            mask = _sky_coreml(image_path, compute_units=compute_units)
            if protect_vegetation or erode_px:
                from dashsplat.detail_quality import protect_vegetation_in_mask

                mask = protect_vegetation_in_mask(image_path, mask, erode_px=erode_px)
            return mask, f"coreml:{compute_units}"
        except Exception:
            pass
    mask = sky_mask_heuristic(image_path, protect_vegetation=protect_vegetation)
    if erode_px > 0:
        from dashsplat.detail_quality import protect_vegetation_in_mask

        mask = protect_vegetation_in_mask(image_path, mask, erode_px=erode_px)
    return mask, "cpu_heuristic"


def _sky_coreml(image_path: Path, *, compute_units: str = "all") -> np.ndarray:
    """Optional Core ML skyseg. Raises if unavailable (Linux CI / no model).

    On macOS, Core ML should be configured with computeUnits covering ANE+GPU
    (never CPU-only when Apple Silicon is available).
    """
    _ = (image_path, compute_units)
    # Placeholder hook — real Core ML inference lands when model is bootstrapped.
    raise RuntimeError("skyseg-coreml not installed in this environment")
