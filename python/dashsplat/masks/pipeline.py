"""Compose ego + sky + mover masks into a Brush-compatible masks/ folder."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from dashsplat.cameras import CameraId
from dashsplat.masks.ego import ego_mask_array
from dashsplat.masks.movers import mover_mask
from dashsplat.masks.sky import sky_mask
from dashsplat.tiers import QualityTier, budget_for

ProgressCb = Callable[[str, float], None]


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Intersect keep-regions (255). Any 0 wins."""
    out = masks[0].astype(np.uint8)
    for m in masks[1:]:
        out = np.minimum(out, m.astype(np.uint8))
    return out


def build_masks_for_frames(
    frames: list[dict],
    out_dir: Path,
    *,
    tier: QualityTier | str = QualityTier.PREVIEW,
    compute_units: str = "all",
    progress: ProgressCb | None = None,
) -> tuple[Path, dict]:
    """frames: [{path, camera, file_name, speed_mps?}]. Writes PNGs named by file stem.

    Returns (masks_dir, device_report).
    """
    budget = budget_for(tier)
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_by_cam: dict[str, Path] = {}
    devices: dict[str, str] = {"sky": "unknown", "movers": "unknown", "compute_units": compute_units}
    for i, fr in enumerate(frames):
        if progress and i % 8 == 0:
            progress(f"Masks {i+1}/{len(frames)}…", 0.2 + 0.2 * i / max(len(frames), 1))
        path = Path(fr["path"])
        cam: CameraId = fr["camera"]  # type: ignore[assignment]
        with Image.open(path) as im:
            w, h = im.size
        ego = ego_mask_array(cam, w, h)
        sky, sky_dev = sky_mask(path, compute_units=compute_units)
        devices["sky"] = sky_dev
        if sky.shape != ego.shape:
            sky = np.asarray(Image.fromarray(sky).resize((w, h), Image.NEAREST))
        parts = [ego, sky]
        if budget.include_mover_masks:
            prev = prev_by_cam.get(cam)
            mv, mv_dev = mover_mask(path, prev, speed_mps=float(fr.get("speed_mps", 0.0)))
            devices["movers"] = mv_dev
            if mv.shape != ego.shape:
                mv = np.asarray(Image.fromarray(mv).resize((w, h), Image.NEAREST))
            parts.append(mv)
        combined = combine_masks(*parts)
        stem = Path(fr.get("file_name") or path.name).stem
        Image.fromarray(combined, mode="L").save(out_dir / f"{stem}.png")
        prev_by_cam[cam] = path
    return out_dir, devices
