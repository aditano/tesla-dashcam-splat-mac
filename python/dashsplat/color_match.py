"""Per-camera affine RGB match against the front camera on road pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def fit_affine_rgb(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel scale+bias: target ≈ scale * source + bias on masked pixels."""
    assert source.shape == target.shape and source.shape[-1] == 3
    if mask is None:
        # Use bottom half (road-ish) and avoid saturated pixels
        h = source.shape[0]
        mask = np.zeros(source.shape[:2], dtype=bool)
        mask[h // 2 :, :] = True
        mid = (source.mean(axis=2) > 20) & (source.mean(axis=2) < 240)
        mask &= mid
    scales = np.ones(3, dtype=np.float64)
    biases = np.zeros(3, dtype=np.float64)
    for c in range(3):
        xs = source[..., c][mask].astype(np.float64)
        ys = target[..., c][mask].astype(np.float64)
        if xs.size < 64:
            continue
        # least squares [x 1] @ [s,b] = y
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
        scales[c], biases[c] = sol
    return scales, biases


def apply_affine(image: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32) * scale.reshape(1, 1, 3) + bias.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def match_cameras_to_front(
    frame_paths: dict[str, Path],
    out_dir: Path,
) -> dict[str, dict[str, list[float]]]:
    """frame_paths: camera -> image path. Writes color-matched copies; returns affines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if "front" not in frame_paths:
        # Nothing to match against — copy as-is
        result = {}
        for cam, p in frame_paths.items():
            dest = out_dir / p.name
            Image.open(p).save(dest)
            result[cam] = {"scale": [1, 1, 1], "bias": [0, 0, 0], "path": str(dest)}
        return result

    front = np.asarray(Image.open(frame_paths["front"]).convert("RGB"))
    affines: dict[str, dict[str, list[float]]] = {}
    for cam, p in frame_paths.items():
        img = np.asarray(Image.open(p).convert("RGB"))
        if cam == "front":
            scale, bias = np.ones(3), np.zeros(3)
            matched = img
        else:
            # Resize front to match if needed
            if img.shape != front.shape:
                front_r = np.asarray(
                    Image.fromarray(front).resize((img.shape[1], img.shape[0]))
                )
            else:
                front_r = front
            scale, bias = fit_affine_rgb(img, front_r)
            matched = apply_affine(img, scale, bias)
        dest = out_dir / p.name
        Image.fromarray(matched).save(dest, quality=92)
        affines[cam] = {
            "scale": [float(x) for x in scale],
            "bias": [float(x) for x in bias],
            "path": str(dest),
        }
    return affines
