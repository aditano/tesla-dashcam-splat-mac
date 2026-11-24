"""Per-camera color match using global / histogram stats (not pixel alignment).

Side cameras face different directions, so comparing the same pixel coordinates
across cameras is meaningless. We match per-channel mean/std (or cumulative
histogram) of each camera's image to the front camera's global statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def fit_affine_rgb(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel scale+bias from **global** stats (not spatially paired pixels).

    ``target ≈ scale * source + bias`` using masked mean/std of each image
    independently — safe across cameras that do not share a viewpoint.
    """
    assert source.shape[-1] == 3 and target.shape[-1] == 3
    if mask is None:
        h = source.shape[0]
        mask = np.zeros(source.shape[:2], dtype=bool)
        mask[h // 2 :, :] = True
        mid = (source.mean(axis=2) > 20) & (source.mean(axis=2) < 240)
        mask &= mid
    # Target mask: same geometric heuristic on its own image (not source coords)
    th = target.shape[0]
    tmask = np.zeros(target.shape[:2], dtype=bool)
    tmask[th // 2 :, :] = True
    tmid = (target.mean(axis=2) > 20) & (target.mean(axis=2) < 240)
    tmask &= tmid

    scales = np.ones(3, dtype=np.float64)
    biases = np.zeros(3, dtype=np.float64)
    for c in range(3):
        xs = source[..., c][mask].astype(np.float64)
        ys = target[..., c][tmask].astype(np.float64)
        if xs.size < 64 or ys.size < 64:
            continue
        xs_mean, xs_std = float(xs.mean()), float(xs.std()) + 1e-6
        ys_mean, ys_std = float(ys.mean()), float(ys.std()) + 1e-6
        scales[c] = ys_std / xs_std
        biases[c] = ys_mean - scales[c] * xs_mean
    return scales, biases


def match_histogram_channel(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """CDF histogram match one channel (uint8) to target's distribution."""
    src = source.astype(np.uint8).ravel()
    tgt = target.astype(np.uint8).ravel()
    s_counts = np.bincount(src, minlength=256).astype(np.float64)
    t_counts = np.bincount(tgt, minlength=256).astype(np.float64)
    s_cdf = np.cumsum(s_counts)
    t_cdf = np.cumsum(t_counts)
    s_cdf /= max(s_cdf[-1], 1.0)
    t_cdf /= max(t_cdf[-1], 1.0)
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and t_cdf[j] < s_cdf[i]:
            j += 1
        lut[i] = j
    return lut[source.astype(np.uint8)]


def apply_affine(image: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32) * scale.reshape(1, 1, 3) + bias.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def match_cameras_to_front(
    frame_paths: dict[str, Path],
    out_dir: Path,
    *,
    masks: dict[str, Path] | None = None,
    method: str = "stats",
) -> dict[str, dict[str, list[float]]]:
    """Match each camera to front using global stats / histograms.

    Writes color-matched **copies** under ``out_dir`` (never overwrites originals).
    Optional per-camera mask paths restrict which pixels contribute to stats
    (run after ego/sky/mover masks when available).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if "front" not in frame_paths:
        result = {}
        for cam, p in frame_paths.items():
            dest = out_dir / p.name
            Image.open(p).save(dest)
            result[cam] = {"scale": [1, 1, 1], "bias": [0, 0, 0], "path": str(dest)}
        return result

    front = np.asarray(Image.open(frame_paths["front"]).convert("RGB"))
    front_mask = None
    if masks and "front" in masks and Path(masks["front"]).exists():
        front_mask = np.asarray(Image.open(masks["front"]).convert("L")) > 127

    affines: dict[str, dict[str, list[float]]] = {}
    for cam, p in frame_paths.items():
        img = np.asarray(Image.open(p).convert("RGB"))
        cam_mask = None
        if masks and cam in masks and Path(masks[cam]).exists():
            cam_mask = np.asarray(Image.open(masks[cam]).convert("L")) > 127
        if cam == "front":
            scale, bias = np.ones(3), np.zeros(3)
            matched = img
        elif method == "histogram":
            # Resize front only to gather a comparable histogram length, not spatial align
            front_r = front
            if img.shape != front.shape:
                front_r = np.asarray(Image.fromarray(front).resize((img.shape[1], img.shape[0])))
            matched = np.stack(
                [match_histogram_channel(img[..., c], front_r[..., c]) for c in range(3)],
                axis=-1,
            )
            scale, bias = fit_affine_rgb(img, front_r, mask=cam_mask)
        else:
            front_r = front
            if img.shape != front.shape:
                # Size match for stats sampling only — not pixel correspondence
                front_r = np.asarray(Image.fromarray(front).resize((img.shape[1], img.shape[0])))
            # Prefer intersection of masks when both exist; else per-image heuristic
            mask = cam_mask
            if mask is not None and front_mask is not None and mask.shape == front_mask.shape:
                pass  # use cam_mask for source; fit_affine uses its own target mask
            scale, bias = fit_affine_rgb(img, front_r, mask=mask)
            matched = apply_affine(img, scale, bias)
        dest = out_dir / p.name
        Image.fromarray(matched).save(dest, quality=92)
        affines[cam] = {
            "scale": [float(x) for x in scale],
            "bias": [float(x) for x in bias],
            "path": str(dest),
        }
    return affines
