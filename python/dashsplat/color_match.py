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
    target_mask: np.ndarray | None = None,
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
    if target_mask is None:
        th = target.shape[0]
        target_mask = np.zeros(target.shape[:2], dtype=bool)
        target_mask[th // 2 :, :] = True
        tmid = (target.mean(axis=2) > 20) & (target.mean(axis=2) < 240)
        target_mask &= tmid

    scales = np.ones(3, dtype=np.float64)
    biases = np.zeros(3, dtype=np.float64)
    for c in range(3):
        xs = source[..., c][mask].astype(np.float64)
        ys = target[..., c][target_mask].astype(np.float64)
        if xs.size < 64 or ys.size < 64:
            continue
        xs_mean, xs_std = float(xs.mean()), float(xs.std()) + 1e-6
        ys_mean, ys_std = float(ys.mean()), float(ys.std()) + 1e-6
        # Clamp scale so a dark/bright outlier frame cannot explode the match
        scales[c] = float(np.clip(ys_std / xs_std, 0.25, 4.0))
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


def _as_path_list(value: Path | list[Path] | str | list[str]) -> list[Path]:
    if isinstance(value, (list, tuple)):
        return [Path(p) for p in value]
    return [Path(value)]


def _channel_stats(
    paths: list[Path],
    mask_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std per channel aggregated across all frames (optionally masked)."""
    sums = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    count = 0
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
        if mask_path is not None and mask_path.exists():
            m = np.asarray(Image.open(mask_path).convert("L")) > 127
            if m.shape == img.shape[:2]:
                pix = img[m]
            else:
                pix = img.reshape(-1, 3)
        else:
            # Lower-half + mid-luminance heuristic
            h = img.shape[0]
            m = np.zeros(img.shape[:2], dtype=bool)
            m[h // 2 :, :] = True
            mid = (img.mean(axis=2) > 20) & (img.mean(axis=2) < 240)
            m &= mid
            pix = img[m] if np.any(m) else img.reshape(-1, 3)
        if pix.size == 0:
            continue
        sums += pix.sum(axis=0)
        sq += (pix * pix).sum(axis=0)
        count += pix.shape[0]
    if count < 64:
        return np.array([128.0, 128.0, 128.0]), np.ones(3)
    mean = sums / count
    var = np.maximum(sq / count - mean * mean, 1e-6)
    return mean, np.sqrt(var)


def match_cameras_to_front(
    frame_paths: dict[str, Path | list[Path]],
    out_dir: Path,
    *,
    masks: dict[str, Path] | None = None,
    method: str = "stats",
) -> dict[str, dict[str, list[float]]]:
    """Match each camera to front using global stats across all listed frames.

    ``frame_paths`` values may be a single path or a list — stats aggregate over
    every frame (not just the first). Front mask is applied to front stats when
    provided (no longer a no-op). Scale is clamped to ``[0.25, 4]``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, list[Path]] = {k: _as_path_list(v) for k, v in frame_paths.items()}
    if "front" not in normalized:
        result = {}
        for cam, paths in normalized.items():
            p = paths[0]
            dest = out_dir / p.name
            Image.open(p).save(dest)
            result[cam] = {"scale": [1, 1, 1], "bias": [0, 0, 0], "path": str(dest)}
        return result

    front_paths = normalized["front"]
    front_mask_path = Path(masks["front"]) if masks and "front" in masks else None
    front_mean, front_std = _channel_stats(front_paths, front_mask_path)
    front_img = np.asarray(Image.open(front_paths[0]).convert("RGB"))

    affines: dict[str, dict[str, list[float]]] = {}
    for cam, paths in normalized.items():
        img = np.asarray(Image.open(paths[0]).convert("RGB"))
        cam_mask_path = Path(masks[cam]) if masks and cam in masks else None
        if cam == "front":
            scale, bias = np.ones(3), np.zeros(3)
            matched = img
        elif method == "histogram":
            front_r = front_img
            if img.shape != front_img.shape:
                front_r = np.asarray(
                    Image.fromarray(front_img).resize((img.shape[1], img.shape[0]))
                )
            matched = np.stack(
                [match_histogram_channel(img[..., c], front_r[..., c]) for c in range(3)],
                axis=-1,
            )
            cam_mask = None
            front_mask = None
            if cam_mask_path and cam_mask_path.exists():
                cam_mask = np.asarray(Image.open(cam_mask_path).convert("L")) > 127
            if front_mask_path and front_mask_path.exists():
                front_mask = np.asarray(Image.open(front_mask_path).convert("L")) > 127
                if front_mask.shape != front_r.shape[:2]:
                    front_mask = None
            scale, bias = fit_affine_rgb(
                img, front_r, mask=cam_mask, target_mask=front_mask
            )
        else:
            src_mean, src_std = _channel_stats(paths, cam_mask_path)
            scale = np.clip(front_std / np.maximum(src_std, 1e-6), 0.25, 4.0)
            bias = front_mean - scale * src_mean
            matched = apply_affine(img, scale, bias)
        dest = out_dir / paths[0].name
        Image.fromarray(matched).save(dest, quality=92)
        affines[cam] = {
            "scale": [float(x) for x in scale],
            "bias": [float(x) for x in bias],
            "path": str(dest),
            "n_frames": len(paths),
        }
    return affines
