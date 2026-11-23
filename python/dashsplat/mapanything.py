"""MapAnything (MLX) multi-view pointmaps.

Uses mlx-community/map-anything-apache-fp16 via mlx-vlm when available.
Falls back to a telemetry-seeded synthetic pointmap for CI / first-boot smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

MAPANYTHING_REPO = "mlx-community/map-anything-apache-fp16"


ProgressCb = Callable[[str, float], None]


def _progress(cb: ProgressCb | None, message: str, frac: float) -> None:
    if cb:
        cb(message, frac)


def run_mapanything_mlx(
    image_paths: list[Path],
    intrinsics: list[list[list[float]]] | None = None,
    camera_poses: list[list[list[float]]] | None = None,
    model_id: str = MAPANYTHING_REPO,
    cache_dir: Path | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Run MapAnything and return stacked pointmaps + poses.

    Returns dict with keys: pts3d (N,H,W,3), colors (N,H,W,3), conf (N,H,W),
    camera_poses, intrinsics, backend.
    """
    _progress(progress, "Loading MapAnything (MLX)…", 0.05)
    try:
        import mlx.core as mx  # type: ignore
        from mlx_vlm import load  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "mlx / mlx-vlm not installed. Install optional deps: pip install 'dashsplat[mlx]'"
        ) from e

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["path"] = str(cache_dir)
    model, processor = load(model_id, **kwargs)

    views_spec: list[dict[str, Any]] = []
    from PIL import Image

    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        entry: dict[str, Any] = {"img": img}
        if intrinsics and i < len(intrinsics):
            entry["intrinsics"] = np.asarray(intrinsics[i], dtype=np.float32)
        if camera_poses and i < len(camera_poses):
            entry["camera_poses"] = np.asarray(camera_poses[i], dtype=np.float32)
        views_spec.append(entry)

    _progress(progress, "Running MapAnything infer…", 0.3)
    views = processor.preprocess(views_spec)
    predictions = model.infer(views)
    mx.eval(predictions)

    pts_list = []
    col_list = []
    conf_list = []
    pose_list = []
    K_list = []
    for i, pred in enumerate(predictions):
        pts = np.array(pred["pts3d"]).reshape(-1, 3) if False else np.array(pred["pts3d"])
        # pred["pts3d"] is (1,H,W,3)
        pts3d = np.array(pred["pts3d"])
        if pts3d.ndim == 4:
            pts3d = pts3d[0]
        conf = np.array(pred["conf"])
        if conf.ndim == 4:
            conf = conf[0]
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        img = Image.open(image_paths[i]).convert("RGB")
        img_np = np.asarray(img).astype(np.float32) / 255.0
        # Resize colors to pointmap resolution if needed
        h, w = pts3d.shape[:2]
        if img_np.shape[0] != h or img_np.shape[1] != w:
            img = img.resize((w, h))
            img_np = np.asarray(img).astype(np.float32) / 255.0
        pts_list.append(pts3d)
        col_list.append(img_np)
        conf_list.append(conf)
        pose_list.append(np.array(pred["camera_poses"])[0] if np.array(pred["camera_poses"]).ndim == 3 else np.array(pred["camera_poses"]))
        K_list.append(np.array(pred["intrinsics"])[0] if np.array(pred["intrinsics"]).ndim == 3 else np.array(pred["intrinsics"]))

    _progress(progress, "MapAnything done", 0.9)
    return {
        "pts3d": np.stack(pts_list, axis=0),
        "colors": np.stack(col_list, axis=0),
        "conf": np.stack(conf_list, axis=0),
        "camera_poses": pose_list,
        "intrinsics": K_list,
        "backend": "mlx-mapanything",
        "model_id": model_id,
    }


def synthetic_pointmap_from_poses(
    image_paths: list[Path],
    camera_poses: list[list[list[float]]],
    intrinsics: list[list[list[float]]],
    *,
    stride: int = 8,
    depth_m: float = 12.0,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """CI / no-MLX fallback: unproject a sparse depth plane per camera using known poses."""
    from PIL import Image

    _progress(progress, "Building synthetic pointmaps (no MLX)…", 0.2)
    pts_list = []
    col_list = []
    conf_list = []
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        K = np.asarray(intrinsics[i], dtype=np.float64)
        c2w = np.asarray(camera_poses[i], dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        grid_x, grid_y = np.meshgrid(xs, ys)
        z = depth_m * (0.7 + 0.3 * (grid_y / max(h - 1, 1)))
        x = (grid_x - cx) * z / fx
        y = (grid_y - cy) * z / fy
        cam_pts = np.stack([x, y, z], axis=-1)  # OpenCV cam
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        world = cam_pts @ R.T + t
        colors = arr[grid_y, grid_x]
        # Dense-ish HxW maps for downstream (fill sparse then interpolate-ish by scatter)
        pts_full = np.zeros((h, w, 3), dtype=np.float32)
        col_full = np.zeros((h, w, 3), dtype=np.float32)
        conf = np.zeros((h, w), dtype=np.float32)
        pts_full[grid_y, grid_x] = world.astype(np.float32)
        col_full[grid_y, grid_x] = colors
        conf[grid_y, grid_x] = 1.0
        pts_list.append(pts_full)
        col_list.append(col_full)
        conf_list.append(conf)

    _progress(progress, "Synthetic pointmaps ready", 0.9)
    return {
        "pts3d": np.stack(pts_list, axis=0),
        "colors": np.stack(col_list, axis=0),
        "conf": np.stack(conf_list, axis=0),
        "camera_poses": camera_poses,
        "intrinsics": intrinsics,
        "backend": "synthetic-fallback",
        "model_id": None,
    }


def pointmap_to_cloud(
    result: dict[str, Any],
    *,
    conf_thresh: float = 0.1,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    pts = result["pts3d"]
    cols = result["colors"]
    conf = result["conf"]
    mask = conf > conf_thresh
    # Also drop zero-filled synthetic cells
    mag = np.linalg.norm(pts, axis=-1)
    mask &= mag > 1e-6
    means = pts[mask]
    colors = cols[mask]
    if means.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(means.shape[0], max_points, replace=False)
        means = means[idx]
        colors = colors[idx]
    return means.astype(np.float32), colors.astype(np.float32)


def save_pointmap_npz(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pts3d=result["pts3d"],
        colors=result["colors"],
        conf=result["conf"],
        backend=np.array(result.get("backend", "")),
    )
    meta = {
        "backend": result.get("backend"),
        "model_id": result.get("model_id"),
        "n_views": int(result["pts3d"].shape[0]),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
