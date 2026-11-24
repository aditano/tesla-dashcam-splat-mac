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
MAPANYTHING_NC_REPO = "mlx-community/map-anything-fp16"  # CC BY-NC; opt-in only


ProgressCb = Callable[[str, float], None]


def _progress(cb: ProgressCb | None, message: str, frac: float) -> None:
    if cb:
        cb(message, frac)


def _load_mapanything(model_id: str, cache_dir: Path | None = None):
    import mlx.core as mx  # type: ignore  # noqa: F401
    from mlx_vlm import load  # type: ignore

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["path"] = str(cache_dir)
    return load(model_id, **kwargs)


def _infer_chunk(
    model: Any,
    processor: Any,
    image_paths: list[Path],
    intrinsics: list[list[list[float]]] | None,
    camera_poses: list[list[list[float]]] | None,
) -> dict[str, Any]:
    """Run one MapAnything window; returns compact arrays (may be downsampled)."""
    import mlx.core as mx  # type: ignore
    from PIL import Image

    views_spec: list[dict[str, Any]] = []
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        entry: dict[str, Any] = {"img": img}
        if intrinsics and i < len(intrinsics):
            entry["intrinsics"] = np.asarray(intrinsics[i], dtype=np.float32)
        if camera_poses and i < len(camera_poses):
            entry["camera_poses"] = np.asarray(camera_poses[i], dtype=np.float32)
        views_spec.append(entry)

    views = processor.preprocess(views_spec)
    predictions = model.infer(views)
    mx.eval(predictions)

    pts_list = []
    col_list = []
    conf_list = []
    pose_list = []
    K_list = []
    for i, pred in enumerate(predictions):
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
        h, w = pts3d.shape[:2]
        if img_np.shape[0] != h or img_np.shape[1] != w:
            img = img.resize((w, h))
            img_np = np.asarray(img).astype(np.float32) / 255.0
        # Downsample full-res depth maps immediately to limit peak memory
        if h * w > 320 * 240:
            step = max(1, int(np.ceil(np.sqrt((h * w) / (320 * 240)))))
            pts3d = pts3d[::step, ::step]
            conf = conf[::step, ::step]
            img_np = img_np[::step, ::step]
        pts_list.append(pts3d.astype(np.float32))
        col_list.append(img_np.astype(np.float32))
        conf_list.append(conf.astype(np.float32))
        pose_list.append(
            np.array(pred["camera_poses"])[0]
            if np.array(pred["camera_poses"]).ndim == 3
            else np.array(pred["camera_poses"])
        )
        K_list.append(
            np.array(pred["intrinsics"])[0]
            if np.array(pred["intrinsics"]).ndim == 3
            else np.array(pred["intrinsics"])
        )

    return {
        "pts3d": np.stack(pts_list, axis=0),
        "colors": np.stack(col_list, axis=0),
        "conf": np.stack(conf_list, axis=0),
        "camera_poses": pose_list,
        "intrinsics": K_list,
    }


def run_mapanything_mlx(
    image_paths: list[Path],
    intrinsics: list[list[list[float]]] | None = None,
    camera_poses: list[list[list[float]]] | None = None,
    model_id: str = MAPANYTHING_REPO,
    cache_dir: Path | None = None,
    progress: ProgressCb | None = None,
    model_and_processor: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Run MapAnything and return stacked pointmaps + poses.

    Returns dict with keys: pts3d (N,H,W,3), colors (N,H,W,3), conf (N,H,W),
    camera_poses, intrinsics, backend.
    """
    _progress(progress, "Loading MapAnything (MLX)…", 0.05)
    try:
        if model_and_processor is None:
            model, processor = _load_mapanything(model_id, cache_dir)
        else:
            model, processor = model_and_processor
    except ImportError as e:
        raise RuntimeError(
            "mlx / mlx-vlm not installed. Install optional deps: pip install 'dashsplat[mlx]'"
        ) from e

    _progress(progress, "Running MapAnything infer…", 0.3)
    chunk = _infer_chunk(model, processor, image_paths, intrinsics, camera_poses)
    _progress(progress, "MapAnything done", 0.9)
    return {
        **chunk,
        "backend": "mlx-mapanything",
        "model_id": model_id,
    }


def run_mapanything_windows(
    image_paths: list[Path],
    intrinsics: list[list[list[float]]],
    camera_poses: list[list[list[float]]],
    *,
    window: int = 16,
    stride: int = 8,
    use_nc_weights: bool = False,
    model_id: str | None = None,
    cache_dir: Path | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Run MapAnything over overlapping windows that mix all cameras/times.

    Loads the model **once** outside the window loop. Accumulates best-confidence
    points without retaining every full-res depth map for all views at once —
    windows are downsampled and merged into fixed-size buffers.
    """
    mid = model_id or (MAPANYTHING_NC_REPO if use_nc_weights else MAPANYTHING_REPO)
    n = len(image_paths)
    if n == 0:
        raise ValueError("no images")

    _progress(progress, "Loading MapAnything once…", 0.05)
    try:
        model, processor = _load_mapanything(mid, cache_dir)
    except ImportError as e:
        raise RuntimeError(
            "mlx / mlx-vlm not installed. Install optional deps: pip install 'dashsplat[mlx]'"
        ) from e

    if n <= window:
        chunk = _infer_chunk(model, processor, image_paths, intrinsics, camera_poses)
        return {
            **chunk,
            "backend": "mlx-mapanything",
            "model_id": mid,
            "use_nc_weights": use_nc_weights,
        }

    all_pts = None
    all_col = None
    all_conf = None
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != n - window:
        starts.append(max(0, n - window))
    for wi, start in enumerate(starts):
        end = min(n, start + window)
        sl = slice(start, end)
        _progress(progress, f"MapAnything window {wi+1}/{len(starts)}…", 0.3 + 0.4 * wi / len(starts))
        chunk = _infer_chunk(
            model,
            processor,
            image_paths[sl],
            intrinsics[sl],
            camera_poses[sl],
        )
        if all_pts is None:
            h, w = chunk["pts3d"].shape[1:3]
            all_pts = np.zeros((n, h, w, 3), dtype=np.float32)
            all_col = np.zeros((n, h, w, 3), dtype=np.float32)
            all_conf = np.zeros((n, h, w), dtype=np.float32)
        # Resize window maps if shape drifted
        for local_i, global_i in enumerate(range(start, end)):
            pts = chunk["pts3d"][local_i]
            conf = chunk["conf"][local_i]
            col = chunk["colors"][local_i]
            th, tw = all_pts.shape[1:3]
            if pts.shape[0] != th or pts.shape[1] != tw:
                # nearest subsample / pad to buffer size
                ys = np.linspace(0, pts.shape[0] - 1, th).astype(np.int32)
                xs = np.linspace(0, pts.shape[1] - 1, tw).astype(np.int32)
                pts = pts[ys][:, xs]
                conf = conf[ys][:, xs]
                col = col[ys][:, xs]
            better = conf > all_conf[global_i]
            all_pts[global_i][better] = pts[better]
            all_col[global_i][better] = col[better]
            all_conf[global_i] = np.maximum(all_conf[global_i], conf)
        # Drop chunk references promptly
        del chunk

    return {
        "pts3d": all_pts,
        "colors": all_col,
        "conf": all_conf,
        "camera_poses": camera_poses,
        "intrinsics": intrinsics,
        "backend": "mlx-mapanything-windows",
        "model_id": mid,
        "use_nc_weights": use_nc_weights,
    }


def voxel_downsample(
    means: np.ndarray,
    colors: np.ndarray,
    voxel_m: float = 0.08,
    max_points: int = 400_000,
) -> tuple[np.ndarray, np.ndarray]:
    if means.shape[0] == 0:
        return means, colors
    keys = np.floor(means / max(voxel_m, 1e-4)).astype(np.int32)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    means_d = means[unique_idx]
    colors_d = colors[unique_idx]
    if means_d.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pick = rng.choice(means_d.shape[0], max_points, replace=False)
        means_d = means_d[pick]
        colors_d = colors_d[pick]
    return means_d.astype(np.float32), colors_d.astype(np.float32)


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
        cam_pts = np.stack([x, y, z], axis=-1)
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        world = cam_pts @ R.T + t
        colors = arr[grid_y, grid_x]
        # Store sparse maps only (not dense HxW) — pack into compact grids
        pts_list.append(world.astype(np.float32))
        col_list.append(colors.astype(np.float32))
        conf_list.append(np.ones(world.shape[:2], dtype=np.float32))

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
