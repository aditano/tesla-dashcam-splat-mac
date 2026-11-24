"""MapAnything multi-view pointmaps via official facebookresearch/map-anything (PyTorch).

Pinned package: mapanything 1.1.4 @ commit 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
HF weights: facebook/map-anything-apache (default) or facebook/map-anything (NC opt-in).

mlx-vlm has no MapAnything model (PR #2400 never merged) — do not use mlx-vlm here.
Falls back to a loud synthetic flat-plane pointmap when PyTorch MapAnything is unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Official package pin (pyproject.toml version 1.1.4 on this commit)
MAPANYTHING_GIT_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
MAPANYTHING_PKG_VERSION = "1.1.4"
MAPANYTHING_GIT_URL = (
    f"git+https://github.com/facebookresearch/map-anything.git@{MAPANYTHING_GIT_COMMIT}"
)

# Hugging Face Hub model IDs (Apache default; NC opt-in)
MAPANYTHING_HF_APACHE = "facebook/map-anything-apache"
MAPANYTHING_HF_NC = "facebook/map-anything"

# Back-compat aliases used by older call sites / docs
MAPANYTHING_REPO = MAPANYTHING_HF_APACHE
MAPANYTHING_NC_REPO = MAPANYTHING_HF_NC

ProgressCb = Callable[[str, float], None]


def _progress(cb: ProgressCb | None, message: str, frac: float) -> None:
    if cb:
        cb(message, frac)


def mapanything_torch_device() -> str:
    """Prefer Apple MPS, then CUDA, else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_pretrained_id(model_id: str, cache_dir: Path | None) -> str:
    """Use a local HF snapshot directory when present; else Hub id."""
    if cache_dir is not None:
        p = Path(cache_dir)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        # Also accept a hub cache layout under cache_dir
        if p.is_dir():
            for cand in p.rglob("config.json"):
                return str(cand.parent)
    return model_id


def load_mapanything_torch(
    model_id: str = MAPANYTHING_HF_APACHE,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> tuple[Any, str]:
    """Load official ``MapAnything.from_pretrained`` onto MPS/CUDA/CPU.

    Returns (model, device_str).
    """
    import torch
    from mapanything.models import MapAnything  # type: ignore

    dev = device or mapanything_torch_device()
    pretrained = _resolve_pretrained_id(model_id, cache_dir)
    model = MapAnything.from_pretrained(pretrained).to(dev)
    model.eval()
    return model, dev


def predictions_to_pointmap(
    predictions: list[dict[str, Any]],
    image_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Adapt MapAnything ``model.infer`` outputs to our stacked pointmap dict.

    Official output per view (see facebookresearch/map-anything README):
      pts3d (B,H,W,3), conf (B,H,W), camera_poses (B,4,4), intrinsics (B,3,3),
      img_no_norm (B,H,W,3) optional.
    """
    from PIL import Image

    pts_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    pose_list: list[np.ndarray] = []
    K_list: list[np.ndarray] = []

    for i, pred in enumerate(predictions):
        pts3d = np.asarray(pred["pts3d"])
        if hasattr(pts3d, "detach"):
            pts3d = pts3d.detach().cpu().numpy()
        pts3d = np.asarray(pts3d, dtype=np.float32)
        if pts3d.ndim == 4:
            pts3d = pts3d[0]

        conf = pred.get("conf")
        if conf is None:
            conf = np.ones(pts3d.shape[:2], dtype=np.float32)
        else:
            conf = np.asarray(conf)
            if hasattr(conf, "detach"):
                conf = conf.detach().cpu().numpy()
            conf = np.asarray(conf, dtype=np.float32)
            if conf.ndim == 4:
                conf = conf[0]
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = conf[..., 0]

        # Colors: prefer denormalized img from model; else reload source image
        colors = None
        if "img_no_norm" in pred and pred["img_no_norm"] is not None:
            img_nn = pred["img_no_norm"]
            if hasattr(img_nn, "detach"):
                img_nn = img_nn.detach().cpu().numpy()
            img_nn = np.asarray(img_nn, dtype=np.float32)
            if img_nn.ndim == 4:
                img_nn = img_nn[0]
            # May be [0,1] or [0,255]
            if img_nn.max() > 1.5:
                img_nn = img_nn / 255.0
            colors = np.clip(img_nn[..., :3], 0, 1)
        elif image_paths is not None and i < len(image_paths):
            img = Image.open(image_paths[i]).convert("RGB")
            img_np = np.asarray(img).astype(np.float32) / 255.0
            h, w = pts3d.shape[:2]
            if img_np.shape[0] != h or img_np.shape[1] != w:
                img = img.resize((w, h))
                img_np = np.asarray(img).astype(np.float32) / 255.0
            colors = img_np
        else:
            colors = np.full((*pts3d.shape[:2], 3), 0.5, dtype=np.float32)

        h, w = pts3d.shape[:2]
        if h * w > 320 * 240:
            step = max(1, int(np.ceil(np.sqrt((h * w) / (320 * 240)))))
            pts3d = pts3d[::step, ::step]
            conf = conf[::step, ::step]
            colors = colors[::step, ::step]

        pose = pred.get("camera_poses")
        if pose is not None:
            pose = np.asarray(pose)
            if hasattr(pose, "detach"):
                pose = pose.detach().cpu().numpy()
            pose = np.asarray(pose, dtype=np.float32)
            if pose.ndim == 3:
                pose = pose[0]
        else:
            pose = np.eye(4, dtype=np.float32)

        K = pred.get("intrinsics")
        if K is not None:
            K = np.asarray(K)
            if hasattr(K, "detach"):
                K = K.detach().cpu().numpy()
            K = np.asarray(K, dtype=np.float32)
            if K.ndim == 3:
                K = K[0]
        else:
            K = np.eye(3, dtype=np.float32)

        pts_list.append(pts3d.astype(np.float32))
        col_list.append(colors.astype(np.float32))
        conf_list.append(conf.astype(np.float32))
        pose_list.append(pose)
        K_list.append(K)

    return {
        "pts3d": np.stack(pts_list, axis=0),
        "colors": np.stack(col_list, axis=0),
        "conf": np.stack(conf_list, axis=0),
        "camera_poses": pose_list,
        "intrinsics": K_list,
    }


def _infer_chunk_torch(
    model: Any,
    device: str,
    image_paths: list[Path],
    intrinsics: list[list[list[float]]] | None,
    camera_poses: list[list[list[float]]] | None,
) -> dict[str, Any]:
    """Run one MapAnything window using the official API."""
    import torch
    from mapanything.utils.image import load_images, preprocess_inputs  # type: ignore

    # Prefer load_images for image-only; use preprocess_inputs when K/poses given
    if intrinsics or camera_poses:
        views_spec: list[dict[str, Any]] = []
        for i, p in enumerate(image_paths):
            from PIL import Image

            img = np.asarray(Image.open(p).convert("RGB"))  # H,W,3 uint8 [0,255]
            entry: dict[str, Any] = {"img": img}
            if intrinsics and i < len(intrinsics):
                entry["intrinsics"] = np.asarray(intrinsics[i], dtype=np.float32)
            if camera_poses and i < len(camera_poses):
                entry["camera_poses"] = np.asarray(camera_poses[i], dtype=np.float32)
            views_spec.append(entry)
        views = preprocess_inputs(views_spec)
    else:
        views = load_images([str(p) for p in image_paths])

    # AMP: bf16 preferred on CUDA; on MPS/CPU skip or use fp16 carefully
    use_amp = device == "cuda"
    amp_dtype = "bf16"
    with torch.inference_mode():
        predictions = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    return predictions_to_pointmap(predictions, image_paths)


def run_mapanything_torch(
    image_paths: list[Path],
    intrinsics: list[list[list[float]]] | None = None,
    camera_poses: list[list[list[float]]] | None = None,
    model_id: str = MAPANYTHING_HF_APACHE,
    cache_dir: Path | None = None,
    progress: ProgressCb | None = None,
    model: Any | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run official MapAnything and return stacked pointmaps + poses."""
    _progress(progress, "Loading MapAnything (PyTorch)…", 0.05)
    try:
        if model is None:
            model, dev = load_mapanything_torch(model_id, cache_dir, device=device)
        else:
            dev = device or mapanything_torch_device()
    except ImportError as e:
        raise RuntimeError(
            "mapanything / torch not installed. "
            f"Install: pip install torch && pip install '{MAPANYTHING_GIT_URL}' "
            "or pip install 'dashsplat[mapanything]'"
        ) from e

    _progress(progress, f"Running MapAnything infer ({dev})…", 0.3)
    chunk = _infer_chunk_torch(model, dev, image_paths, intrinsics, camera_poses)
    _progress(progress, "MapAnything done", 0.9)
    return {
        **chunk,
        "backend": f"torch-mapanything-{dev}",
        "model_id": model_id,
        "pkg_version": MAPANYTHING_PKG_VERSION,
        "pkg_commit": MAPANYTHING_GIT_COMMIT,
        "device": dev,
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
    """Run MapAnything over overlapping windows (model loaded once)."""
    mid = model_id or (MAPANYTHING_HF_NC if use_nc_weights else MAPANYTHING_HF_APACHE)
    n = len(image_paths)
    if n == 0:
        raise ValueError("no images")

    _progress(progress, "Loading MapAnything once (PyTorch)…", 0.05)
    try:
        model, dev = load_mapanything_torch(mid, cache_dir)
    except ImportError as e:
        raise RuntimeError(
            "mapanything / torch not installed. "
            f"Install: pip install torch && pip install '{MAPANYTHING_GIT_URL}'"
        ) from e

    if n <= window:
        chunk = _infer_chunk_torch(model, dev, image_paths, intrinsics, camera_poses)
        return {
            **chunk,
            "backend": f"torch-mapanything-{dev}",
            "model_id": mid,
            "use_nc_weights": use_nc_weights,
            "pkg_version": MAPANYTHING_PKG_VERSION,
            "pkg_commit": MAPANYTHING_GIT_COMMIT,
            "device": dev,
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
        _progress(
            progress,
            f"MapAnything window {wi + 1}/{len(starts)}…",
            0.3 + 0.4 * wi / len(starts),
        )
        chunk = _infer_chunk_torch(
            model, dev, image_paths[sl], intrinsics[sl], camera_poses[sl]
        )
        if all_pts is None:
            h, w = chunk["pts3d"].shape[1:3]
            all_pts = np.zeros((n, h, w, 3), dtype=np.float32)
            all_col = np.zeros((n, h, w, 3), dtype=np.float32)
            all_conf = np.zeros((n, h, w), dtype=np.float32)
        for local_i, global_i in enumerate(range(start, end)):
            pts = chunk["pts3d"][local_i]
            conf = chunk["conf"][local_i]
            col = chunk["colors"][local_i]
            th, tw = all_pts.shape[1:3]
            if pts.shape[0] != th or pts.shape[1] != tw:
                ys = np.linspace(0, pts.shape[0] - 1, th).astype(np.int32)
                xs = np.linspace(0, pts.shape[1] - 1, tw).astype(np.int32)
                pts = pts[ys][:, xs]
                conf = conf[ys][:, xs]
                col = col[ys][:, xs]
            better = conf > all_conf[global_i]
            all_pts[global_i][better] = pts[better]
            all_col[global_i][better] = col[better]
            all_conf[global_i] = np.maximum(all_conf[global_i], conf)
        del chunk

    return {
        "pts3d": all_pts,
        "colors": all_col,
        "conf": all_conf,
        "camera_poses": camera_poses,
        "intrinsics": intrinsics,
        "backend": f"torch-mapanything-windows-{dev}",
        "model_id": mid,
        "use_nc_weights": use_nc_weights,
        "pkg_version": MAPANYTHING_PKG_VERSION,
        "pkg_commit": MAPANYTHING_GIT_COMMIT,
        "device": dev,
    }


# Keep name for older imports; mlx path removed (no real mlx MapAnything).
run_mapanything_mlx = run_mapanything_torch


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
    """Loud fallback when MapAnything weights/package are unavailable."""
    from PIL import Image

    _progress(
        progress,
        "WARNING: synthetic flat-plane geometry (MapAnything unavailable)…",
        0.2,
    )
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
        pts_list.append(world.astype(np.float32))
        col_list.append(colors.astype(np.float32))
        conf_list.append(np.ones(world.shape[:2], dtype=np.float32))

    _progress(progress, "Synthetic flat-plane pointmaps ready (LOW QUALITY)", 0.9)
    return {
        "pts3d": np.stack(pts_list, axis=0),
        "colors": np.stack(col_list, axis=0),
        "conf": np.stack(conf_list, axis=0),
        "camera_poses": camera_poses,
        "intrinsics": intrinsics,
        "backend": "synthetic-fallback",
        "model_id": None,
        "warning": (
            "MapAnything unavailable — using synthetic flat-plane geometry fallback. "
            "Splat quality will be low until facebook/map-anything-apache weights load."
        ),
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
        "pkg_version": result.get("pkg_version"),
        "pkg_commit": result.get("pkg_commit"),
        "n_views": int(result["pts3d"].shape[0]),
        "warning": result.get("warning"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def smoke_infer_tiny(
    image_paths: list[Path],
    *,
    cache_dir: Path | None = None,
    model_id: str = MAPANYTHING_HF_APACHE,
) -> dict[str, Any]:
    """Minimal real inference for local smoke tests (requires weights)."""
    return run_mapanything_torch(
        image_paths,
        model_id=model_id,
        cache_dir=cache_dir,
    )
