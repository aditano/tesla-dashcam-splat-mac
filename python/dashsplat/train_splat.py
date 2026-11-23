"""3DGS training: OpenSplat (preferred on Mac Metal) with seeded Python fallback.

Why OpenSplat
-------------
OpenSplat (WebODM/OpenSplat) is a lean C++ 3DGS trainer with a first-class MPS/Metal
backend on Apple Silicon. It reads COLMAP/nerfstudio projects, seeds from sparse
points, and exports .ply / .spz / cameras JSON — matching our deliverables.

mpsplat / gsplat-mps are Python ports of the same Metal kernels but are less
maintained and AGPL-tainted via OpenSplat's Metal code. MetalSplat (MIT, pure
Python + runtime Metal shaders) is a strong alternative when you want no native
binary; we keep a seeded fallback that does not require it for smoke tests.

Selection rule used by this module:
1. If `opensplat` is on PATH (or DASHSPLAT_OPENSPLAT is set) → use OpenSplat.
2. Else → seed Gaussians from MapAnything points and write .ply (+ stub .spz).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dashsplat.export import write_cameras_json, write_gaussian_ply, write_spz_stub

ProgressCb = Callable[[str, float], None]


def find_opensplat() -> Path | None:
    env = os.environ.get("DASHSPLAT_OPENSPLAT")
    if env and Path(env).exists():
        return Path(env)
    which = shutil.which("opensplat")
    return Path(which) if which else None


def write_colmap_sparse(
    out_dir: Path,
    means: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict[str, Any]],
) -> Path:
    """Write a minimal COLMAP text model OpenSplat can ingest."""
    sparse = out_dir / "sparse" / "0"
    images = out_dir / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    # cameras.txt
    cam_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for i, cam in enumerate(cameras, start=1):
        w = int(cam.get("w", 1280))
        h = int(cam.get("h", 960))
        fx = float(cam.get("fl_x", w * 0.9))
        fy = float(cam.get("fl_y", fx))
        cx = float(cam.get("cx", w / 2))
        cy = float(cam.get("cy", h / 2))
        cam_lines.append(f"{i} PINHOLE {w} {h} {fx} {fy} {cx} {cy}")
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    # images.txt — COLMAP uses world-to-camera quaternion + translation
    img_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for i, cam in enumerate(cameras, start=1):
        c2w = np.asarray(cam["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        qw, qx, qy, qz = _rotmat_to_quat(R)
        name = cam.get("file_name") or f"frame_{i:05d}.jpg"
        # Ensure image exists (copy or placeholder)
        src = cam.get("file_path")
        dst = images / Path(name).name
        if src and Path(src).exists() and not dst.exists():
            shutil.copy2(src, dst)
        elif not dst.exists():
            _write_placeholder_jpeg(dst, int(cam.get("w", 64)), int(cam.get("h", 48)))
        cam_id = min(i, len(cameras))
        img_lines.append(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {dst.name}")
        img_lines.append("")  # empty POINTS2D line
    (sparse / "images.txt").write_text("\n".join(img_lines) + "\n")

    # points3D.txt
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    cols_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    for i in range(means.shape[0]):
        pts_lines.append(
            f"{i+1} {means[i,0]} {means[i,1]} {means[i,2]} "
            f"{cols_u8[i,0]} {cols_u8[i,1]} {cols_u8[i,2]} 0"
        )
    (sparse / "points3D.txt").write_text("\n".join(pts_lines) + "\n")
    return out_dir


def _rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    # Shepperd's method → (w,x,y,z)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def _write_placeholder_jpeg(path: Path, w: int, h: int) -> None:
    from PIL import Image

    Image.new("RGB", (max(w, 8), max(h, 8)), (40, 40, 40)).save(path, quality=80)


def train_with_opensplat(
    project_dir: Path,
    out_ply: Path,
    *,
    iterations: int = 2000,
    opensplat: Path | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    bin_path = opensplat or find_opensplat()
    if not bin_path:
        raise RuntimeError("opensplat binary not found")
    if progress:
        progress("Training with OpenSplat (Metal/MPS)…", 0.4)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(bin_path),
        str(project_dir),
        "-n",
        str(iterations),
        "--output",
        str(out_ply),
        "--output-cameras",
        str(out_ply.with_name("cameras.json")),
    ]
    # Prefer SPZ when supported
    spz = out_ply.with_suffix(".spz")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"OpenSplat failed ({proc.returncode}): {proc.stderr[-2000:]}")
    if progress:
        progress("OpenSplat finished", 0.95)
    return {
        "trainer": "opensplat",
        "ply": str(out_ply),
        "spz": str(spz) if spz.exists() else None,
        "log": proc.stdout[-2000:],
        "exit_code": proc.returncode,
    }


def train_seeded_fallback(
    means: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict[str, Any]],
    out_dir: Path,
    *,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Seed 3DGS attributes from MapAnything points — no GPU trainer required."""
    if progress:
        progress("Seeding Gaussians from MapAnything points…", 0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / "splat.ply"
    spz = out_dir / "splat.spz"
    # Heuristic scale from nearest-neighbor spacing
    if means.shape[0] > 1:
        sample = means[: min(2000, means.shape[0])]
        d = np.linalg.norm(sample[None, :, :] - sample[:, None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = np.min(d, axis=1)
        med = float(np.median(nn[np.isfinite(nn)])) if np.any(np.isfinite(nn)) else 0.05
        scale = max(0.01, min(0.25, med * 1.5))
    else:
        scale = 0.05
    scales = np.full((means.shape[0], 3), scale, dtype=np.float32)
    write_gaussian_ply(ply, means, colors, scales=scales)
    write_spz_stub(spz, means, colors)
    write_cameras_json(out_dir / "cameras.json", cameras)
    meta = {
        "trainer": "seeded-fallback",
        "reason": "opensplat not available; exported MapAnything-seeded Gaussians",
        "num_gaussians": int(means.shape[0]),
        "scale": scale,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    if progress:
        progress("Seeded splat export done", 0.95)
    return {
        "trainer": "seeded-fallback",
        "ply": str(ply),
        "spz": str(spz),
        "cameras": str(out_dir / "cameras.json"),
        "exit_code": 0,
        **meta,
    }


def train_splat(
    means: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict[str, Any]],
    out_dir: Path,
    *,
    iterations: int = 2000,
    force_fallback: bool = False,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    opensplat = None if force_fallback else find_opensplat()
    if opensplat:
        project = write_colmap_sparse(out_dir / "opensplat_project", means, colors, cameras)
        try:
            return train_with_opensplat(
                project,
                out_dir / "splat.ply",
                iterations=iterations,
                opensplat=opensplat,
                progress=progress,
            )
        except Exception as e:
            if progress:
                progress(f"OpenSplat failed ({e}); using seeded fallback", 0.45)
    return train_seeded_fallback(means, colors, cameras, out_dir, progress=progress)
