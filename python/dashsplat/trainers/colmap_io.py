"""COLMAP text model writer shared by Brush / msplat / seeded trainers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np


def rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
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


def write_colmap_sparse(
    out_dir: Path,
    means: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict[str, Any]],
    *,
    masks_dir: Path | None = None,
) -> Path:
    sparse = out_dir / "sparse" / "0"
    images = out_dir / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    cam_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for i, cam in enumerate(cameras, start=1):
        w = int(cam.get("w", 1280))
        h = int(cam.get("h", 960))
        fx = float(cam.get("fl_x", w * 0.9))
        fy = float(cam.get("fl_y", fx))
        cx = float(cam.get("cx", w / 2))
        cy = float(cam.get("cy", h / 2))
        cam_lines.append(f"{i} PINHOLE {w} {h} {fx} {fy} {cx} {cy}")
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    img_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    mask_out = out_dir / "masks"
    if masks_dir and masks_dir.exists():
        mask_out.mkdir(parents=True, exist_ok=True)

    for i, cam in enumerate(cameras, start=1):
        c2w = np.asarray(cam["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        qw, qx, qy, qz = rotmat_to_quat(R)
        name = cam.get("file_name") or f"frame_{i:05d}.jpg"
        src = cam.get("file_path")
        dst = images / Path(name).name
        if src and Path(src).exists():
            if not dst.exists() or Path(src).resolve() != dst.resolve():
                shutil.copy2(src, dst)
        elif not dst.exists():
            _write_placeholder_jpeg(dst, int(cam.get("w", 64)), int(cam.get("h", 48)))
        # Copy matching mask if provided (same stem)
        if masks_dir and masks_dir.exists():
            stem = Path(name).stem
            for ext in (".png", ".jpg", ".jpeg"):
                cand = masks_dir / f"{stem}{ext}"
                if cand.exists():
                    shutil.copy2(cand, mask_out / cand.name)
                    break
        cam_id = min(i, len(cameras))
        img_lines.append(
            f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {dst.name}"
        )
        img_lines.append("")
    (sparse / "images.txt").write_text("\n".join(img_lines) + "\n")

    pts_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    cols_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    for i in range(means.shape[0]):
        pts_lines.append(
            f"{i+1} {means[i,0]} {means[i,1]} {means[i,2]} "
            f"{int(cols_u8[i,0])} {int(cols_u8[i,1])} {int(cols_u8[i,2])} 0"
        )
    (sparse / "points3D.txt").write_text("\n".join(pts_lines) + "\n")
    return out_dir
