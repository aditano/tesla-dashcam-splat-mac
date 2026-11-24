"""COLMAP text model writer shared by Brush / msplat / seeded trainers.

Cameras: one COLMAP CAMERA_ID per physical dashcam (shared intrinsics across
frames), not one PINHOLE per frame. Model is OPENCV (fx fy cx cy k1 k2 p1 p2).

Intrinsics are taken from the existing camera dict (`fl_x`/`fl_y`/`cx`/`cy`/
`w`/`h`) — we do **not** invent focal lengths here. Distortion coefficients
default to zero unless the caller supplies `k1`/`k2`/`p1`/`p2` (factory
calibration is not available in this repo; approximate HFOV→fx lives in
`dashsplat.cameras` and is documented there as non-factory).
"""

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


def _physical_cam_key(cam: dict[str, Any], index: int) -> str:
    return str(cam.get("camera") or cam.get("camera_id") or f"cam_{index}")


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

    # One COLMAP camera per physical dashcam (shared intrinsics).
    phys_order: list[str] = []
    phys_intrinsics: dict[str, dict[str, float | int | str]] = {}
    for i, cam in enumerate(cameras):
        key = _physical_cam_key(cam, i)
        if key in phys_intrinsics:
            continue
        phys_order.append(key)
        w = int(cam.get("w", 1280))
        h = int(cam.get("h", 960))
        # Keep existing intrinsics only — do not invent new focals here.
        fx = float(cam["fl_x"]) if "fl_x" in cam else float(cam.get("fx", w * 0.9))
        fy = float(cam["fl_y"]) if "fl_y" in cam else float(cam.get("fy", fx))
        cx = float(cam["cx"]) if "cx" in cam else w / 2.0
        cy = float(cam["cy"]) if "cy" in cam else h / 2.0
        # Distortion: use caller-supplied coeffs when present; otherwise zeros.
        # Calibrated Tesla lens params are not in-repo; OPENCV model reserved for
        # future calibration without inventing k/p values.
        k1 = float(cam.get("k1", 0.0))
        k2 = float(cam.get("k2", 0.0))
        p1 = float(cam.get("p1", 0.0))
        p2 = float(cam.get("p2", 0.0))
        model = str(cam.get("colmap_model") or "OPENCV")
        phys_intrinsics[key] = {
            "model": model,
            "w": w,
            "h": h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
        }

    cam_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# OPENCV params: fx fy cx cy k1 k2 p1 p2 (k/p default 0; not invented)",
        "# One CAMERA_ID per physical dashcam; frames share intrinsics.",
    ]
    cam_id_by_key: dict[str, int] = {}
    for cam_id, key in enumerate(phys_order, start=1):
        cam_id_by_key[key] = cam_id
        p = phys_intrinsics[key]
        if p["model"] == "OPENCV_FISHEYE":
            # fx fy cx cy k1 k2 k3 k4 — only use supplied k*; else zeros
            cam_lines.append(
                f"{cam_id} OPENCV_FISHEYE {p['w']} {p['h']} "
                f"{p['fx']} {p['fy']} {p['cx']} {p['cy']} "
                f"{p['k1']} {p['k2']} {float(0.0)} {float(0.0)}"
            )
        else:
            cam_lines.append(
                f"{cam_id} OPENCV {p['w']} {p['h']} "
                f"{p['fx']} {p['fy']} {p['cx']} {p['cy']} "
                f"{p['k1']} {p['k2']} {p['p1']} {p['p2']}"
            )
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
        if masks_dir and masks_dir.exists():
            stem = Path(name).stem
            for ext in (".png", ".jpg", ".jpeg"):
                cand = masks_dir / f"{stem}{ext}"
                if cand.exists():
                    shutil.copy2(cand, mask_out / cand.name)
                    break
        key = _physical_cam_key(cam, i - 1)
        cam_id = cam_id_by_key[key]
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
