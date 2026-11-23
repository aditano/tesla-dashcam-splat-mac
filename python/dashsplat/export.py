"""PLY / SPZ / cameras JSON export."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


def write_cameras_json(path: Path, cameras: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cameras": cameras}, indent=2))


def write_gaussian_ply(
    path: Path,
    means: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
) -> None:
    """Write a 3DGS-compatible PLY (xyz + rgb + scale + opacity + identity rotation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(means.shape[0])
    if scales is None:
        scales = np.full((n, 3), 0.05, dtype=np.float32)
    if opacities is None:
        opacities = np.full((n,), 0.9, dtype=np.float32)
    colors = np.clip(colors, 0, 1)
    # DC SH approx: (c - 0.5) / 0.28209479177387814
    sh_c0 = 0.28209479177387814
    f_dc = (colors - 0.5) / sh_c0
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # wxyz identity

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            # Store log-scale like many 3DGS exporters
            log_s = np.log(np.maximum(scales[i], 1e-6))
            # Inverse sigmoid opacity
            o = float(np.clip(opacities[i], 1e-4, 1 - 1e-4))
            logit_o = math_logit(o)
            f.write(
                struct.pack(
                    "<17f",
                    float(means[i, 0]),
                    float(means[i, 1]),
                    float(means[i, 2]),
                    0.0,
                    0.0,
                    0.0,
                    float(f_dc[i, 0]),
                    float(f_dc[i, 1]),
                    float(f_dc[i, 2]),
                    logit_o,
                    float(log_s[0]),
                    float(log_s[1]),
                    float(log_s[2]),
                    float(quats[i, 0]),
                    float(quats[i, 1]),
                    float(quats[i, 2]),
                    float(quats[i, 3]),
                )
            )


def math_logit(p: float) -> float:
    import math

    return math.log(p / (1.0 - p))


def write_simple_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """ASCII XYZRGB point cloud PLY (viewer-friendly fallback)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for i in range(n):
        lines.append(
            f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
            f"{colors_u8[i, 0]} {colors_u8[i, 1]} {colors_u8[i, 2]}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_spz_stub(path: Path, means: np.ndarray, colors: np.ndarray) -> None:
    """Minimal .spz sidecar: JSON gzip of positions+colors for Spark-compatible tooling.

    True Niantic SPZ binary needs a Metal trainer that emits SPZ (Brush/msplat when available).
    This stub keeps the export path working in the seeded fallback trainer.
    """
    import gzip

    payload = {
        "format": "dashsplat-spz-stub-v1",
        "count": int(means.shape[0]),
        "means": means.astype(np.float32).reshape(-1).tolist(),
        "colors": np.clip(colors, 0, 1).astype(np.float32).reshape(-1).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(payload).encode("utf-8"))
