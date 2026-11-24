"""PLY / SPZ / cameras JSON export."""

from __future__ import annotations

import json
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
    """Write a 3DGS-compatible PLY (xyz + rgb + scale + opacity + identity rotation).

    Vectorized numpy pack — avoids a Python loop over every splat.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(means.shape[0])
    if scales is None:
        scales = np.full((n, 3), 0.05, dtype=np.float32)
    if opacities is None:
        opacities = np.full((n,), 0.9, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float64), 0, 1)
    sh_c0 = 0.28209479177387814
    f_dc = ((colors - 0.5) / sh_c0).astype(np.float32)
    log_s = np.log(np.maximum(np.asarray(scales, dtype=np.float64), 1e-6)).astype(np.float32)
    o = np.clip(np.asarray(opacities, dtype=np.float64), 1e-4, 1 - 1e-4)
    logit_o = np.log(o / (1.0 - o)).astype(np.float32)
    zeros = np.zeros((n, 3), dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    # Interleave columns: x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3
    packed = np.column_stack(
        [
            means.astype(np.float32),
            zeros,
            f_dc,
            logit_o,
            log_s,
            quats,
        ]
    ).astype("<f4", copy=False)

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
        f.write(packed.tobytes(order="C"))


def math_logit(p: float) -> float:
    import math

    return math.log(p / (1.0 - p))


def write_simple_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """ASCII XYZRGB point cloud PLY (viewer-friendly fallback)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Vectorized body formatting via numpy savetxt-style stack
    body = np.column_stack([points.astype(np.float64), colors_u8.astype(np.float64)])
    with path.open("w") as f:
        f.write(header)
        np.savetxt(f, body, fmt="%.6f %.6f %.6f %d %d %d")


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
