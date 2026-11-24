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


def read_gaussian_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz + approximate RGB from a 3DGS or XYZRGB PLY written by this package."""
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            text = line.decode("ascii", errors="replace").rstrip("\n").rstrip("\r")
            header_lines.append(text)
            if text == "end_header":
                break
        header = "\n".join(header_lines)
        n = 0
        for hl in header_lines:
            if hl.startswith("element vertex"):
                n = int(hl.split()[-1])
                break
        props: list[tuple[str, str]] = []
        in_vertex = False
        for hl in header_lines:
            if hl.startswith("element "):
                in_vertex = hl.startswith("element vertex")
                continue
            if in_vertex and hl.startswith("property "):
                parts = hl.split()
                if len(parts) >= 3:
                    props.append((parts[1], parts[2]))
        is_ascii = "format ascii" in header
        is_le = "binary_little_endian" in header
        if is_ascii:
            means = np.zeros((n, 3), dtype=np.float64)
            colors = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                parts = f.readline().decode("ascii").split()
                by_name = {name: float(parts[j]) for j, (_, name) in enumerate(props) if j < len(parts)}
                means[i] = [by_name.get("x", 0), by_name.get("y", 0), by_name.get("z", 0)]
                if "red" in by_name:
                    colors[i] = [by_name["red"] / 255.0, by_name["green"] / 255.0, by_name["blue"] / 255.0]
                elif "f_dc_0" in by_name:
                    sh = 0.28209479177387814
                    colors[i] = [
                        by_name["f_dc_0"] * sh + 0.5,
                        by_name["f_dc_1"] * sh + 0.5,
                        by_name["f_dc_2"] * sh + 0.5,
                    ]
            return means, np.clip(colors, 0, 1)
        # binary
        type_size = {
            "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
            "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
            "int": 4, "uint": 4, "int32": 4, "uint32": 4,
            "float": 4, "float32": 4, "double": 8, "float64": 8,
        }
        stride = sum(type_size.get(t, 4) for t, _ in props)
        raw = f.read(n * stride)
        if len(raw) < n * stride:
            raise ValueError(f"short PLY body: {path}")
        endian = "<" if is_le else ">"
        dt_map = {
            "float": f"{endian}f4", "float32": f"{endian}f4",
            "double": f"{endian}f8", "float64": f"{endian}f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "ushort": f"{endian}u2", "uint16": f"{endian}u2",
            "short": f"{endian}i2", "int16": f"{endian}i2",
            "uint": f"{endian}u4", "uint32": f"{endian}u4",
            "int": f"{endian}i4", "int32": f"{endian}i4",
        }
        dtype = np.dtype([(name, dt_map.get(t, f"{endian}f4")) for t, name in props])
        arr = np.frombuffer(raw, dtype=dtype, count=n)
        means = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
        names = {name for _, name in props}
        if "red" in names:
            colors = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.float64) / 255.0
        elif "f_dc_0" in names:
            sh = 0.28209479177387814
            colors = np.column_stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]]).astype(np.float64)
            colors = colors * sh + 0.5
        else:
            colors = np.full((n, 3), 0.7, dtype=np.float64)
        return means, np.clip(colors, 0, 1)


def merge_gaussian_plies(
    paths: list[Path],
    dest: Path,
    *,
    voxel_m: float = 0.05,
    max_points: int | None = None,
) -> Path:
    """Concatenate tile PLYs in a common world frame and voxel-dedupe overlap."""
    if not paths:
        raise ValueError("no PLY paths to merge")
    all_m: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    for p in paths:
        m, c = read_gaussian_ply_xyz_rgb(p)
        if m.shape[0]:
            all_m.append(m)
            all_c.append(c)
    if not all_m:
        write_gaussian_ply(dest, np.zeros((0, 3)), np.zeros((0, 3)))
        return dest
    means = np.concatenate(all_m, axis=0)
    colors = np.concatenate(all_c, axis=0)
    if voxel_m > 0 and means.shape[0]:
        keys = np.floor(means / voxel_m).astype(np.int64)
        # Stable unique: keep first occurrence per voxel
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx)
        means = means[idx]
        colors = colors[idx]
    if max_points is not None and means.shape[0] > max_points:
        step = max(1, means.shape[0] // max_points)
        means = means[::step][:max_points]
        colors = colors[::step][:max_points]
    write_gaussian_ply(dest, means, colors)
    return dest


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
