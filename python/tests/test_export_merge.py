"""Tile PLY merge must span all tiles and preserve Gaussian properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dashsplat.export import (
    merge_gaussian_plies,
    read_gaussian_ply_vertices,
    read_gaussian_ply_xyz_rgb,
    write_gaussian_ply,
    write_gaussian_ply_vertices,
)


def test_merge_gaussian_plies_spans_all_tiles(tmp_path: Path):
    paths = []
    for i, x0 in enumerate([0.0, 100.0, 200.0]):
        means = np.column_stack(
            [
                np.linspace(x0, x0 + 50.0, 40),
                np.zeros(40),
                np.zeros(40),
            ]
        )
        colors = np.full((40, 3), 0.5 + 0.1 * i)
        p = tmp_path / f"tile_{i}.ply"
        write_gaussian_ply(p, means, colors)
        paths.append(p)
    dest = tmp_path / "merged.ply"
    _, n = merge_gaussian_plies(paths, dest, voxel_m=0.5, max_points=10_000)
    m, _c = read_gaussian_ply_xyz_rgb(dest)
    assert n == m.shape[0]
    span = float(m[:, 0].max() - m[:, 0].min())
    assert span > 150.0, f"merged span {span} m — expected all tiles, not last only"


def test_merge_preserves_opacity_scale_rotation(tmp_path: Path):
    means = np.zeros((8, 3), dtype=np.float32)
    means[:, 0] = np.arange(8, dtype=np.float32)
    colors = np.full((8, 3), 0.6, dtype=np.float32)
    scales = np.full((8, 3), 0.12, dtype=np.float32)
    opacities = np.full((8,), 0.77, dtype=np.float32)
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    write_gaussian_ply(a, means, colors, scales=scales, opacities=opacities)
    write_gaussian_ply(b, means + np.array([50, 0, 0], dtype=np.float32), colors, scales=scales, opacities=opacities)
    dest = tmp_path / "m.ply"
    merge_gaussian_plies([a, b], dest, voxel_m=0.0)
    props, arr = read_gaussian_ply_vertices(dest)
    names = [n for _, n in props]
    assert "opacity" in names and "scale_0" in names and "rot_0" in names
    assert float(np.mean(arr["opacity"])) != 0.0


def test_merge_fails_on_layout_mismatch(tmp_path: Path):
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    write_gaussian_ply(a, np.zeros((4, 3)), np.ones((4, 3)) * 0.5)
    # Minimal xyz-only PLY with different props
    props = [("float", "x"), ("float", "y"), ("float", "z")]
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    verts = np.zeros(4, dtype=dtype)
    write_gaussian_ply_vertices(b, props, verts)
    with pytest.raises(ValueError, match="property layout mismatch"):
        merge_gaussian_plies([a, b], tmp_path / "out.ply")


def test_overcap_subsamples_across_all_tiles_not_dropping_last(tmp_path: Path):
    paths = []
    for i, x0 in enumerate([0.0, 100.0, 200.0]):
        means = np.column_stack([np.full(100, x0 + 25.0), np.zeros(100), np.zeros(100)])
        colors = np.ones((100, 3)) * 0.5
        p = tmp_path / f"t{i}.ply"
        write_gaussian_ply(p, means, colors)
        paths.append(p)
    dest = tmp_path / "cap.ply"
    _, n = merge_gaussian_plies(paths, dest, voxel_m=0.0, max_points=30)
    assert n == 30
    m, _ = read_gaussian_ply_xyz_rgb(dest)
    # Must still include points from the last tile (~200)
    assert float(m[:, 0].max()) >= 200.0


def test_merge_dedupes_overlap(tmp_path: Path):
    means = np.zeros((20, 3))
    means[:, 0] = np.linspace(0, 1, 20)
    colors = np.ones((20, 3)) * 0.4
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    write_gaussian_ply(a, means, colors)
    write_gaussian_ply(b, means, colors)
    dest = tmp_path / "m.ply"
    merge_gaussian_plies([a, b], dest, voxel_m=0.2)
    m, _ = read_gaussian_ply_xyz_rgb(dest)
    assert m.shape[0] < 40
