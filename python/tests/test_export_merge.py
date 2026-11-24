"""Tile PLY merge must span all tiles, not only the last ~100 m."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dashsplat.export import merge_gaussian_plies, read_gaussian_ply_xyz_rgb, write_gaussian_ply


def test_merge_gaussian_plies_spans_all_tiles(tmp_path: Path):
    # Three tiles along +X at 0, 100, 200 m
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
    merge_gaussian_plies(paths, dest, voxel_m=0.5, max_points=10_000)
    m, _c = read_gaussian_ply_xyz_rgb(dest)
    assert m.shape[0] > 0
    span = float(m[:, 0].max() - m[:, 0].min())
    assert span > 150.0, f"merged span {span} m — expected all tiles, not last only"


def test_merge_dedupes_overlap(tmp_path: Path):
    means = np.zeros((20, 3))
    means[:, 0] = np.linspace(0, 1, 20)
    colors = np.ones((20, 3)) * 0.4
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    write_gaussian_ply(a, means, colors)
    write_gaussian_ply(b, means, colors)  # identical overlap
    dest = tmp_path / "m.ply"
    merge_gaussian_plies([a, b], dest, voxel_m=0.2)
    m, _ = read_gaussian_ply_xyz_rgb(dest)
    assert m.shape[0] < 40  # deduped
