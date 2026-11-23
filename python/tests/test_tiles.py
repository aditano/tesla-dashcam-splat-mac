from __future__ import annotations

from dashsplat.tiles import route_length_m, split_route_tiles
from dashsplat.trajectory import Pose


def _line(n: int = 201, step: float = 1.0) -> list[Pose]:
    return [
        Pose(i * 0.1, (i * step, 0.0, 0.0), 0.0, 90.0, step / 0.1, "sei") for i in range(n)
    ]


def test_route_length():
    poses = _line(101, 1.0)
    assert abs(route_length_m(poses) - 100.0) < 1e-6


def test_tiles_100m_with_20m_overlap():
    poses = _line(401, 1.0)  # 400 m
    tiles = split_route_tiles(poses, length_m=100.0, overlap_m=20.0)
    assert len(tiles) >= 4
    # First tile covers ~0-100
    assert tiles[0].start_m == 0.0
    assert abs(tiles[0].end_m - 100.0) < 1e-6
    # Second starts at 80 (100-20)
    assert abs(tiles[1].start_m - 80.0) < 1e-6
    # Overlap: last poses of tile0 should intersect first of tile1
    assert set(tiles[0].pose_indices) & set(tiles[1].pose_indices)
