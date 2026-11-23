"""Pose interpolation + ego vehicle export for path-following GLB proxy."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dashsplat.ego_vehicle import (
    DEFAULT_GLB_NAME,
    bundled_assets_dir,
    default_manifest,
    export_ego_sidecar,
    interpolate_ego_pose,
    sample_path,
)


def _synthetic_path(n: int = 21, radius: float = 30.0) -> list[dict]:
    """Circular ENU path with heading tangent to the circle."""
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        ang = t * 2 * math.pi
        east = radius * math.cos(ang)
        north = radius * math.sin(ang)
        # Forward tangent: (-sin, cos)
        yaw = math.atan2(math.cos(ang), -math.sin(ang))
        poses.append(
            {
                "time_sec": t * 10.0,
                "position": [east, north, 0.0],
                "yaw_rad": yaw,
                "heading_deg": 90.0 - math.degrees(yaw),
            }
        )
    return poses


def test_bundled_glb_and_license_exist():
    root = bundled_assets_dir()
    assert (root / DEFAULT_GLB_NAME).is_file()
    assert (root / "LICENSE.txt").is_file()
    assert "MIT" in (root / "LICENSE.txt").read_text()
    m = default_manifest()
    assert m["license"] == "MIT"
    assert m["swap_hint"]["model_variant"] is None


def test_interpolate_midpoint_on_straight_path():
    poses = [
        {"time_sec": 0.0, "position": [0.0, 0.0, 0.0], "yaw_rad": 0.0, "heading_deg": 90.0},
        {"time_sec": 2.0, "position": [10.0, 0.0, 0.0], "yaw_rad": 0.0, "heading_deg": 90.0},
    ]
    mid = interpolate_ego_pose(poses, 1.0)
    assert mid is not None
    assert abs(mid.east - 5.0) < 1e-6
    assert abs(mid.north) < 1e-6
    assert abs(mid.yaw_rad) < 1e-6


def test_interpolate_yaw_shortest_arc():
    poses = [
        {"time_sec": 0.0, "position": [0, 0, 0], "yaw_rad": math.radians(170), "heading_deg": 0},
        {"time_sec": 1.0, "position": [0, 0, 0], "yaw_rad": math.radians(-170), "heading_deg": 0},
    ]
    mid = interpolate_ego_pose(poses, 0.5)
    assert mid is not None
    # Shortest path crosses ±180 → mid near ±180
    assert abs(abs(math.degrees(mid.yaw_rad)) - 180.0) < 5.0


def test_sample_path_density_and_endpoints():
    poses = _synthetic_path(11)
    samples = sample_path(poses, dt=0.5)
    assert len(samples) >= 10
    assert abs(samples[0].time_sec - 0.0) < 1e-6
    assert abs(samples[-1].time_sec - 10.0) < 1e-6
    # Car stays on the circle radius ~30
    for s in samples[::3]:
        r = math.hypot(s.east, s.north)
        assert 28.0 < r < 32.0


def test_export_ego_sidecar(tmp_path: Path):
    poses = _synthetic_path(15)
    info = export_ego_sidecar(tmp_path, poses, dt=0.2)
    assert Path(info["glb"]).is_file()
    path_json = Path(info["path"])
    assert path_json.is_file()
    data = json.loads(path_json.read_text())
    assert data["coordinate_frame"] == "enu"
    assert len(data["samples"]) == info["n_samples"]
    assert (tmp_path / "ego" / "ego_vehicle.manifest.json").is_file()
    assert (tmp_path / "ego_path.json").is_file()
    # First sample has Y-up remapped position
    s0 = data["samples"][0]
    assert "position_y_up" in s0
    assert len(s0["position_y_up"]) == 3
