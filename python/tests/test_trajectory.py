from __future__ import annotations

import math

from dashsplat.sei import SeiSample
from dashsplat.trajectory import (
    build_trajectory_from_sei,
    create_enu_origin,
    heading_deg_to_enu_yaw,
    llh_to_enu,
)


def test_enu_east_west():
    origin = create_enu_origin(37.3947, -122.1503)
    # Move ~111.32m east ≈ 0.001 deg lon / cos(lat)
    dlon = 100.0 / origin.meters_per_deg_lon
    e, n, u = llh_to_enu(37.3947, -122.1503 + dlon, origin)
    assert abs(e - 100.0) < 0.5
    assert abs(n) < 0.5
    assert u == 0.0


def test_heading_north_is_plus_y():
    yaw = heading_deg_to_enu_yaw(0.0)  # north
    assert abs(yaw - math.radians(90)) < 1e-9


def test_build_trajectory_from_sei_samples():
    samples = []
    lat0, lon0 = 37.3947, -122.1503
    for i in range(36):
        t = i / 12.0
        lon = lon0 + (10.0 * t) / (111320 * math.cos(math.radians(lat0)))
        samples.append(
            SeiSample(
                version=1,
                gear_state=1,
                frame_seq_no=i,
                vehicle_speed_mps=10.0,
                latitude_deg=lat0,
                longitude_deg=lon,
                heading_deg=90.0,
                time_sec=t,
            )
        )
    traj = build_trajectory_from_sei(samples)
    assert traj.source == "sei"
    assert len(traj.poses) == 36
    # After ~3s at 10 m/s heading east, expect ~30m east
    last = traj.poses[-1]
    assert last.position[0] > 20.0
    assert abs(last.position[1]) < 5.0


def test_fallback_without_gps():
    traj = build_trajectory_from_sei([])
    assert traj.source == "filename-fallback"
    assert traj.warning
    assert len(traj.poses) >= 2
