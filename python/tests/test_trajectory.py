from __future__ import annotations

import math

import numpy as np

from dashsplat.cameras import CAMERA_RIG
from dashsplat.sei import SeiSample
from dashsplat.trajectory import (
    Pose,
    apply_vehicle_camera_offset,
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


def test_fallback_without_gps_marks_unknown_no_invented_prior():
    traj = build_trajectory_from_sei([])
    assert traj.source == "unknown"
    assert traj.warning
    assert traj.poses == []


def test_pose_times_keep_event_clock_not_rezeroed():
    samples = [
        SeiSample(1, 1, 0, 10.0, 37.39, -122.15, 90.0, time_sec=60.0 + i)
        for i in range(5)
    ]
    # nudge GPS so samples stay usable
    for i, s in enumerate(samples):
        s.longitude_deg = -122.15 + i * 1e-5
    traj = build_trajectory_from_sei(samples)
    assert traj.poses[0].time_sec == 60.0
    assert traj.poses[-1].time_sec == 64.0


def test_c2w_right_vector_faces_east_when_heading_north():
    """Facing north (heading 0° → ENU yaw +90°): right column must be +east."""
    pose = Pose(time_sec=0, position=(0, 0, 0), yaw_rad=heading_deg_to_enu_yaw(0.0), heading_deg=0, speed_mps=0)
    c2w = np.asarray(pose.c2w_matrix(), dtype=np.float64)
    right = c2w[:3, 0]
    forward = c2w[:3, 2]
    # forward ≈ north (+Y), right ≈ east (+X)
    assert forward[1] > 0.9
    assert right[0] > 0.9
    assert abs(right[1]) < 0.1


def test_left_repeater_forward_has_positive_left_component():
    """Left repeater on +Y body must look left/rearward (west when vehicle north)."""
    vehicle = Pose(
        time_sec=0,
        position=(0, 0, 0),
        yaw_rad=heading_deg_to_enu_yaw(0.0),
        heading_deg=0,
        speed_mps=0,
    )
    cam = CAMERA_RIG["left_repeater"]
    cam_pose = apply_vehicle_camera_offset(vehicle, cam.position, cam.rotation_deg[1])
    c2w = np.asarray(cam_pose.c2w_matrix(z=cam.position[2]), dtype=np.float64)
    forward = c2w[:3, 2]
    # Vehicle north: left = west = −east. Forward should have negative east and rearward (−north).
    assert forward[0] < 0.0, f"left camera forward east component should be <0, got {forward}"
    assert forward[1] < 0.0, f"left camera should be rear-biased, got {forward}"


def test_right_repeater_forward_has_negative_left_component():
    vehicle = Pose(
        time_sec=0,
        position=(0, 0, 0),
        yaw_rad=heading_deg_to_enu_yaw(0.0),
        heading_deg=0,
        speed_mps=0,
    )
    cam = CAMERA_RIG["right_repeater"]
    cam_pose = apply_vehicle_camera_offset(vehicle, cam.position, cam.rotation_deg[1])
    c2w = np.asarray(cam_pose.c2w_matrix(z=cam.position[2]), dtype=np.float64)
    forward = c2w[:3, 2]
    # Right = east = +X when facing north; rear-biased → −north
    assert forward[0] > 0.0, f"right camera forward east component should be >0, got {forward}"
    assert forward[1] < 0.0


def test_pillar_yaws_face_outward():
    vehicle = Pose(
        time_sec=0,
        position=(0, 0, 0),
        yaw_rad=heading_deg_to_enu_yaw(0.0),
        heading_deg=0,
        speed_mps=0,
    )
    left = apply_vehicle_camera_offset(
        vehicle, CAMERA_RIG["left_pillar"].position, CAMERA_RIG["left_pillar"].rotation_deg[1]
    )
    right = apply_vehicle_camera_offset(
        vehicle, CAMERA_RIG["right_pillar"].position, CAMERA_RIG["right_pillar"].rotation_deg[1]
    )
    lf = np.asarray(left.c2w_matrix(), dtype=np.float64)[:3, 2]
    rf = np.asarray(right.c2w_matrix(), dtype=np.float64)[:3, 2]
    assert lf[0] < 0.0  # left pillar → west
    assert rf[0] > 0.0  # right pillar → east


def test_camera_yaw_offset_subtracts_from_compass_heading():
    """Compass heading is clockwise; ENU yaw offset is CCW → opposite sign."""
    vehicle = Pose(
        time_sec=0,
        position=(0, 0, 0),
        yaw_rad=heading_deg_to_enu_yaw(90.0),
        heading_deg=90.0,
        speed_mps=0,
    )
    cam = apply_vehicle_camera_offset(vehicle, (0.0, 0.0, 0.0), cam_yaw_offset_deg=30.0)
    assert abs(cam.heading_deg - 60.0) < 1e-6
    assert abs(cam.yaw_rad - (vehicle.yaw_rad + np.radians(30.0))) < 1e-9
