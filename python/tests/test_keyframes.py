"""Failing-first tests for distance-based keyframe selection."""

from __future__ import annotations

import math

from dashsplat.keyframes import QualityTier, select_keyframes
from dashsplat.trajectory import Pose


def _straight_poses(n: int = 100, speed_mps: float = 10.0, dt: float = 0.1) -> list[Pose]:
    poses: list[Pose] = []
    for i in range(n):
        t = i * dt
        poses.append(
            Pose(
                time_sec=t,
                position=(speed_mps * t, 0.0, 0.0),
                yaw_rad=0.0,
                heading_deg=90.0,
                speed_mps=speed_mps,
                source="sei",
            )
        )
    return poses


def test_preview_spacing_about_3m_and_cap():
    poses = _straight_poses(200, speed_mps=10.0, dt=0.1)  # 200 m
    selected = select_keyframes(
        poses,
        cameras=["front", "back", "left_repeater", "right_repeater"],
        tier=QualityTier.PREVIEW,
        blur_scores=None,
    )
    # ~1 per 3 m over 200 m → ~67, capped at 60 per camera
    assert len(selected["front"]) <= 60
    assert len(selected["front"]) >= 40
    # Distance between consecutive selected pose indices should average ~3 m
    idxs = selected["front"]
    dists = []
    for a, b in zip(idxs, idxs[1:]):
        pa, pb = poses[a].position, poses[b].position
        dists.append(math.hypot(pb[0] - pa[0], pb[1] - pa[1]))
    assert 2.0 <= (sum(dists) / len(dists)) <= 4.5


def test_full_spacing_about_1m_and_higher_cap():
    poses = _straight_poses(400, speed_mps=10.0, dt=0.1)  # 400 m
    selected = select_keyframes(
        poses,
        cameras=["front"],
        tier=QualityTier.FULL,
        blur_scores=None,
    )
    assert len(selected["front"]) <= 300
    assert len(selected["front"]) >= 200


def test_skips_hard_turns():
    poses: list[Pose] = []
    for i in range(50):
        # Sudden 90° yaw spike in the middle
        yaw = math.radians(80) if 20 <= i <= 25 else 0.0
        poses.append(
            Pose(
                time_sec=i * 0.1,
                position=(i * 1.0, 0.0, 0.0),
                yaw_rad=yaw,
                heading_deg=math.degrees(yaw),
                speed_mps=10.0,
                source="sei",
            )
        )
    selected = select_keyframes(
        poses,
        cameras=["front"],
        tier=QualityTier.PREVIEW,
        blur_scores=None,
    )
    # Pose indices in the hard-turn window should be excluded
    assert not any(20 <= i <= 25 for i in selected["front"])


def test_skips_blurry_frames():
    poses = _straight_poses(40, speed_mps=5.0, dt=0.2)
    blur = {i: 0.9 for i in range(40)}
    blur[10] = 0.05  # very blurry
    blur[11] = 0.05
    selected = select_keyframes(
        poses,
        cameras=["front"],
        tier=QualityTier.PREVIEW,
        blur_scores={"front": blur},
    )
    assert 10 not in selected["front"]
    assert 11 not in selected["front"]
