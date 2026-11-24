"""Keyframes must cover the whole route, not only the first N×spacing meters."""

from __future__ import annotations

import math

from dashsplat.keyframes import QualityTier, select_keyframes
from dashsplat.trajectory import Pose


def _straight(n: int = 500, step_m: float = 1.0) -> list[Pose]:
    return [
        Pose(i * 0.1, (i * step_m, 0.0, 0.0), 0.0, 90.0, step_m / 0.1, "sei") for i in range(n)
    ]


def test_full_keyframes_reach_end_of_long_route():
    poses = _straight(400, 1.0)  # 399 m
    selected = select_keyframes(poses, cameras=["front"], tier=QualityTier.FULL)
    idxs = selected["front"]
    assert len(idxs) <= 300
    assert idxs[0] < 50
    # Cap would previously stop ~300 m in; spread must reach near the end
    assert poses[idxs[-1]].position[0] > 350.0, (
        f"last keyframe at {poses[idxs[-1]].position[0]} m — expected near end of 399 m route"
    )


def test_preview_keyframes_spread_over_200m():
    poses = _straight(201, 1.0)  # 200 m
    selected = select_keyframes(poses, cameras=["front"], tier=QualityTier.PREVIEW)
    idxs = selected["front"]
    assert len(idxs) <= 60
    assert poses[idxs[-1]].position[0] > 150.0
    # Roughly uniform spacing
    dists = [
        math.hypot(
            poses[b].position[0] - poses[a].position[0],
            poses[b].position[1] - poses[a].position[1],
        )
        for a, b in zip(idxs, idxs[1:])
    ]
    if dists:
        avg = sum(dists) / len(dists)
        assert 2.0 <= avg <= 6.0
