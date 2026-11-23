"""Route tiling: ~100 m tiles with overlap along the ENU trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dashsplat.tiers import QualityTier, budget_for
from dashsplat.trajectory import Pose


@dataclass(frozen=True)
class RouteTile:
    index: int
    start_m: float
    end_m: float
    pose_indices: list[int]


def route_length_m(poses: list[Pose]) -> float:
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(poses)):
        a, b = poses[i - 1].position, poses[i].position
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def split_route_tiles(
    poses: list[Pose],
    *,
    tier: QualityTier | str = QualityTier.PREVIEW,
    length_m: float | None = None,
    overlap_m: float | None = None,
) -> list[RouteTile]:
    budget = budget_for(tier)
    length = length_m if length_m is not None else budget.tile_length_m
    overlap = overlap_m if overlap_m is not None else budget.tile_overlap_m
    if length <= 0:
        raise ValueError("tile length must be positive")
    if overlap >= length:
        raise ValueError("overlap must be < tile length")

    if not poses:
        return []

    cum = [0.0]
    for i in range(1, len(poses)):
        a, b = poses[i - 1].position, poses[i].position
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cum[-1]
    if total < 1e-3:
        return [RouteTile(0, 0.0, 0.0, list(range(len(poses))))]

    step = length - overlap
    tiles: list[RouteTile] = []
    start = 0.0
    idx = 0
    while start < total - 1e-6:
        end = min(total, start + length)
        pose_idx = [i for i, d in enumerate(cum) if start - 1e-6 <= d <= end + 1e-6]
        if not pose_idx:
            # pick nearest
            nearest = min(range(len(cum)), key=lambda i: abs(cum[i] - (start + end) / 2))
            pose_idx = [nearest]
        tiles.append(RouteTile(idx, start, end, pose_idx))
        idx += 1
        if end >= total - 1e-6:
            break
        start += step
    return tiles
