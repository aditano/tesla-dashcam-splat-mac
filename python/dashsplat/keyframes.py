"""Distance-based keyframe selection across the HW3 rig."""

from __future__ import annotations

import math
from typing import Iterable

from dashsplat.tiers import QualityTier, budget_for
from dashsplat.trajectory import Pose

# Re-export for tests
__all__ = ["QualityTier", "select_keyframes", "yaw_rate_rad_s", "laplacian_blur_score"]


def _wrap_delta(dyaw: float) -> float:
    while dyaw > math.pi:
        dyaw -= 2 * math.pi
    while dyaw < -math.pi:
        dyaw += 2 * math.pi
    return dyaw


def yaw_rate_rad_s(poses: list[Pose], i: int) -> float:
    """Instantaneous |d(yaw)/dt| using neighbouring samples (handles plateaus via pad)."""
    if i <= 0:
        return 0.0
    dt = max(1e-3, poses[i].time_sec - poses[i - 1].time_sec)
    dyaw = _wrap_delta(poses[i].yaw_rad - poses[i - 1].yaw_rad)
    return abs(dyaw) / dt


def hard_turn_indices(poses: list[Pose], max_yaw_rate: float, pad: int = 2) -> set[int]:
    """Indices in/near hard turns.

    Frame-to-frame yaw spikes are padded, and the closed interval between the
    first and last spike in a cluster is filled so sustained turns stay excluded.
    """
    spikes = sorted(i for i in range(len(poses)) if yaw_rate_rad_s(poses, i) > max_yaw_rate)
    hard: set[int] = set()
    if not spikes:
        return hard
    # Cluster spikes that are close together (< 15 frames)
    cluster = [spikes[0]]
    clusters: list[list[int]] = []
    for s in spikes[1:]:
        if s - cluster[-1] <= 15:
            cluster.append(s)
        else:
            clusters.append(cluster)
            cluster = [s]
    clusters.append(cluster)
    for cl in clusters:
        lo = max(0, cl[0] - pad)
        hi = min(len(poses) - 1, cl[-1] + pad)
        hard.update(range(lo, hi + 1))
    return hard


def laplacian_blur_score(gray) -> float:
    """Variance of Laplacian — higher = sharper. Accepts HxW float/uint8 array."""
    import numpy as np

    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("gray must be 2D")
    # 3x3 Laplacian
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Valid convolution via pad
    padded = np.pad(g, 1, mode="edge")
    out = (
        kernel[0, 1] * padded[0:-2, 1:-1]
        + kernel[1, 0] * padded[1:-1, 0:-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 1] * padded[2:, 1:-1]
    )
    return float(out.var())


def select_keyframes(
    poses: list[Pose],
    cameras: Iterable[str],
    *,
    tier: QualityTier | str = QualityTier.PREVIEW,
    blur_scores: dict[str, dict[int, float]] | None = None,
    max_yaw_rate: float = 0.8,
    min_blur_score: float = 0.15,
) -> dict[str, list[int]]:
    """Return per-camera lists of pose indices to keep.

    Spacing is along the trajectory arc length. Hard turns (high yaw rate) and
    blurry frames (optional scores) are skipped. Caps come from the tier budget.
    """
    budget = budget_for(tier)
    cams = list(cameras)
    if not poses:
        return {c: [] for c in cams}

    # Precompute cumulative distance
    cum = [0.0]
    for i in range(1, len(poses)):
        a, b = poses[i - 1].position, poses[i].position
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))

    hard = hard_turn_indices(poses, max_yaw_rate)

    out: dict[str, list[int]] = {}
    for cam in cams:
        chosen: list[int] = []
        next_dist = 0.0
        blur_map = (blur_scores or {}).get(cam) or {}
        for i, d in enumerate(cum):
            if i in hard:
                continue
            if blur_map and blur_map.get(i, 1.0) < min_blur_score:
                continue
            if d + 1e-6 < next_dist and chosen:
                continue
            chosen.append(i)
            next_dist = d + budget.meters_per_keyframe
            if len(chosen) >= budget.max_frames_per_camera:
                break
        if not chosen:
            chosen = [0]
        out[cam] = chosen
    return out
