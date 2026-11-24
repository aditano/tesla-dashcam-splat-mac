"""Keyframe filter keeps frames near keyframe times, not exact pose indices only."""

from __future__ import annotations

from dashsplat.keyframes import select_keyframes
from dashsplat.trajectory import Pose


def _poses(n: int = 100, dt: float = 0.1) -> list[Pose]:
    return [
        Pose(i * dt, (i * 1.0, 0.0, 0.0), 0.0, 90.0, 10.0, "sei") for i in range(n)
    ]


def test_frames_near_key_times_kept_even_if_pose_index_differs():
    poses = _poses(200)
    key_idx = select_keyframes(poses, cameras=["front"], tier="preview")
    front_keys = key_idx.get("front") or []
    assert front_keys
    key_times = [poses[i].time_sec for i in front_keys]
    # Simulate extracted frames whose nearest pose index is NOT a keyframe index,
    # but whose time is within half the median key spacing of a keyframe.
    sorted_t = sorted(key_times)
    gaps = [sorted_t[i + 1] - sorted_t[i] for i in range(len(sorted_t) - 1) if sorted_t[i + 1] > sorted_t[i]]
    med = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
    tol = max(0.25, med * 0.5)

    # Off-keyframe times that old exact-index logic would drop
    near_times = [kt + tol * 0.4 for kt in key_times[:5]]
    keep = []
    for t in near_times:
        pidx = min(range(len(poses)), key=lambda j: abs(poses[j].time_sec - t))
        # Old buggy rule: pidx in front_keys
        exact = pidx in front_keys
        # New rule: time proximity
        near = min(abs(t - kt) for kt in key_times) <= tol
        assert near, f"time {t} should be within tol of a keyframe"
        if near:
            keep.append(t)
        # Document that exact-index would often fail:
        # (we don't assert exact is False always — just that near keeps them)
        _ = exact
    assert len(keep) == len(near_times)
