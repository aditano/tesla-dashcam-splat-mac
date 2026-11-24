"""SEI from every clip must build a continuous trajectory past minute 1."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from dashsplat.events import DashcamEvent, EventSegment
from dashsplat.sei import extract_sei_from_event
from dashsplat.trajectory import build_trajectory_from_sei


def _load_gen():
    script = Path(__file__).resolve().parents[1] / "tools" / "generate_synthetic_clip.py"
    spec = importlib.util.spec_from_file_location("gsc_multi", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_multi_clip_sei_poses_advance_after_minute_1(tmp_path: Path):
    """Two 60s-offset clips: poses after t=60 must keep moving east, not snap to clip-1 end."""
    gen = _load_gen()
    fps = 12
    n = 24
    # Clip 0: east 0→~20 m; clip 1 continues from 600 m (60s * 10 m/s)
    nals0 = gen.make_trajectory_samples(n, fps, start_east_m=0.0, start_seq=0)
    nals1 = gen.make_trajectory_samples(n, fps, start_east_m=600.0, start_seq=n)
    mp4_0 = gen.build_minimal_mp4_with_sei(nals0)
    mp4_1 = gen.build_minimal_mp4_with_sei(nals1)
    c0 = tmp_path / "2025-11-20_14-03-12-front.mp4"
    c1 = tmp_path / "2025-11-20_14-04-12-front.mp4"
    c0.write_bytes(mp4_0)
    c1.write_bytes(mp4_1)

    start = datetime(2025, 11, 20, 14, 3, 12, tzinfo=timezone.utc)
    event = DashcamEvent(
        id="multi",
        label="multi",
        start=start,
        duration_sec=120.0,
        cameras=["front"],
        location_label=None,
        latitude=37.3947,
        longitude=-122.1503,
        source_folder=str(tmp_path),
        event_json=None,
        segments=[
            EventSegment(
                timestamp_key="2025-11-20_14-03-12",
                wall_clock=start,
                time_offset_sec=0.0,
                cameras={"front": c0},
            ),
            EventSegment(
                timestamp_key="2025-11-20_14-04-12",
                wall_clock=datetime(2025, 11, 20, 14, 4, 12, tzinfo=timezone.utc),
                time_offset_sec=60.0,
                cameras={"front": c1},
            ),
        ],
    )
    samples = extract_sei_from_event(event, fps=float(fps))
    assert len(samples) == 2 * n
    assert samples[-1].time_sec >= 60.0
    assert samples[n].time_sec >= 60.0 - 1e-3

    traj = build_trajectory_from_sei(samples)
    assert traj.poses
    first_clip_end_east = traj.poses[n - 1].position[0]
    late = [p for p in traj.poses if p.time_sec >= 60.0]
    assert late, "expected poses after minute 1"
    late_east = late[-1].position[0]
    assert late_east > first_clip_end_east + 100.0, (
        f"pose after minute 1 snapped/stalled: first_end={first_clip_end_east}, late={late_east}"
    )
