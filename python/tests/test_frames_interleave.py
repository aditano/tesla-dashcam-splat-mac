"""Extracted frames are interleaved by time across cameras."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from dashsplat.events import DashcamEvent, EventSegment
from dashsplat.frames import extract_event_frames
from dashsplat.trajectory import Pose


def test_cameras_interleaved_by_time(tmp_path: Path, monkeypatch):
    # Bypass ffmpeg: force placeholder path with synthetic pts via monkeypatch
    def fake_have():
        return False

    monkeypatch.setattr("dashsplat.frames._have_ffmpeg", fake_have)

    cams = ["front", "back", "left_repeater"]
    camera_paths = {}
    for cam in cams:
        p = tmp_path / f"2025-11-20_14-03-12-{cam}.mp4"
        p.write_bytes(b"not-a-real-mp4")
        camera_paths[cam] = p

    start = datetime(2025, 11, 20, 14, 3, 12, tzinfo=timezone.utc)
    event = DashcamEvent(
        id="t",
        label="t",
        start=start,
        duration_sec=60.0,
        cameras=cams,  # type: ignore[arg-type]
        location_label=None,
        latitude=None,
        longitude=None,
        source_folder=str(tmp_path),
        event_json=None,
        segments=[
            EventSegment(
                timestamp_key="2025-11-20_14-03-12",
                wall_clock=start,
                time_offset_sec=0.0,
                cameras=camera_paths,  # type: ignore[arg-type]
            )
        ],
    )
    poses = [Pose(i * 0.5, (i, 0, 0), 0.0, 90.0, 5.0, "sei") for i in range(20)]
    _imgs, _Ks, _c2ws, cameras, report = extract_event_frames(
        event, tmp_path / "work", poses, max_frames_per_camera=4
    )
    assert report.get("placeholder_used") is True
    assert report.get("warning")
    assert cameras
    # Times should be non-decreasing (sorted interleave)
    times = [c["time_sec"] for c in cameras]
    assert times == sorted(times)
    # Within a short window, multiple camera names should appear (not one cam then the next)
    first_half = cameras[: max(3, len(cameras) // 2)]
    names = {c["camera"] for c in first_half}
    assert len(names) >= 2, f"expected interleaved cameras, got order {[c['camera'] for c in cameras]}"
