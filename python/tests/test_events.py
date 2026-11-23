from __future__ import annotations

import json
from pathlib import Path

from dashsplat.events import (
    format_event_label,
    group_into_events,
    parse_event_json,
    parse_timestamp_key,
    scan_teslacam,
    should_ignore_file,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x00")
    return path


def test_should_ignore_file():
    assert should_ignore_file("thumb.png")
    assert should_ignore_file(".DS_Store")
    assert should_ignore_file("foo.tmp")
    assert not should_ignore_file("2025-11-20_14-03-12-front.mp4")


def test_parse_timestamp_and_label():
    d = parse_timestamp_key("2025-11-20_14-03-12")
    assert d is not None
    assert format_event_label(d) == "2025-11-20 14:03:12"


def test_parse_event_json():
    info = parse_event_json(
        json.dumps({"city": "Palo Alto", "est_lat": 37.44, "est_lon": -122.14, "reason": "sentry_aware"})
    )
    assert info is not None
    assert info["city"] == "Palo Alto"
    assert abs(info["latitude"] - 37.44) < 1e-6
    assert abs(info["longitude"] - (-122.14)) < 1e-6


def test_group_single_timestamp(tmp_path: Path):
    files = [
        (_touch(tmp_path / "2025-11-20_14-03-12-front.mp4"), "2025-11-20_14-03-12-front.mp4"),
        (_touch(tmp_path / "2025-11-20_14-03-12-back.mp4"), "2025-11-20_14-03-12-back.mp4"),
        (_touch(tmp_path / "thumb.png"), "thumb.png"),
    ]
    events = group_into_events(files)
    assert len(events) == 1
    assert events[0].cameras == ["back", "front"]
    assert len(events[0].segments) == 1


def test_stitch_event_folder_segments(tmp_path: Path):
    folder = "SavedClips/2025-11-20_14-03-12"
    files = [
        (
            _touch(tmp_path / folder / "2025-11-20_14-03-12-front.mp4"),
            f"{folder}/2025-11-20_14-03-12-front.mp4",
        ),
        (
            _touch(tmp_path / folder / "2025-11-20_14-04-12-front.mp4"),
            f"{folder}/2025-11-20_14-04-12-front.mp4",
        ),
        (
            _touch(tmp_path / folder / "2025-11-20_14-03-12-back.mp4"),
            f"{folder}/2025-11-20_14-03-12-back.mp4",
        ),
    ]
    events = group_into_events(files)
    assert len(events) == 1
    assert events[0].id == "2025-11-20_14-03-12"
    assert len(events[0].segments) == 2
    assert events[0].segments[1].time_offset_sec == 60
    assert "front" in events[0].cameras
    assert "back" in events[0].cameras
    assert events[0].source_folder == "SavedClips"


def test_separate_saved_and_sentry_events(tmp_path: Path):
    a = "SavedClips/2025-11-20_14-03-12"
    b = "SentryClips/2025-11-20_15-00-00"
    files = [
        (_touch(tmp_path / a / "2025-11-20_14-03-12-front.mp4"), f"{a}/2025-11-20_14-03-12-front.mp4"),
        (_touch(tmp_path / b / "2025-11-20_15-00-00-front.mp4"), f"{b}/2025-11-20_15-00-00-front.mp4"),
    ]
    (tmp_path / a / "event.json").write_text(
        json.dumps({"city": "Palo Alto", "est_lat": 37.4, "est_lon": -122.1})
    )
    (tmp_path / b / "event.json").write_text(
        json.dumps({"city": "Mountain View", "est_lat": 37.39, "est_lon": -122.08})
    )
    events = scan_teslacam(tmp_path)
    assert len(events) == 2
    ids = {e.id for e in events}
    assert ids == {"2025-11-20_14-03-12", "2025-11-20_15-00-00"}
    by_id = {e.id: e for e in events}
    assert by_id["2025-11-20_14-03-12"].location_label == "Palo Alto"
    assert by_id["2025-11-20_15-00-00"].location_label == "Mountain View"


def test_recent_clips_adjacent_merge(tmp_path: Path):
    files = [
        (
            _touch(tmp_path / "RecentClips" / "2025-11-20_14-03-12-front.mp4"),
            "RecentClips/2025-11-20_14-03-12-front.mp4",
        ),
        (
            _touch(tmp_path / "RecentClips" / "2025-11-20_14-04-12-front.mp4"),
            "RecentClips/2025-11-20_14-04-12-front.mp4",
        ),
    ]
    events = group_into_events(files)
    assert len(events) >= 1
    total_segs = sum(len(e.segments) for e in events)
    assert total_segs == 2
