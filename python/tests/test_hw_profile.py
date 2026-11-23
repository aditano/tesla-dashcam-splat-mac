"""Failing-first tests for HW3/HW4 profile detection."""

from __future__ import annotations

from pathlib import Path

from dashsplat.hw_profile import (
    HWGeneration,
    detect_hw_profile,
    profile_for_cameras,
    resolve_clip_resolution,
)


def _touch_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal ftyp+mdat stub (not a real video); detection uses filename set.
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32)


def test_hw3_four_cameras_no_pillars():
    cams = {"front", "back", "left_repeater", "right_repeater"}
    p = profile_for_cameras(cams, width=1280, height=960)
    assert p.generation is HWGeneration.HW3
    assert p.cameras == ("front", "back", "left_repeater", "right_repeater")
    assert "left_pillar" not in p.cameras
    assert p.width == 1280
    assert p.height == 960
    assert p.fps_nominal == 36.0


def test_hw4_six_cameras_with_pillars():
    cams = {
        "front",
        "back",
        "left_repeater",
        "right_repeater",
        "left_pillar",
        "right_pillar",
    }
    p = profile_for_cameras(cams, width=1280, height=960)
    assert p.generation is HWGeneration.HW4
    assert "left_pillar" in p.cameras
    assert "right_pillar" in p.cameras


def test_detect_from_synthetic_hw3_folder(tmp_path: Path):
    root = tmp_path / "TeslaCam"
    ev = root / "SavedClips" / "2025-11-20_14-03-12"
    for cam in ("front", "back", "left_repeater", "right_repeater"):
        _touch_mp4(ev / f"2025-11-20_14-03-12-{cam}.mp4")
    (ev / "event.json").write_text('{"city":"Palo Alto"}')
    profile = detect_hw_profile(root)
    assert profile.generation is HWGeneration.HW3


def test_detect_from_synthetic_hw4_folder(tmp_path: Path):
    root = tmp_path / "TeslaCam"
    ev = root / "SentryClips" / "2025-11-20_15-00-00"
    for cam in (
        "front",
        "back",
        "left_repeater",
        "right_repeater",
        "left_pillar",
        "right_pillar",
    ):
        _touch_mp4(ev / f"2025-11-20_15-00-00-{cam}.mp4")
    profile = detect_hw_profile(root)
    assert profile.generation is HWGeneration.HW4


def test_multi_event_saved_sentry_recent(tmp_path: Path):
    from dashsplat.events import scan_teslacam

    root = tmp_path / "TeslaCam"
    # SavedClips HW3
    s = root / "SavedClips" / "2025-11-20_14-03-12"
    for cam in ("front", "back", "left_repeater", "right_repeater"):
        _touch_mp4(s / f"2025-11-20_14-03-12-{cam}.mp4")
    (s / "event.json").write_text('{"city":"Palo Alto","est_lat":37.3,"est_lon":-122.1}')
    # SentryClips HW3
    se = root / "SentryClips" / "2025-11-20_15-00-00"
    for cam in ("front", "back", "left_repeater", "right_repeater"):
        _touch_mp4(se / f"2025-11-20_15-00-00-{cam}.mp4")
    (se / "event.json").write_text('{"city":"Mountain View"}')
    # RecentClips flat
    for cam in ("front", "back", "left_repeater", "right_repeater"):
        _touch_mp4(root / "RecentClips" / f"2025-11-20_16-00-00-{cam}.mp4")

    events = scan_teslacam(root)
    assert len(events) >= 3
    ids = {e.id for e in events}
    assert any("14-03-12" in i for i in ids)
    assert any("15-00-00" in i for i in ids)
    profile = detect_hw_profile(root)
    assert profile.generation is HWGeneration.HW3


def test_resolution_defaults_cited_hw3():
    w, h, fps = resolve_clip_resolution(None, None, None, generation=HWGeneration.HW3)
    assert (w, h) == (1280, 960)
    assert fps == 36.0
