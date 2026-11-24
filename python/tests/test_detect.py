from __future__ import annotations

from pathlib import Path

from dashsplat.detect import check_event_telemetry, looks_encrypted_mp4
from dashsplat.events import scan_teslacam


def test_looks_encrypted_detects_encv(tmp_path: Path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00" * 32 + b"encv" + b"\x00" * 32)
    assert looks_encrypted_mp4(p) is True


def test_looks_encrypted_reads_only_header(tmp_path: Path):
    """Must not need the whole file — encv past the header window is ignored."""
    p = tmp_path / "huge.mp4"
    # Header clean; marker only after 8 KiB
    p.write_bytes(b"ftypisom" + b"\x00" * 8192 + b"encv" + b"\x00" * 100)
    assert looks_encrypted_mp4(p, header_bytes=4096) is False
    assert looks_encrypted_mp4(p, header_bytes=16_000) is True


def test_sample_fixture_has_sei():
    root = Path(__file__).resolve().parents[2] / "fixtures" / "sample_teslacam"
    events = scan_teslacam(root)
    assert events
    health = check_event_telemetry(events[0])
    assert health.has_sei is True
    assert health.ok is True
    assert health.message is None
