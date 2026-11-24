from __future__ import annotations

from pathlib import Path

from dashsplat.detect import check_event_telemetry, looks_encrypted_mp4
from dashsplat.events import scan_teslacam


def test_looks_encrypted_detects_encv(tmp_path: Path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00" * 32 + b"encv" + b"\x00" * 32)
    assert looks_encrypted_mp4(p) is True


def test_looks_encrypted_reads_head_and_tail_only(tmp_path: Path):
    """Scans head + tail windows — never the whole file; middle markers ignored."""
    p = tmp_path / "huge.mp4"
    # 64 KiB file: encv only in the middle (outside first/last 4 KiB)
    size = 64 * 1024
    data = bytearray(b"ftypisom" + b"\x00" * (size - 8))
    mid = size // 2
    data[mid : mid + 4] = b"encv"
    p.write_bytes(data)
    assert looks_encrypted_mp4(p, header_bytes=4096) is False
    # Marker in the tail window is detected
    data2 = bytearray(b"ftypisom" + b"\x00" * (size - 8))
    data2[-100:-96] = b"encv"
    p.write_bytes(data2)
    assert looks_encrypted_mp4(p, header_bytes=4096) is True


def test_sample_fixture_has_sei():
    root = Path(__file__).resolve().parents[2] / "fixtures" / "sample_teslacam"
    events = scan_teslacam(root)
    assert events
    health = check_event_telemetry(events[0])
    assert health.has_sei is True
    assert health.ok is True
    assert health.message is None
