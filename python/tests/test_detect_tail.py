from __future__ import annotations

from pathlib import Path

from dashsplat.detect import looks_encrypted_mp4


def test_looks_encrypted_reads_file_tail(tmp_path: Path):
    """moov/senc at EOF must be detected even when the head is clean."""
    p = tmp_path / "tail_encv.mp4"
    # 1 MiB of clean data, encv only in the last 1 KiB
    head = b"ftypisom" + b"\x00" * (1024 * 1024 - 8)
    tail = b"\x00" * 900 + b"encv" + b"\x00" * 100
    p.write_bytes(head + tail)
    assert looks_encrypted_mp4(p, header_bytes=256 * 1024) is True


def test_clean_middle_not_scanned(tmp_path: Path):
    """Markers only in the unscanned middle stay undetected (head+tail window)."""
    p = tmp_path / "mid.mp4"
    # 2 MiB: encv at 512 KiB offset — outside first and last 256 KiB
    size = 2 * 1024 * 1024
    data = bytearray(size)
    data[0:8] = b"ftypisom"
    data[512 * 1024 : 512 * 1024 + 4] = b"encv"
    p.write_bytes(data)
    assert looks_encrypted_mp4(p, header_bytes=256 * 1024) is False
