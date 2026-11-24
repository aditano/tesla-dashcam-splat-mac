#!/usr/bin/env python3
"""Generate synthetic Tesla-style dashcam MP4s with embedded SEI telemetry.

No real Tesla sample clips with SEI were found in public sources (teslamotors/dashcam
does not ship fixtures). This script builds minimal MP4s whose mdat contains
length-prefixed H.264 SEI NALs matching Tesla's 0x42…0x69 framing + protobuf,
so the browser SEI parser and unit tests have a ground-truth fixture.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            break
    return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def encode_sei_metadata(
    *,
    version: int,
    gear_state: int,
    frame_seq_no: int,
    speed_mps: float,
    lat: float,
    lon: float,
    heading: float,
    accel_x: float = 0.0,
    accel_y: float = 0.0,
    accel_z: float = 9.8,
) -> bytes:
    parts = [
        _key(1, 0) + _varint(version),
        _key(2, 0) + _varint(gear_state),
        _key(3, 0) + _varint(frame_seq_no),
        _key(4, 5) + struct.pack("<f", speed_mps),
        _key(11, 1) + struct.pack("<d", lat),
        _key(12, 1) + struct.pack("<d", lon),
        _key(13, 1) + struct.pack("<d", heading),
        _key(14, 1) + struct.pack("<d", accel_x),
        _key(15, 1) + struct.pack("<d", accel_y),
        _key(16, 1) + struct.pack("<d", accel_z),
    ]
    return b"".join(parts)


def build_sei_nal(proto: bytes) -> bytes:
    """NAL type 6, payload type 5, Tesla UUID 0x42*16 + 0x69 + proto + 0x80."""
    uuid = bytes([0x42] * 16)
    user = uuid + bytes([0x69]) + proto + bytes([0x80])
    if len(user) >= 255:
        raise ValueError("payload too large")
    return bytes([0x06, 0x05, len(user)]) + user


def box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def fullbox(typ: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return box(typ, struct.pack(">I", (version << 24) | flags) + payload)


def _annexb_to_avcc_nals(data: bytes) -> list[bytes]:
    """Split Annex-B byte-stream into NAL units (without start codes)."""
    # Find all start-code positions and NAL payload starts
    starts: list[tuple[int, int]] = []  # (nal_start, start_code_len)
    i = 0
    while i + 3 <= len(data):
        if i + 4 <= len(data) and data[i : i + 4] == b"\x00\x00\x00\x01":
            starts.append((i + 4, 4))
            i += 4
        elif data[i : i + 3] == b"\x00\x00\x01":
            starts.append((i + 3, 3))
            i += 3
        else:
            i += 1
    nals: list[bytes] = []
    for si, (nal_start, _) in enumerate(starts):
        if si + 1 < len(starts):
            next_sc = starts[si + 1][0] - starts[si + 1][1]
            end = next_sc
        else:
            end = len(data)
        nal = data[nal_start:end]
        if nal:
            nals.append(nal)
    return nals


def generate_testsrc_bitstream(
    n_frames: int,
    *,
    width: int = 320,
    height: int = 240,
    fps: int = 12,
) -> tuple[list[bytes], bytes, bytes] | None:
    """Return (vcl_nals, sps, pps) from ffmpeg lavfi testsrc (real decodable video)."""
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("ffmpeg"):
        return None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "testsrc.h264"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}",
            "-frames:v",
            str(n_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "1",  # all IDR for simple mux
            "-bf",
            "0",
            "-f",
            "h264",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists():
            return None
        raw = out.read_bytes()
    nals = _annexb_to_avcc_nals(raw)
    sps_list = [n for n in nals if n and (n[0] & 0x1F) == 7]
    pps_list = [n for n in nals if n and (n[0] & 0x1F) == 8]
    vcl = [n for n in nals if n and (n[0] & 0x1F) in (1, 5)]
    if len(vcl) < n_frames or not sps_list or not pps_list:
        return None
    return vcl[:n_frames], sps_list[0], pps_list[0]


def generate_testsrc_idr_nals(
    n_frames: int,
    *,
    width: int = 320,
    height: int = 240,
    fps: int = 12,
) -> list[bytes] | None:
    packed = generate_testsrc_bitstream(n_frames, width=width, height=height, fps=fps)
    return packed[0] if packed else None


def build_minimal_mp4_with_sei(
    sei_nals: list[bytes],
    width: int = 320,
    height: int = 240,
    timescale: int = 90000,
    fps: int = 12,
    video_nals: list[bytes] | None = None,
    sps: bytes | None = None,
    pps: bytes | None = None,
) -> bytes:
    """Build a tiny avc1 MP4: ftyp + moov + mdat with SEI + real (or fallback) VCL NALs."""
    # Prefer real ffmpeg testsrc NALs; fall back only if unavailable
    if video_nals is None or sps is None or pps is None:
        packed = generate_testsrc_bitstream(len(sei_nals), width=width, height=height, fps=fps)
        if packed:
            video_nals, sps, pps = packed
    if video_nals is None or len(video_nals) < len(sei_nals):
        # Last-resort tiny IDR — should not run on CI with ffmpeg installed
        video_nals = [bytes([0x65, 0x88, 0x84, 0x00, 0x10])] * len(sei_nals)
        sps = sps or bytes([0x67, 0x42, 0x00, 0x1E, 0xAB, 0x40, 0xF0, 0x28, 0xD3, 0x70])
        pps = pps or bytes([0x68, 0xCE, 0x38, 0x80])

    mdat_payload = bytearray()
    sample_sizes: list[int] = []
    for i, sei in enumerate(sei_nals):
        vcl = video_nals[i]
        sample = struct.pack(">I", len(sei)) + sei + struct.pack(">I", len(vcl)) + vcl
        sample_sizes.append(len(sample))
        mdat_payload.extend(sample)

    duration = len(sei_nals) * (timescale // fps)

    # mvhd v0: creation, modification, timescale, duration, rate, volume, reserved...
    mvhd_payload = struct.pack(">IIII", 0, 0, timescale, duration)
    mvhd_payload += struct.pack(">IH", 0x00010000, 0x0100)  # rate 1.0, volume 1.0
    mvhd_payload += b"\x00" * 10
    mvhd_payload += struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    mvhd_payload += struct.pack(">6I", 0, 0, 0, 0, 0, 0)
    mvhd_payload += struct.pack(">I", 2)
    mvhd = fullbox(b"mvhd", 0, 0, mvhd_payload)

    # tkhd v0: creation, modification, track_ID, reserved, duration, ...
    tkhd_payload = struct.pack(">IIIII", 0, 0, 1, 0, duration)
    tkhd_payload += b"\x00" * 8  # reserved
    tkhd_payload += struct.pack(">hh", 0, 0)  # layer, alternate_group
    tkhd_payload += struct.pack(">H", 0x0100) + b"\x00\x00"  # volume + reserved
    tkhd_payload += struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    tkhd_payload += struct.pack(">II", width << 16, height << 16)
    tkhd = fullbox(b"tkhd", 0, 3, tkhd_payload)

    # mdhd v0
    mdhd_payload = struct.pack(">IIII", 0, 0, timescale, duration)
    mdhd_payload += struct.pack(">HH", 0x55C4, 0)  # language 'und', quality
    mdhd = fullbox(b"mdhd", 0, 0, mdhd_payload)

    hdlr_payload = struct.pack(">I", 0) + b"vide" + b"\x00" * 12 + b"VideoHandler\x00"
    hdlr = fullbox(b"hdlr", 0, 0, hdlr_payload)

    vmhd = fullbox(b"vmhd", 0, 1, struct.pack(">HHH", 0, 0, 0))
    url = fullbox(b"url ", 0, 1, b"")
    dref = fullbox(b"dref", 0, 0, struct.pack(">I", 1) + url)
    dinf = box(b"dinf", dref)

    # avcC from real SPS/PPS (profile/compat/level from SPS bytes 1..3)
    assert sps is not None and pps is not None
    avc_c = (
        bytes([0x01, sps[1], sps[2], sps[3], 0xFF, 0xE1])
        + struct.pack(">H", len(sps))
        + sps
        + bytes([0x01])
        + struct.pack(">H", len(pps))
        + pps
    )

    avc1_payload = b"\x00" * 6 + struct.pack(">H", 1)
    avc1_payload += b"\x00" * 16
    avc1_payload += struct.pack(">HH", width, height)
    avc1_payload += struct.pack(">II", 0x00480000, 0x00480000)
    avc1_payload += struct.pack(">I", 0)
    avc1_payload += struct.pack(">H", 1)
    avc1_payload += b"\x00" * 32
    avc1_payload += struct.pack(">H", 0x0018)
    avc1_payload += struct.pack(">h", -1)
    avc1_payload += box(b"avcC", avc_c)
    avc1 = box(b"avc1", avc1_payload)
    stsd = fullbox(b"stsd", 0, 0, struct.pack(">I", 1) + avc1)

    stts = fullbox(
        b"stts",
        0,
        0,
        struct.pack(">III", 1, len(sei_nals), timescale // fps),
    )
    stsz = fullbox(
        b"stsz",
        0,
        0,
        struct.pack(">II", 0, len(sample_sizes))
        + b"".join(struct.pack(">I", s) for s in sample_sizes),
    )
    stsc = fullbox(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, len(sample_sizes), 1))

    ftyp = box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2avc1mp41")

    def build_moov(chunk_offset: int) -> bytes:
        stco = fullbox(b"stco", 0, 0, struct.pack(">II", 1, chunk_offset))
        stbl = box(b"stbl", stsd + stts + stsc + stsz + stco)
        minf = box(b"minf", vmhd + dinf + stbl)
        mdia = box(b"mdia", mdhd + hdlr + minf)
        trak = box(b"trak", tkhd + mdia)
        return box(b"moov", mvhd + trak)

    # Iterate once to stabilize stco offset
    moov = build_moov(0)
    mdat_offset = len(ftyp) + len(moov) + 8
    moov = build_moov(mdat_offset)
    mdat_offset = len(ftyp) + len(moov) + 8
    moov = build_moov(mdat_offset)
    mdat = box(b"mdat", bytes(mdat_payload))
    return ftyp + moov + mdat


def make_trajectory_samples(n: int = 36, fps: int = 12) -> list[bytes]:
    lat0, lon0 = 37.3947, -122.1503
    speed = 10.0
    heading = 90.0
    nals: list[bytes] = []
    for i in range(n):
        t = i / fps
        meters_east = speed * t
        lon = lon0 + meters_east / (111320 * math.cos(math.radians(lat0)))
        proto = encode_sei_metadata(
            version=1,
            gear_state=1,
            frame_seq_no=i,
            speed_mps=speed,
            lat=lat0,
            lon=lon,
            heading=heading,
            accel_x=0.1,
            accel_y=0.0,
            accel_z=9.81,
        )
        nals.append(build_sei_nal(proto))
    return nals


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "fixtures" / "synthetic",
    )
    parser.add_argument("--timestamp", default="2025-11-20_14-03-12")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    nals = make_trajectory_samples(36, 12)
    packed = generate_testsrc_bitstream(len(nals), width=320, height=240, fps=12)
    if packed:
        video_nals, sps, pps = packed
        print(f"Using ffmpeg testsrc VCL NALs ({len(video_nals)} frames) with real SPS/PPS")
        mp4 = build_minimal_mp4_with_sei(nals, video_nals=video_nals, sps=sps, pps=pps)
        real = True
    else:
        print("WARNING: ffmpeg testsrc unavailable; using tiny fallback VCL NAL")
        mp4 = build_minimal_mp4_with_sei(nals)
        real = False
    for cam in ["front", "back", "left_repeater", "right_repeater"]:
        path = args.out / f"{args.timestamp}-{cam}.mp4"
        path.write_bytes(mp4)
        print(f"Wrote {path} ({len(mp4)} bytes, {len(nals)} SEI NALs, real_video={real})")

    lat0, lon0 = 37.3947, -122.1503
    speed = 10.0
    rows = ["frame_seq_no,vehicle_speed_mps,latitude_deg,longitude_deg,heading_deg"]
    for i in [0, 17, 35]:
        t = i / 12
        lon = lon0 + (speed * t) / (111320 * math.cos(math.radians(lat0)))
        rows.append(f"{i},{speed},{lat0},{lon},{90.0}")
    csv_path = args.out / "expected_sei.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"Wrote {csv_path}")

    (args.out / "expected_first.json").write_text(
        '{"version":1,"gear_state":"GEAR_DRIVE","frame_seq_no":0,'
        '"vehicle_speed_mps":10,"latitude_deg":37.3947,"longitude_deg":-122.1503,"heading_deg":90}\n'
    )


if __name__ == "__main__":
    main()
