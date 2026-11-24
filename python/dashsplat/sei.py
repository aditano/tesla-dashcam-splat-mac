"""Tesla SEI telemetry parser — ported from tools/sei_to_transforms.py + src/sei/parser.ts."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

NAL_ID_SEI = 6
NAL_SEI_USER = 5


@dataclass
class SeiSample:
    version: int = 0
    gear_state: int = 0
    frame_seq_no: int = 0
    vehicle_speed_mps: float = 0.0
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    heading_deg: float = 0.0
    time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "gear_state": self.gear_state,
            "frame_seq_no": self.frame_seq_no,
            "vehicle_speed_mps": self.vehicle_speed_mps,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "heading_deg": self.heading_deg,
            "time_sec": self.time_sec,
        }


def strip_emulation_prevention_bytes(data: bytes) -> bytes:
    stripped = bytearray()
    zero_count = 0
    for byte in data:
        if zero_count >= 2 and byte == 0x03:
            zero_count = 0
            continue
        stripped.append(byte)
        zero_count = 0 if byte != 0 else zero_count + 1
    return bytes(stripped)


def extract_proto_payload(nal: bytes) -> bytes | None:
    if len(nal) < 4:
        return None
    i = 3
    while i < len(nal) and nal[i] == 0x42:
        i += 1
    if i <= 3 or i + 1 >= len(nal) or nal[i] != 0x69:
        return None
    return strip_emulation_prevention_bytes(nal[i + 1 : -1])


def find_mdat(fp) -> tuple[int, int]:
    fp.seek(0)
    while True:
        header = fp.read(8)
        if len(header) < 8:
            raise RuntimeError("mdat atom not found")
        size32, atom_type = struct.unpack(">I4s", header)
        if size32 == 1:
            large = fp.read(8)
            atom_size = struct.unpack(">Q", large)[0]
            header_size = 16
        else:
            atom_size = size32 if size32 else 0
            header_size = 8
        if atom_type == b"mdat":
            payload_size = atom_size - header_size if atom_size else 0
            return fp.tell(), payload_size
        if atom_size < header_size:
            raise RuntimeError("invalid MP4 atom size")
        fp.seek(atom_size - header_size, 1)


def iter_sei_payloads(path: Path) -> Iterator[bytes]:
    with path.open("rb") as fp:
        offset, size = find_mdat(fp)
        fp.seek(offset)
        consumed = 0
        while size == 0 or consumed < size:
            header = fp.read(4)
            if len(header) < 4:
                break
            nal_size = struct.unpack(">I", header)[0]
            if nal_size < 2:
                fp.seek(nal_size, 1)
                consumed += 4 + nal_size
                continue
            first_two = fp.read(2)
            if len(first_two) != 2:
                break
            if (first_two[0] & 0x1F) != NAL_ID_SEI or first_two[1] != NAL_SEI_USER:
                fp.seek(nal_size - 2, 1)
                consumed += 4 + nal_size
                continue
            rest = fp.read(nal_size - 2)
            nal = first_two + rest
            consumed += 4 + nal_size
            payload = extract_proto_payload(nal)
            if payload:
                yield payload


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def decode_partial_manual(payload: bytes) -> SeiSample | None:
    out = SeiSample()
    i = 0
    found = False
    try:
        while i < len(payload):
            key, i = _read_varint(payload, i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                val, i = _read_varint(payload, i)
                if field == 1:
                    out.version = val
                    found = True
                elif field == 2:
                    out.gear_state = val
                elif field == 3:
                    out.frame_seq_no = val
            elif wire == 5:
                val = struct.unpack_from("<f", payload, i)[0]
                i += 4
                if field == 4:
                    out.vehicle_speed_mps = val
            elif wire == 1:
                val = struct.unpack_from("<d", payload, i)[0]
                i += 8
                if field == 11:
                    out.latitude_deg = val
                    found = True
                elif field == 12:
                    out.longitude_deg = val
                elif field == 13:
                    out.heading_deg = val
            elif wire == 2:
                length, i = _read_varint(payload, i)
                i += length
            else:
                break
    except Exception:
        return out if found else None
    return out if found else None


def extract_sei_from_mp4(path: Path, fps: float = 36.0) -> list[SeiSample]:
    """Parse SEI samples and stamp each with a real timeline.

    Prefer ffprobe packet PTS when available; else `frame_seq_no / fps`; else
    sequential index / fps. Never invent a hard-coded 60 s clip length here.
    """
    samples: list[SeiSample] = []
    for payload in iter_sei_payloads(path):
        decoded = decode_partial_manual(payload)
        if decoded and abs(decoded.latitude_deg) > 0.01 and abs(decoded.longitude_deg) > 0.01:
            samples.append(decoded)

    pts: list[float] = []
    try:
        from dashsplat.accel.decode import probe_frame_pts_sec

        pts = probe_frame_pts_sec(path)
    except Exception:
        pts = []

    for i, s in enumerate(samples):
        if pts and i < len(pts):
            s.time_sec = float(pts[i])
        elif s.frame_seq_no:
            s.time_sec = float(s.frame_seq_no) / float(fps)
        else:
            s.time_sec = float(i) / float(fps)
    return samples
