#!/usr/bin/env python3
"""Extract Tesla SEI poses and write nerfstudio / COLMAP-style transforms.json.

Usage:
  python tools/sei_to_transforms.py /path/to/clip-dir --out tools/output/transforms.json

Then train with gsplat or nerfstudio splatfacto (see README / train_gsplat.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

# Reuse SEI framing from generate_synthetic_clip / Tesla sei_extractor logic.
NAL_ID_SEI = 6
NAL_SEI_USER = 5


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


def iter_sei_payloads(path: Path):
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


def try_decode_with_protobuf(payload: bytes) -> dict | None:
    try:
        from google.protobuf.json_format import MessageToDict

        import dashcam_pb2  # type: ignore
    except Exception:
        return decode_partial_manual(payload)

    meta = dashcam_pb2.SeiMetadata()
    try:
        meta.ParseFromString(payload)
    except Exception:
        return None
    return MessageToDict(meta, preserving_proto_field_name=True)


def decode_partial_manual(payload: bytes) -> dict | None:
    """Minimal protobuf decode for fields we need (no protoc required)."""
    out: dict = {}
    i = 0
    try:
        while i < len(payload):
            key, i = _read_varint(payload, i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                val, i = _read_varint(payload, i)
                if field == 1:
                    out["version"] = val
                elif field == 2:
                    out["gear_state"] = val
                elif field == 3:
                    out["frame_seq_no"] = val
            elif wire == 5:
                val = struct.unpack_from("<f", payload, i)[0]
                i += 4
                if field == 4:
                    out["vehicle_speed_mps"] = val
            elif wire == 1:
                val = struct.unpack_from("<d", payload, i)[0]
                i += 8
                if field == 11:
                    out["latitude_deg"] = val
                elif field == 12:
                    out["longitude_deg"] = val
                elif field == 13:
                    out["heading_deg"] = val
            elif wire == 2:
                length, i = _read_varint(payload, i)
                i += length
            else:
                return out or None
    except Exception:
        return out or None
    return out or None


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


def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    east = (lon - lon0) * meters_per_deg_lon
    north = (lat - lat0) * meters_per_deg_lat
    return east, north


def heading_to_c2w(east: float, north: float, heading_deg: float, z: float = 1.2) -> list[list[float]]:
    """Camera-to-world for a forward-facing camera at vehicle pose (nerfstudio OpenCV-ish)."""
    yaw = math.radians(90.0 - heading_deg)  # ENU yaw
    c, s = math.cos(yaw), math.sin(yaw)
    # Columns: right, down, forward — right = down × forward (same as trajectory.py)
    down = [0.0, 0.0, -1.0]
    forward = [c, s, 0.0]
    right = [s, -c, 0.0]
    return [
        [right[0], down[0], forward[0], east],
        [right[1], down[1], forward[1], north],
        [right[2], down[2], forward[2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path, help="MP4 file or directory of Tesla clips")
    parser.add_argument("--out", type=Path, default=Path("tools/output/transforms.json"))
    parser.add_argument("--camera", default="front", help="Which camera filename suffix to prefer")
    args = parser.parse_args()

    files: list[Path]
    if args.clips.is_file():
        files = [args.clips]
    else:
        files = sorted(args.clips.glob(f"*-{args.camera}.mp4"))
        if not files:
            files = sorted(args.clips.glob("*.mp4"))
    if not files:
        print("No MP4s found", file=sys.stderr)
        sys.exit(1)

    samples: list[dict] = []
    for path in files:
        for payload in iter_sei_payloads(path):
            decoded = try_decode_with_protobuf(payload)
            if decoded and "latitude_deg" in decoded:
                samples.append(decoded)

    if not samples:
        print("No SEI GPS samples found. Need firmware 2025.44.25+ clips.", file=sys.stderr)
        sys.exit(2)

    lat0 = float(samples[0]["latitude_deg"])
    lon0 = float(samples[0]["longitude_deg"])
    frames = []
    for i, s in enumerate(samples):
        east, north = ll_to_enu(float(s["latitude_deg"]), float(s["longitude_deg"]), lat0, lon0)
        heading = float(s.get("heading_deg", 0))
        frames.append(
            {
                "file_path": f"images/frame_{i:05d}",
                "transform_matrix": heading_to_c2w(east, north, heading),
                "time": i / 36.0,
                "speed_mps": s.get("vehicle_speed_mps"),
                "heading_deg": heading,
            }
        )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": 1280 * 0.5 / math.tan(math.radians(50) * 0.5),
        "fl_y": 1280 * 0.5 / math.tan(math.radians(50) * 0.5),
        "cx": 640,
        "cy": 480,
        "w": 1280,
        "h": 960,
        "frames": frames,
        "origin_lat_lon": [lat0, lon0],
        "notes": "Poses from Tesla SEI telemetry (teslamotors/dashcam schema). Extract frames separately for training.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(transforms, indent=2))
    print(f"Wrote {args.out} with {len(frames)} frames")


if __name__ == "__main__":
    main()
