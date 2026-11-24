"""Timestamps must be assigned before GPS filtering so times stay aligned to PTS."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dashsplat.sei import decode_partial_manual, extract_sei_from_mp4


def _gen():
    script = Path(__file__).resolve().parents[1] / "tools" / "generate_synthetic_clip.py"
    spec = importlib.util.spec_from_file_location("gsc_ts", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_bad_gps_payload_does_not_shift_later_timestamps(tmp_path: Path, monkeypatch):
    gen = _gen()
    # Build normal SEI NALs then inject a bad-GPS payload at the front of the stream
    nals = gen.make_trajectory_samples(12, 12)
    bad_proto = gen.encode_sei_metadata(
        version=1,
        gear_state=1,
        frame_seq_no=0,
        speed_mps=0.0,
        lat=0.0,  # filtered out
        lon=0.0,
        heading=0.0,
    )
    bad_nal = gen.build_sei_nal(bad_proto)
    mp4 = gen.build_minimal_mp4_with_sei([bad_nal] + nals)
    path = tmp_path / "clip.mp4"
    path.write_bytes(mp4)

    # Force PTS to a known grid: index i → i seconds (including the bad first payload)
    fake_pts = [float(i) for i in range(13)]

    def fake_probe(_path, max_frames=None):
        return fake_pts[: max_frames or len(fake_pts)]

    monkeypatch.setattr("dashsplat.accel.decode.probe_frame_pts_sec", fake_probe)

    samples = extract_sei_from_mp4(path, fps=12.0)
    assert samples
    # First *kept* sample was the second payload → must get PTS index 1 (=1.0), not 0
    assert abs(samples[0].time_sec - 1.0) < 1e-6, (
        f"expected time 1.0 after skipping bad GPS at index 0, got {samples[0].time_sec}"
    )
