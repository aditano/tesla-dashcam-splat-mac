from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dashsplat.sei import decode_partial_manual, extract_sei_from_mp4


FIXTURES = Path(__file__).resolve().parents[1] / "tools"
ROOT_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture(scope="module")
def synthetic_clips(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synth")
    # Import generate script
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "tools" / "generate_synthetic_clip.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_clip", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.main(["--out", str(out), "--timestamp", "2025-11-20_14-03-12"])
    return out


def test_decode_manual_proto():
    import importlib.util
    from pathlib import Path as P

    script = P(__file__).resolve().parents[1] / "tools" / "generate_synthetic_clip.py"
    spec = importlib.util.spec_from_file_location("gsc", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    proto = mod.encode_sei_metadata(
        version=1,
        gear_state=1,
        frame_seq_no=0,
        speed_mps=10.0,
        lat=37.3947,
        lon=-122.1503,
        heading=90.0,
    )
    sample = decode_partial_manual(proto)
    assert sample is not None
    assert sample.version == 1
    assert sample.frame_seq_no == 0
    assert abs(sample.vehicle_speed_mps - 10.0) < 1e-5
    assert abs(sample.latitude_deg - 37.3947) < 1e-6
    assert abs(sample.longitude_deg - (-122.1503)) < 1e-6
    assert abs(sample.heading_deg - 90.0) < 1e-6


def test_extract_sei_from_synthetic_mp4(synthetic_clips: Path):
    front = synthetic_clips / "2025-11-20_14-03-12-front.mp4"
    samples = extract_sei_from_mp4(front, fps=12.0)
    assert len(samples) == 36
    expected = json.loads((synthetic_clips / "expected_first.json").read_text())
    assert abs(samples[0].latitude_deg - expected["latitude_deg"]) < 1e-6
    assert abs(samples[0].longitude_deg - expected["longitude_deg"]) < 1e-6
    assert abs(samples[0].heading_deg - expected["heading_deg"]) < 1e-6

    rows = list(csv.DictReader((synthetic_clips / "expected_sei.csv").open()))
    by_seq = {int(r["frame_seq_no"]): r for r in rows}
    for idx in (0, 17, 35):
        s = samples[idx]
        exp = by_seq[idx]
        assert abs(s.latitude_deg - float(exp["latitude_deg"])) < 1e-5
        assert abs(s.longitude_deg - float(exp["longitude_deg"])) < 1e-5
