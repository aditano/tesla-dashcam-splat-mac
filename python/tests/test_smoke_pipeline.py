from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dashsplat.pipeline import build_folder


def _generate_synth(out: Path) -> Path:
    script = Path(__file__).resolve().parents[1] / "tools" / "generate_synthetic_clip.py"
    spec = importlib.util.spec_from_file_location("gsc", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    # SavedClips-style tree
    event_dir = out / "SavedClips" / "2025-11-20_14-03-12"
    event_dir.mkdir(parents=True)
    mod.main(["--out", str(event_dir), "--timestamp", "2025-11-20_14-03-12"])
    (event_dir / "event.json").write_text(
        json.dumps({"city": "Palo Alto", "est_lat": 37.3947, "est_lon": -122.1503, "reason": "user_saved"})
    )
    (event_dir / "thumb.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return out


def test_smoke_pipeline_synthetic(tmp_path: Path):
    root = _generate_synth(tmp_path / "TeslaCam")
    out = tmp_path / "jobs"
    summary = build_folder(
        root,
        out,
        all_events=True,
        use_mlx=False,
        force_fallback_trainer=True,
        iterations=10,
        max_frames_per_camera=2,
        tier="preview",
    )
    assert summary["exit_code"] == 0, summary
    assert summary["ok"] is True
    assert len(summary["results"]) == 1
    r = summary["results"][0]
    assert Path(r["ply"]).exists()
    assert Path(r["spz"]).exists()
    assert Path(r["cameras"]).exists()
    assert r["trainer"] == "seeded"
    assert r["tier"] == "preview"
    assert r["trajectory_source"] == "sei"
    assert r["num_gaussians"] > 0
    assert (out / "library.json").exists()
