from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dashsplat.bench import held_out_psnr_proxy, run_bench
import numpy as np


def test_held_out_psnr_proxy_runs():
    rng = np.random.default_rng(0)
    means = rng.normal(size=(200, 3))
    colors = rng.random(size=(200, 3))
    r = held_out_psnr_proxy(means, colors, holdout_every=8)
    assert "psnr_db" in r
    assert r["holdout_count"] > 0


def test_bench_cli_synthetic(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "tools" / "generate_synthetic_clip.py"
    spec = importlib.util.spec_from_file_location("gsc", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    event_dir = tmp_path / "SavedClips" / "2025-11-20_14-03-12"
    event_dir.mkdir(parents=True)
    mod.main(["--out", str(event_dir), "--timestamp", "2025-11-20_14-03-12"])
    (event_dir / "event.json").write_text(json.dumps({"city": "Test"}))
    out = tmp_path / "bench"
    report = run_bench(tmp_path, "2025-11-20_14-03-12", out, use_mlx=False, trainer="seeded")
    assert report["exit_code"] == 0
    assert report["ok"] is True
    assert "timings_sec" in report
    assert "psnr_db" in (report.get("metrics") or {}) or report.get("num_gaussians", 0) > 0
