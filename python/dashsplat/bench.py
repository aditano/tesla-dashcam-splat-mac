"""Benchmark harness: stage timings + held-out PSNR proxy on fixtures / real events."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def held_out_psnr_proxy(
    means: np.ndarray,
    colors: np.ndarray,
    holdout_every: int = 8,
) -> dict[str, float]:
    """Cheap geometric proxy until a real renderer is wired.

    Splits points into train/holdout by index stride and reports:
    - mean nearest-neighbour distance from holdout → train (lower better)
    - a synthetic PSNR-like score mapped from that distance for CI tracking.
    """
    if means.shape[0] < holdout_every * 2:
        return {"psnr_db": 0.0, "mean_nn_m": 0.0, "holdout_count": 0}
    idx = np.arange(means.shape[0])
    hold = idx[::holdout_every]
    train = np.setdiff1d(idx, hold)
    if train.size == 0 or hold.size == 0:
        return {"psnr_db": 0.0, "mean_nn_m": 0.0, "holdout_count": 0}
    # Sample for speed
    rng = np.random.default_rng(0)
    hold_s = hold if hold.size <= 500 else rng.choice(hold, 500, replace=False)
    train_s = train if train.size <= 2000 else rng.choice(train, 2000, replace=False)
    hpts = means[hold_s]
    tpts = means[train_s]
    # Chunked NN
    dists = []
    for i in range(0, hpts.shape[0], 50):
        chunk = hpts[i : i + 50]
        d = np.linalg.norm(chunk[:, None, :] - tpts[None, :, :], axis=-1)
        dists.append(d.min(axis=1))
    mean_nn = float(np.mean(np.concatenate(dists)))
    # Map to a soft PSNR-like score for regression tests (not photometric PSNR).
    psnr = float(10.0 * np.log10(1.0 / (mean_nn**2 + 1e-6)))
    return {"psnr_db": psnr, "mean_nn_m": mean_nn, "holdout_count": int(hold.size)}


def run_bench(
    folder: Path,
    event_id: str | None,
    out_dir: Path,
    *,
    use_mlx: bool = False,
    trainer: str = "seeded",
) -> dict[str, Any]:
    from dashsplat.events import scan_teslacam
    from dashsplat.pipeline import build_event

    events = scan_teslacam(folder)
    if not events:
        return {"ok": False, "exit_code": 2, "error": "no events"}
    if event_id:
        selected = [e for e in events if e.id == event_id]
        if not selected:
            return {"ok": False, "exit_code": 2, "error": f"event not found: {event_id}"}
        event = selected[0]
    else:
        event = events[0]

    timings: dict[str, float] = {}
    t0 = time.time()
    result = build_event(
        event,
        out_dir,
        use_mlx=use_mlx,
        force_fallback_trainer=(trainer == "seeded"),
        trainer_name=trainer,
        tier="preview",
        progress=None,
    )
    timings["total_sec"] = time.time() - t0
    timings.update(result.get("stage_timings_sec") or {})

    # Load point means if present for proxy metric
    metrics: dict[str, float] = {}
    npz = out_dir / "pointmap.npz"
    if npz.exists():
        data = np.load(npz)
        pts = data["pts3d"]
        conf = data["conf"]
        mag = np.linalg.norm(pts, axis=-1)
        mask = (conf > 0.1) & (mag > 1e-6)
        means = pts[mask]
        cols = data["colors"][mask]
        if means.shape[0] > 0:
            metrics = held_out_psnr_proxy(means.astype(np.float32), cols.astype(np.float32))

    report = {
        "ok": bool(result.get("ok")),
        "exit_code": int(result.get("exit_code", 1)),
        "event_id": event.id,
        "trainer": result.get("trainer"),
        "tier": result.get("tier"),
        "num_gaussians": result.get("num_gaussians"),
        "timings_sec": timings,
        "metrics": metrics,
        "note": "psnr_db is a geometric proxy until photometric render PSNR is wired",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"type": "bench", **report}), flush=True)
    return report
