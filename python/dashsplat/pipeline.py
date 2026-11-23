"""End-to-end per-event reconstruction pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from dashsplat.bootstrap import support_layout
from dashsplat.events import DashcamEvent, scan_teslacam
from dashsplat.frames import extract_event_frames
from dashsplat.mapanything import (
    pointmap_to_cloud,
    run_mapanything_mlx,
    save_pointmap_npz,
    synthetic_pointmap_from_poses,
)
from dashsplat.sei import extract_sei_from_mp4
from dashsplat.train_splat import train_splat
from dashsplat.trajectory import build_trajectory_from_sei
from dashsplat.yolo_mask import apply_masks_to_pointmap, mask_moving_objects

ProgressCb = Callable[[str, float], None]


class Cancelled(Exception):
    pass


def _emit(progress: ProgressCb | None, message: str, frac: float) -> None:
    if progress:
        progress(message, frac)
    print(json.dumps({"type": "progress", "message": message, "fraction": frac}), flush=True)


def build_event(
    event: DashcamEvent,
    out_dir: Path,
    *,
    use_mlx: bool = True,
    force_fallback_trainer: bool = False,
    iterations: int = 1500,
    max_frames_per_camera: int = 6,
    cancel_check: Callable[[], bool] | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for one event. Returns result dict + exit_code."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    def check() -> None:
        if cancel_check and cancel_check():
            raise Cancelled("cancelled")

    _emit(progress, f"Parsing SEI for {event.id}…", 0.05)
    check()
    samples = []
    for seg in event.segments:
        for cam, path in seg.cameras.items():
            if cam == "front" or not samples:
                samples = extract_sei_from_mp4(path)
                if samples:
                    break
        if samples:
            break

    traj = build_trajectory_from_sei(samples)
    (out_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "source": traj.source,
                "warning": traj.warning,
                "origin_lat_lon": traj.origin_lat_lon,
                "poses": [
                    {
                        "time_sec": p.time_sec,
                        "position": list(p.position),
                        "yaw_rad": p.yaw_rad,
                        "heading_deg": p.heading_deg,
                        "speed_mps": p.speed_mps,
                        "source": p.source,
                    }
                    for p in traj.poses
                ],
            },
            indent=2,
        )
    )

    _emit(progress, "Extracting frames…", 0.15)
    check()
    images, Ks, c2ws, cameras = extract_event_frames(
        event,
        out_dir,
        traj.poses,
        max_frames_per_camera=max_frames_per_camera,
        progress=progress,
    )
    if not images:
        return {"ok": False, "exit_code": 3, "error": "no frames extracted", "event_id": event.id}

    _emit(progress, "MapAnything / pointmaps…", 0.3)
    check()
    pointmap = None
    if use_mlx:
        try:
            model_dir = support_layout()["models"] / "map-anything-apache-fp16"
            pointmap = run_mapanything_mlx(
                images,
                intrinsics=Ks,
                camera_poses=c2ws if traj.source == "sei" else None,
                cache_dir=model_dir if model_dir.exists() else None,
                progress=progress,
            )
        except Exception as e:
            _emit(progress, f"MapAnything unavailable ({e}); synthetic fallback", 0.35)
    if pointmap is None:
        # No SEI → also used as MapAnything pose fallback path
        pointmap = synthetic_pointmap_from_poses(images, c2ws, Ks, progress=progress)

    save_pointmap_npz(out_dir / "pointmap.npz", pointmap)

    _emit(progress, "Masking dynamics…", 0.45)
    check()
    masks = mask_moving_objects(images, progress=progress)
    pts, cols, conf = apply_masks_to_pointmap(
        pointmap["pts3d"], pointmap["colors"], pointmap["conf"], masks
    )
    pointmap = {**pointmap, "pts3d": pts, "colors": cols, "conf": conf}
    means, colors = pointmap_to_cloud(pointmap, max_points=120_000)

    _emit(progress, "Training / seeding splat…", 0.55)
    check()
    train_result = train_splat(
        means,
        colors,
        cameras,
        out_dir,
        iterations=iterations,
        force_fallback=force_fallback_trainer,
        progress=progress,
    )

    elapsed = time.time() - t0
    result = {
        "ok": True,
        "exit_code": 0,
        "event_id": event.id,
        "label": event.label,
        "elapsed_sec": elapsed,
        "trajectory_source": traj.source,
        "pointmap_backend": pointmap.get("backend"),
        "trainer": train_result.get("trainer"),
        "ply": train_result.get("ply"),
        "spz": train_result.get("spz"),
        "cameras": str(out_dir / "cameras.json"),
        "num_gaussians": int(means.shape[0]),
        "warning": traj.warning,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    _emit(progress, f"Done in {elapsed:.1f}s", 1.0)
    print(json.dumps({"type": "result", **result}), flush=True)
    return result


def build_folder(
    root: Path,
    out_root: Path,
    *,
    event_ids: list[str] | None = None,
    all_events: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    events = scan_teslacam(root)
    if not events:
        return {"ok": False, "exit_code": 2, "error": "no events found", "events": []}
    selected = events
    if event_ids:
        wanted = set(event_ids)
        selected = [e for e in events if e.id in wanted]
    elif not all_events:
        selected = events[:1]

    results = []
    cancel_flag = {"cancelled": False}

    def cancel_check() -> bool:
        cancel_path = out_root / "CANCEL"
        if cancel_path.exists():
            cancel_flag["cancelled"] = True
            return True
        return False

    for i, ev in enumerate(selected):
        job_dir = out_root / ev.id.replace("/", "_")
        try:
            r = build_event(ev, job_dir, cancel_check=cancel_check, **kwargs)
            results.append(r)
        except Cancelled:
            results.append({"ok": False, "exit_code": 130, "error": "cancelled", "event_id": ev.id})
            break
        # Progress across batch
        frac = (i + 1) / len(selected)
        print(json.dumps({"type": "batch_progress", "index": i, "total": len(selected), "fraction": frac}), flush=True)

    ok = all(r.get("ok") for r in results) if results else False
    exit_code = 0 if ok else max((r.get("exit_code") or 1) for r in results) if results else 1
    summary = {
        "ok": ok,
        "exit_code": exit_code,
        "events_found": [e.to_dict() for e in events],
        "results": results,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
