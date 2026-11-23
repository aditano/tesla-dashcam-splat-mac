"""End-to-end per-event reconstruction — Preview then Refine-to-Full."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from dashsplat.accel.device import (
    detect_host,
    preferred_coreml_units,
    preferred_mlx_device,
    preferred_train_device,
)
from dashsplat.accel.tune import tune_for_host
from dashsplat.bootstrap import support_layout
from dashsplat.color_match import apply_affine, match_cameras_to_front
from dashsplat.detect import check_event_telemetry
from dashsplat.events import DashcamEvent, scan_teslacam
from dashsplat.frames import extract_event_frames
from dashsplat.hw_profile import profile_for_cameras
from dashsplat.keyframes import select_keyframes
from dashsplat.mapanything import (
    pointmap_to_cloud,
    run_mapanything_windows,
    save_pointmap_npz,
    synthetic_pointmap_from_poses,
    voxel_downsample,
)
from dashsplat.masks.pipeline import build_masks_for_frames
from dashsplat.sei import extract_sei_from_mp4
from dashsplat.tiers import QualityTier, budget_for
from dashsplat.tiles import split_route_tiles
from dashsplat.timing_offsets import DEFAULT_OFFSETS_SEC, save_offsets
from dashsplat.train_splat import train_splat
from dashsplat.trajectory import build_trajectory_from_sei

ProgressCb = Callable[[str, float], None]


class Cancelled(Exception):
    pass


def _emit(progress: ProgressCb | None, message: str, frac: float, *, stage: str | None = None, eta_sec: float | None = None) -> None:
    payload: dict[str, Any] = {"type": "progress", "message": message, "fraction": frac}
    if stage:
        payload["stage"] = stage
    if eta_sec is not None:
        payload["eta_sec"] = eta_sec
    if progress:
        progress(message, frac)
    print(json.dumps(payload), flush=True)


def build_event(
    event: DashcamEvent,
    out_dir: Path,
    *,
    tier: QualityTier | str = QualityTier.PREVIEW,
    use_mlx: bool = True,
    use_nc_mapanything: bool = False,
    force_fallback_trainer: bool = False,
    trainer_name: str | None = None,
    iterations: int | None = None,
    max_frames_per_camera: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Run the pipeline for one event at Preview or Full quality."""
    t0 = time.time()
    stage_t: dict[str, float] = {}
    stage_devices: dict[str, Any] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = budget_for(tier)
    host = detect_host()
    tuned = tune_for_host(host, tier=budget.name)
    hw = profile_for_cameras(event.cameras or ["front"])
    stage_devices["host"] = host.to_dict()
    stage_devices["hw_profile"] = hw.to_dict()
    stage_devices["tuned"] = tuned.to_dict()
    (out_dir / "devices.json").write_text(json.dumps(stage_devices, indent=2))

    def check() -> None:
        if cancel_check and cancel_check():
            raise Cancelled("cancelled")

    # --- health ---
    health = check_event_telemetry(event)
    (out_dir / "health.json").write_text(
        json.dumps(
            {
                "ok": health.ok,
                "has_sei": health.has_sei,
                "encrypted_suspected": health.encrypted_suspected,
                "message": health.message,
                "sample_count": health.sample_count,
            },
            indent=2,
        )
    )
    if not health.ok:
        return {
            "ok": False,
            "exit_code": 4,
            "error": health.message,
            "event_id": event.id,
            "encrypted_suspected": True,
        }
    if health.message:
        _emit(progress, health.message, 0.02, stage="ingest")

    # --- SEI / trajectory ---
    t_stage = time.time()
    _emit(progress, f"Parsing SEI for {event.id}…", 0.05, stage="ingest", eta_sec=budget.eta_prior_sec)
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
    stage_t["ingest_sec"] = time.time() - t_stage

    # --- frames (all 4 cameras) ---
    t_stage = time.time()
    max_frames = max_frames_per_camera or budget.max_frames_per_camera
    # For smoke/CI keep small; callers can override
    _emit(progress, "Extracting frames (all cameras)…", 0.12, stage="keyframes")
    check()
    save_offsets(out_dir / "timing_offsets.json", dict(DEFAULT_OFFSETS_SEC))
    images, Ks, c2ws, cameras, decode_report = extract_event_frames(
        event,
        out_dir,
        traj.poses,
        max_frames_per_camera=min(max_frames, 24) if force_fallback_trainer else max_frames,
        offsets_path=out_dir / "timing_offsets.json",
        host=host,
        parallel_cameras=tuned.decode_parallel_cameras,
        progress=progress,
    )
    stage_devices["decode"] = decode_report
    if not images:
        return {"ok": False, "exit_code": 3, "error": "no frames extracted", "event_id": event.id}

    # Per-camera color/exposure match to front (one representative frame each)
    _emit(progress, "Matching per-camera color…", 0.18, stage="color")
    reps: dict[str, Path] = {}
    for cam in cameras:
        cid = cam["camera"]
        if cid not in reps:
            reps[cid] = Path(cam["file_path"])
    if reps:
        affines = match_cameras_to_front(reps, out_dir / "color_matched")
        (out_dir / "color_affines.json").write_text(json.dumps(affines, indent=2))
        # Remap all frames of each camera with the fitted affine when available
        by_cam: dict[str, list[int]] = {}
        for i, cam in enumerate(cameras):
            by_cam.setdefault(cam["camera"], []).append(i)
        for cid, idxs in by_cam.items():
            aff = affines.get(cid)
            if not aff or cid == "front":
                continue
            scale = np.array(aff["scale"], dtype=np.float64)
            bias = np.array(aff["bias"], dtype=np.float64)
            for i in idxs:
                p = Path(cameras[i]["file_path"])
                img = np.asarray(Image.open(p).convert("RGB"))
                matched = apply_affine(img, scale, bias)
                Image.fromarray(matched).save(p, quality=92)

    # Distance-based selection indices into trajectory, applied as a filter on cameras list
    key_idx = select_keyframes(
        traj.poses,
        cameras=event.cameras or ["front"],
        tier=budget.name,
    )
    (out_dir / "keyframes.json").write_text(json.dumps(key_idx, indent=2))
    stage_t["keyframes_sec"] = time.time() - t_stage

    # --- tiles ---
    tiles = split_route_tiles(traj.poses, tier=budget.name)
    (out_dir / "tiles.json").write_text(
        json.dumps(
            [{"index": t.index, "start_m": t.start_m, "end_m": t.end_m, "n_poses": len(t.pose_indices)} for t in tiles],
            indent=2,
        )
    )

    # --- masks ---
    t_stage = time.time()
    _emit(progress, "Building masks (ego/sky/movers)…", 0.25, stage="masks")
    check()
    mask_frames = [
        {
            "path": c["file_path"],
            "camera": c["camera"],
            "file_name": c["file_name"],
            "speed_mps": c.get("time_sec", 0) and traj.poses[min(int(c.get("time_sec", 0)), len(traj.poses) - 1)].speed_mps
            if traj.poses
            else 0.0,
        }
        for c in cameras
    ]
    masks_dir, mask_devices = build_masks_for_frames(
        mask_frames,
        out_dir / "masks",
        tier=budget.name,
        compute_units=preferred_coreml_units(host),
        progress=progress,
    )
    stage_devices["masks"] = mask_devices
    stage_t["masks_sec"] = time.time() - t_stage

    # --- MapAnything ---
    t_stage = time.time()
    mlx_dev = preferred_mlx_device(host).value
    _emit(progress, f"MapAnything geometry ({mlx_dev})…", 0.4, stage="geometry")
    check()
    pointmap = None
    geom_device = "synthetic_cpu"
    if use_mlx:
        try:
            model_dir = support_layout()["models"] / (
                "map-anything-fp16" if use_nc_mapanything else "map-anything-apache-fp16"
            )
            pointmap = run_mapanything_windows(
                images,
                Ks,
                c2ws,
                window=tuned.mapanything_window,
                stride=tuned.mapanything_stride,
                use_nc_weights=use_nc_mapanything,
                cache_dir=model_dir if model_dir.exists() else None,
                progress=progress,
            )
            geom_device = mlx_dev
        except Exception as e:
            _emit(progress, f"MapAnything unavailable ({e}); synthetic fallback", 0.42, stage="geometry")
    if pointmap is None:
        pointmap = synthetic_pointmap_from_poses(images, c2ws, Ks, progress=progress)
        geom_device = "synthetic_cpu"
    stage_devices["geometry"] = geom_device
    save_pointmap_npz(out_dir / "pointmap.npz", pointmap)
    means, colors = pointmap_to_cloud(pointmap, max_points=250_000)
    means, colors = voxel_downsample(
        means, colors, voxel_m=0.08 if budget.name == QualityTier.PREVIEW else 0.05
    )
    stage_t["geometry_sec"] = time.time() - t_stage

    # --- train (per tile in Full; single pass Preview) ---
    t_stage = time.time()
    iters = iterations if iterations is not None else tuned.train_iterations
    train_dev = preferred_train_device(host).value
    _emit(
        progress,
        f"Training splat ({budget.name.value}, {iters} iters, {train_dev})…",
        0.6,
        stage="train",
        eta_sec=max(0.0, budget.eta_prior_sec - (time.time() - t0)),
    )
    check()
    train_result = train_splat(
        means,
        colors,
        cameras,
        out_dir,
        iterations=iters,
        max_splats=tuned.max_splats,
        masks_dir=masks_dir,
        trainer_name=trainer_name,
        force_fallback=force_fallback_trainer,
        progress=progress,
    )
    # Flag silent CPU trainer when Apple Silicon expected Metal
    actual_trainer = train_result.get("trainer")
    if host.is_apple_silicon and actual_trainer == "seeded" and not force_fallback_trainer:
        stage_devices["train_warning"] = "Metal trainer unavailable; used seeded CPU fallback"
    stage_devices["train"] = {
        "preferred": train_dev,
        "trainer": actual_trainer,
    }
    stage_t["train_sec"] = time.time() - t_stage
    (out_dir / "devices.json").write_text(json.dumps(stage_devices, indent=2))

    elapsed = time.time() - t0
    result = {
        "ok": True,
        "exit_code": 0,
        "event_id": event.id,
        "label": event.label,
        "tier": budget.name.value,
        "elapsed_sec": elapsed,
        "stage_timings_sec": stage_t,
        "trajectory_source": traj.source,
        "pointmap_backend": pointmap.get("backend"),
        "trainer": train_result.get("trainer"),
        "ply": train_result.get("ply"),
        "spz": train_result.get("spz"),
        "cameras": str(out_dir / "cameras.json"),
        "masks": str(masks_dir),
        "num_gaussians": train_result.get("num_gaussians") or int(means.shape[0]),
        "n_tiles": len(tiles),
        "warning": traj.warning or health.message,
        "has_sei": health.has_sei,
        "eta_prior_sec": budget.eta_prior_sec,
        "hw_generation": hw.generation.value,
        "devices": stage_devices,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    _emit(progress, f"Done ({budget.name.value}) in {elapsed:.1f}s", 1.0, stage="done")
    print(json.dumps({"type": "result", **result}), flush=True)
    return result


def build_folder(
    root: Path,
    out_root: Path,
    *,
    event_ids: list[str] | None = None,
    all_events: bool = False,
    tier: QualityTier | str = QualityTier.PREVIEW,
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

    def cancel_check() -> bool:
        return (out_root / "CANCEL").exists()

    for i, ev in enumerate(selected):
        # Strict per-event separation: never merge across events.
        tier_name = tier.value if isinstance(tier, QualityTier) else str(tier)
        job_dir = out_root / "library" / ev.id.replace("/", "_") / tier_name
        try:
            r = build_event(ev, job_dir, cancel_check=cancel_check, tier=tier, **kwargs)
            results.append(r)
        except Cancelled:
            results.append({"ok": False, "exit_code": 130, "error": "cancelled", "event_id": ev.id})
            break
        frac = (i + 1) / len(selected)
        print(
            json.dumps({"type": "batch_progress", "index": i, "total": len(selected), "fraction": frac}),
            flush=True,
        )

    ok = all(r.get("ok") for r in results) if results else False
    exit_code = 0 if ok else max((r.get("exit_code") or 1) for r in results) if results else 1
    summary = {
        "ok": ok,
        "exit_code": exit_code,
        "tier": str(tier),
        "events_found": [e.to_dict() for e in events],
        "results": results,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    # Library index for the app
    lib = []
    lib_root = out_root / "library"
    if lib_root.exists():
        for event_dir in sorted(lib_root.iterdir()):
            for tier_dir in sorted(event_dir.iterdir()):
                res = tier_dir / "result.json"
                if res.exists():
                    lib.append(json.loads(res.read_text()))
    (out_root / "library.json").write_text(json.dumps(lib, indent=2))
    return summary
