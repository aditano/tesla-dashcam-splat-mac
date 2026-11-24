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
from dashsplat.detail_quality import detail_for_tier, effective_iterations, effective_max_splats
from dashsplat.detect import check_event_telemetry
from dashsplat.ego_vehicle import export_ego_sidecar
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
from dashsplat.export import merge_gaussian_plies, write_gaussian_ply
from dashsplat.sei import extract_sei_from_event
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
    detail = detail_for_tier(budget.name.value)
    # Vegetation-aware splat budget (RAM-scaled) for Full thin structures
    veg_splats = effective_max_splats(detail, host.ram_gb)
    train_iters = iterations if iterations is not None else effective_iterations(detail)
    hw = profile_for_cameras(event.cameras or ["front"])
    stage_devices["host"] = host.to_dict()
    stage_devices["hw_profile"] = hw.to_dict()
    stage_devices["tuned"] = tuned.to_dict()
    stage_devices["detail"] = detail.to_dict()
    (out_dir / "devices.json").write_text(json.dumps(stage_devices, indent=2))
    (out_dir / "trainer_detail.json").write_text(json.dumps(detail.trainer_extra(), indent=2))

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
    # SEI from every clip, stitched onto the event timeline (not just minute 1)
    samples = extract_sei_from_event(event)
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

    # --- frames (all cameras, all clips in the event) ---
    t_stage = time.time()
    max_frames = max_frames_per_camera or budget.max_frames_per_camera
    _emit(progress, "Extracting frames (all cameras/clips)…", 0.12, stage="keyframes")
    check()
    offsets_path = out_dir / "timing_offsets.json"
    # Do not clobber calibrated offsets from a prior run
    if not offsets_path.exists():
        save_offsets(offsets_path, dict(DEFAULT_OFFSETS_SEC))
    images, Ks, c2ws, cameras, decode_report = extract_event_frames(
        event,
        out_dir,
        traj.poses,
        max_frames_per_camera=min(max_frames, 24) if force_fallback_trainer else max_frames,
        offsets_path=offsets_path,
        host=host,
        parallel_cameras=tuned.decode_parallel_cameras,
        progress=progress,
    )
    stage_devices["decode"] = decode_report
    if not images:
        return {"ok": False, "exit_code": 3, "error": "no frames extracted", "event_id": event.id}

    # Distance-based keyframe times — keep frames near those times (tolerance),
    # not only when the nearest pose index is exactly a keyframe index.
    key_idx = select_keyframes(
        traj.poses,
        cameras=event.cameras or ["front"],
        tier=budget.name,
    )
    (out_dir / "keyframes.json").write_text(json.dumps(key_idx, indent=2))
    if traj.poses and any(key_idx.values()):
        key_times_by_cam: dict[str, list[float]] = {}
        all_key_times: list[float] = []
        for cam_name, idxs in key_idx.items():
            times = [float(traj.poses[i].time_sec) for i in idxs if 0 <= i < len(traj.poses)]
            key_times_by_cam[cam_name] = times
            all_key_times.extend(times)
        # Tolerance: half the median keyframe spacing (fallback 0.75s)
        tol = 0.75
        if len(all_key_times) >= 2:
            sorted_t = sorted(all_key_times)
            gaps = [sorted_t[i + 1] - sorted_t[i] for i in range(len(sorted_t) - 1)]
            gaps = [g for g in gaps if g > 1e-3]
            if gaps:
                gaps_sorted = sorted(gaps)
                med = gaps_sorted[len(gaps_sorted) // 2]
                tol = max(0.25, med * 0.5)
        keep: list[int] = []
        for i, cam in enumerate(cameras):
            t = float(cam.get("time_sec", 0.0))
            times = key_times_by_cam.get(cam["camera"]) or key_times_by_cam.get("front") or all_key_times
            if not times or min(abs(t - kt) for kt in times) <= tol:
                keep.append(i)
        if keep:
            images = [images[i] for i in keep]
            Ks = [Ks[i] for i in keep]
            c2ws = [c2ws[i] for i in keep]
            cameras = [cameras[i] for i in keep]

    # --- ego 3D proxy (path-following GLB; fills masked ego region in viewer) ---
    _emit(progress, "Exporting ego vehicle path…", 0.20, stage="ego")
    ego_export = export_ego_sidecar(out_dir, traj.poses, dt=0.1)
    stage_devices["ego"] = {
        "glb": ego_export.get("glb"),
        "n_samples": ego_export.get("n_samples"),
        "model_id": (ego_export.get("manifest") or {}).get("model_id"),
    }
    stage_t["keyframes_sec"] = time.time() - t_stage

    # --- tiles (used for Full multi-pass training below) ---
    tiles = split_route_tiles(traj.poses, tier=budget.name)
    (out_dir / "tiles.json").write_text(
        json.dumps(
            [{"index": t.index, "start_m": t.start_m, "end_m": t.end_m, "n_poses": len(t.pose_indices)} for t in tiles],
            indent=2,
        )
    )

    def _speed_at(time_sec: float) -> float:
        if not traj.poses:
            return 0.0
        pidx = min(range(len(traj.poses)), key=lambda i: abs(traj.poses[i].time_sec - time_sec))
        return float(traj.poses[pidx].speed_mps)

    # --- masks (before color match so stats ignore ego/sky/movers) ---
    t_stage = time.time()
    _emit(progress, "Building masks (ego/sky/movers)…", 0.25, stage="masks")
    check()
    mask_frames = [
        {
            "path": c["file_path"],
            "camera": c["camera"],
            "file_name": c["file_name"],
            "speed_mps": _speed_at(float(c.get("time_sec", 0.0))),
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

    # Per-camera color match AFTER masks; writes copies (never overwrites originals)
    _emit(progress, "Matching per-camera color…", 0.28, stage="color")
    reps: dict[str, list[Path]] = {}
    mask_reps: dict[str, Path] = {}
    for cam in cameras:
        cid = cam["camera"]
        reps.setdefault(cid, []).append(Path(cam["file_path"]))
        if cid not in mask_reps:
            stem = Path(cam["file_name"]).stem
            for ext in (".png", ".jpg"):
                cand = masks_dir / f"{stem}{ext}"
                if cand.exists():
                    mask_reps[cid] = cand
                    break
    if reps:
        affines = match_cameras_to_front(
            reps, out_dir / "color_matched", masks=mask_reps or None
        )
        (out_dir / "color_affines.json").write_text(json.dumps(affines, indent=2))
        by_cam: dict[str, list[int]] = {}
        for i, cam in enumerate(cameras):
            by_cam.setdefault(cam["camera"], []).append(i)
        matched_dir = out_dir / "color_matched"
        matched_dir.mkdir(parents=True, exist_ok=True)
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
                dest = matched_dir / Path(cameras[i]["file_name"]).name
                Image.fromarray(matched).save(dest, quality=92)
                cameras[i]["file_path"] = str(dest)
                images[i] = dest

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
    geom_warning = None
    if pointmap.get("backend") in ("synthetic-fallback", "synthetic_cpu") or geom_device == "synthetic_cpu":
        geom_warning = (
            "MapAnything unavailable — using synthetic flat-plane geometry fallback. "
            "Splat quality will be low until MLX MapAnything weights load."
        )
        stage_devices["geometry_warning"] = geom_warning
        _emit(progress, geom_warning, 0.45, stage="geometry")
    if decode_report.get("placeholder_used") and decode_report.get("warning"):
        stage_devices["decode_warning"] = decode_report["warning"]
        _emit(progress, decode_report["warning"], 0.15, stage="keyframes")
    save_pointmap_npz(out_dir / "pointmap.npz", pointmap)
    means, colors = pointmap_to_cloud(pointmap, max_points=250_000)
    means, colors = voxel_downsample(
        means, colors, voxel_m=0.08 if budget.name == QualityTier.PREVIEW else 0.05
    )
    stage_t["geometry_sec"] = time.time() - t_stage

    # --- train (per tile in Full when route is long enough; single pass Preview) ---
    t_stage = time.time()
    iters = train_iters
    splat_cap = max(tuned.max_splats, veg_splats)
    train_dev = preferred_train_device(host).value
    _emit(
        progress,
        f"Training splat ({budget.name.value}, {iters} iters, {train_dev}, detail={detail.densify_mode})…",
        0.6,
        stage="train",
        eta_sec=max(0.0, budget.eta_prior_sec - (time.time() - t0)),
    )
    check()

    def _cameras_for_tile(tile_pose_indices: list[int]) -> list[dict[str, Any]]:
        if not traj.poses or not tile_pose_indices:
            return cameras
        allowed = set(tile_pose_indices)
        out: list[dict[str, Any]] = []
        for c in cameras:
            t = float(c.get("time_sec", 0.0))
            pidx = min(range(len(traj.poses)), key=lambda i: abs(traj.poses[i].time_sec - t))
            if pidx in allowed:
                out.append(c)
        return out or cameras

    def _means_for_tile(tile) -> tuple[np.ndarray, np.ndarray]:
        """Keep points whose ground projection falls near poses in this tile."""
        if means.shape[0] == 0 or not traj.poses or len(tiles) <= 1:
            return means, colors
        idxs = tile.pose_indices
        xs = np.array([traj.poses[i].position[0] for i in idxs], dtype=np.float64)
        ys = np.array([traj.poses[i].position[1] for i in idxs], dtype=np.float64)
        pad = 25.0
        xmin, xmax = float(xs.min()) - pad, float(xs.max()) + pad
        ymin, ymax = float(ys.min()) - pad, float(ys.max()) + pad
        mask = (
            (means[:, 0] >= xmin)
            & (means[:, 0] <= xmax)
            & (means[:, 1] >= ymin)
            & (means[:, 1] <= ymax)
        )
        if not np.any(mask):
            return means, colors
        return means[mask], colors[mask]

    use_tiles = budget.name == QualityTier.FULL and len(tiles) > 1
    if use_tiles:
        tile_results = []
        tile_plys: list[Path] = []
        all_means = []
        all_colors = []
        for tile in tiles:
            check()
            t_cams = _cameras_for_tile(tile.pose_indices)
            t_means, t_colors = _means_for_tile(tile)
            tile_dir = out_dir / f"tile_{tile.index:03d}"
            _emit(
                progress,
                f"Training tile {tile.index + 1}/{len(tiles)} ({len(t_cams)} cams)…",
                0.6 + 0.3 * tile.index / len(tiles),
                stage="train",
            )
            tr = train_splat(
                t_means,
                t_colors,
                t_cams,
                tile_dir,
                iterations=iters,
                max_splats=max(10_000, splat_cap // max(len(tiles), 1)),
                masks_dir=masks_dir,
                trainer_name=trainer_name,
                force_fallback=force_fallback_trainer,
                detail=detail.trainer_extra(),
                progress=progress,
            )
            tile_results.append(tr)
            all_means.append(t_means)
            all_colors.append(t_colors)
            if tr.get("ply") and Path(tr["ply"]).exists():
                tile_plys.append(Path(tr["ply"]))
        # Merge ALL tile outputs into one world-frame PLY (do not keep only the last tile)
        dest = out_dir / "splat.ply"
        if tile_plys:
            merge_gaussian_plies(tile_plys, dest, voxel_m=0.05, max_points=splat_cap)
            trainers_used = [r.get("trainer") for r in tile_results]
            train_result = {
                "trainer": trainers_used[-1],
                "ply": str(dest),
                "spz": None,
                "cameras": str(out_dir / "cameras.json"),
                "exit_code": 0,
                "num_gaussians": None,
                "n_tiles_trained": len(tile_results),
                "tile_trainers": trainers_used,
                "cpu_fallback": all(t == "seeded" for t in trainers_used),
            }
        else:
            merged_m = np.concatenate(all_means, axis=0) if all_means else means
            merged_c = np.concatenate(all_colors, axis=0) if all_colors else colors
            write_gaussian_ply(dest, merged_m, merged_c)
            train_result = train_splat(
                merged_m,
                merged_c,
                cameras,
                out_dir,
                iterations=iters,
                max_splats=splat_cap,
                masks_dir=masks_dir,
                trainer_name="seeded",
                force_fallback=True,
                detail=detail.trainer_extra(),
                progress=progress,
            )
            train_result = {**train_result, "n_tiles_trained": len(tile_results)}
        train_result["n_tiles_trained"] = len(tile_results)
    else:
        train_result = train_splat(
            means,
            colors,
            cameras,
            out_dir,
            iterations=iters,
            max_splats=splat_cap,
            masks_dir=masks_dir,
            trainer_name=trainer_name,
            force_fallback=force_fallback_trainer,
            detail=detail.trainer_extra(),
            progress=progress,
        )
    # Surface which trainer actually produced the PLY (never claim Full Metal for CPU)
    actual_trainer = train_result.get("trainer")
    cpu_fallback = bool(train_result.get("cpu_fallback")) or (
        actual_trainer == "seeded" and not force_fallback_trainer
    )
    train_warning = None
    if cpu_fallback:
        train_warning = (
            f"Seeded CPU fallback used (requested {trainer_name or 'brush'}; "
            f"fallback_from={train_result.get('fallback_from')}). "
            "Not a Metal Full success."
        )
        stage_devices["train_warning"] = train_warning
        _emit(progress, train_warning, 0.96, stage="train")
    stage_devices["train"] = {
        "preferred": train_dev,
        "trainer": actual_trainer,
        "cpu_fallback": cpu_fallback,
        "fallback_from": train_result.get("fallback_from"),
        "fallback_errors": train_result.get("fallback_errors"),
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
        "cpu_fallback": cpu_fallback,
        "fallback_from": train_result.get("fallback_from"),
        "ply": train_result.get("ply"),
        "spz": train_result.get("spz"),
        "cameras": str(out_dir / "cameras.json"),
        "masks": str(masks_dir),
        "num_gaussians": train_result.get("num_gaussians") or int(means.shape[0]),
        "n_tiles": len(tiles),
        "warning": train_warning
        or geom_warning
        or decode_report.get("warning")
        or traj.warning
        or health.message,
        "pointmap_synthetic": bool(geom_warning),
        "placeholder_frames": bool(decode_report.get("placeholder_used")),
        "has_sei": health.has_sei,
        "eta_prior_sec": budget.eta_prior_sec,
        "hw_generation": hw.generation.value,
        "devices": stage_devices,
        "ego_vehicle": ego_export,
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
