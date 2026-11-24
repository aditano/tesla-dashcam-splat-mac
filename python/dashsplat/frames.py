"""Extract subsampled frames from TeslaCam MP4s for MapAnything / training."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from dashsplat.accel.decode import (
    decode_cameras_parallel,
    ffmpeg_extract_frames,
    probe_duration_sec,
)
from dashsplat.accel.device import HostInfo, detect_host
from dashsplat.cameras import CAMERA_RIG, CameraId, CameraExtrinsics, intrinsics_matrix
from dashsplat.events import DashcamEvent
from dashsplat.timing_offsets import apply_offset, load_offsets
from dashsplat.trajectory import Pose, apply_vehicle_camera_offset

ProgressCb = Callable[[str, float], None]


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_frames_ffmpeg(
    mp4: Path,
    out_dir: Path,
    *,
    max_frames: int = 8,
    prefix: str = "frame",
    host: HostInfo | None = None,
) -> tuple[list[Path], list[float]]:
    frames, _device, pts = ffmpeg_extract_frames(
        mp4, out_dir, max_frames=max_frames, prefix=prefix, host=host
    )
    return frames, pts


def extract_placeholder_frames(
    mp4: Path,
    out_dir: Path,
    *,
    max_frames: int = 4,
    prefix: str = "frame",
    width: int = 320,
    height: int = 240,
) -> list[Path]:
    """When ffmpeg is missing (or synthetic SEI-only MP4s), write solid-color placeholders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hue = sum(ord(c) for c in prefix) % 200 + 30
    frames: list[Path] = []
    for i in range(max_frames):
        img = Image.new("RGB", (width, height), (hue, 40 + i * 10, 60))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, width - 20, height - 20], outline=(220, 220, 220), width=2)
        draw.text((30, 30), f"{prefix} #{i}", fill=(240, 240, 240))
        path = out_dir / f"{prefix}_{i:04d}.jpg"
        img.save(path, quality=85)
        frames.append(path)
    return frames


def extract_event_frames(
    event: DashcamEvent,
    work_dir: Path,
    poses: list[Pose],
    *,
    max_frames_per_camera: int = 6,
    offsets_path: Path | None = None,
    host: HostInfo | None = None,
    parallel_cameras: int = 4,
    progress: ProgressCb | None = None,
) -> tuple[list[Path], list[list[list[float]]], list[list[list[float]]], list[dict], dict]:
    """Return image paths, intrinsics, c2w poses, camera dicts, and decode device report.

    Iterates **all** event segments/clips per camera. Frame timestamps come from
    the actual extracted frames (ffmpeg showinfo pts). Results are **interleaved
    by time** across cameras so MapAnything windows see multiple cameras at
    similar times.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    offsets = load_offsets(offsets_path or (work_dir / "timing_offsets.json"))
    host = host or detect_host()
    decode_devices: dict[str, str] = {}
    used_placeholder = False

    cam_ids: list[CameraId] = list(event.cameras) or ["front"]  # type: ignore[arg-type]
    use_ffmpeg = _have_ffmpeg()

    cam_clips: dict[str, list[tuple[float, Path, CameraExtrinsics]]] = {}
    for cam_id in cam_ids:
        if cam_id not in CAMERA_RIG:
            continue
        cam = CAMERA_RIG[cam_id]
        clips: list[tuple[float, Path, CameraExtrinsics]] = []
        for seg in event.segments:
            if cam_id in seg.cameras:
                clips.append((float(seg.time_offset_sec), seg.cameras[cam_id], cam))
        if clips:
            clips.sort(key=lambda x: x[0])
            cam_clips[cam_id] = clips

    jobs: list[tuple[Path, Path, str, int]] = []
    job_meta: dict[str, tuple[str, float, Path, CameraExtrinsics, int]] = {}
    for cam_id, clips in cam_clips.items():
        n_clips = len(clips)
        base = max(1, max_frames_per_camera // n_clips)
        rem = max_frames_per_camera - base * n_clips
        for si, (seg_off, mp4, cam) in enumerate(clips):
            n_here = base + (1 if si < rem else 0)
            out = work_dir / "frames" / cam_id / f"seg_{si:03d}"
            prefix = f"{cam_id}_s{si:03d}"
            job_meta[prefix] = (cam_id, seg_off, mp4, cam, n_here)
            if use_ffmpeg:
                jobs.append((mp4, out, prefix, n_here))

    decoded: dict[str, tuple[list[Path], str, list[float]]] = {}
    if jobs:
        if progress:
            progress("Decoding cameras/clips in parallel…", 0.1)
        decoded = decode_cameras_parallel(
            jobs,
            host=host,
            max_workers=max(1, min(parallel_cameras, len(jobs))),
            progress=progress,
        )

    # Collect unsorted rows, then interleave by time_sec
    rows: list[tuple[float, Path, list[list[float]], list[list[float]], dict]] = []

    for ci, cam_id in enumerate(cam_ids):
        if cam_id not in cam_clips:
            continue
        cam = CAMERA_RIG[cam_id]
        K = intrinsics_matrix(cam)
        global_fi = 0
        for si, (seg_off, mp4, _cam) in enumerate(cam_clips[cam_id]):
            prefix = f"{cam_id}_s{si:03d}"
            out = work_dir / "frames" / cam_id / f"seg_{si:03d}"
            n_here = job_meta[prefix][4]
            pts: list[float] = []
            if prefix in decoded:
                frames, device, pts = decoded[prefix]
                decode_devices[cam_id] = device
            elif use_ffmpeg:
                frames, pts = extract_frames_ffmpeg(
                    mp4, out, max_frames=n_here, prefix=prefix, host=host
                )
                decode_devices[cam_id] = "ffmpeg"
            else:
                frames = []
                decode_devices[cam_id] = "placeholder"
            if not frames:
                frames = extract_placeholder_frames(
                    mp4,
                    out,
                    max_frames=n_here,
                    prefix=prefix,
                    width=cam.width // 4,
                    height=cam.height // 4,
                )
                used_placeholder = True
                clip_dur = probe_duration_sec(mp4) or float(max(n_here - 1, 1))
                pts = [
                    clip_dur * (i / max(n_here - 1, 1)) if n_here > 1 else 0.0
                    for i in range(len(frames))
                ]
                if decode_devices.get(cam_id) in ("videotoolbox", "ffmpeg"):
                    decode_devices[cam_id] = "software_fallback_or_placeholder"
                else:
                    decode_devices[cam_id] = decode_devices.get(cam_id, "placeholder")

            while len(pts) < len(frames):
                pts.append(float(len(pts)))

            for fi, frame in enumerate(frames):
                with Image.open(frame) as im:
                    w, h = im.size
                scale_x = w / cam.width
                scale_y = h / cam.height
                K_scaled = [
                    [K[0][0] * scale_x, 0.0, K[0][2] * scale_x],
                    [0.0, K[1][1] * scale_y, K[1][2] * scale_y],
                    [0.0, 0.0, 1.0],
                ]
                local_t = float(pts[fi])
                raw_t = seg_off + local_t
                t = apply_offset(raw_t, cam_id, offsets)
                if poses:
                    pidx = min(
                        range(len(poses)),
                        key=lambda i: abs(poses[i].time_sec - t),
                    )
                    vehicle = poses[pidx]
                else:
                    from dashsplat.trajectory import Pose as P

                    vehicle = P(0, (0, 0, 0), 0, 0, 0)
                yaw_off = cam.rotation_deg[1]
                cam_pose = apply_vehicle_camera_offset(vehicle, cam.position, yaw_off)
                c2w = cam_pose.c2w_matrix(z=cam.position[2])
                name = f"{cam_id}_{global_fi:04d}.jpg"
                rows.append(
                    (
                        t,
                        frame,
                        K_scaled,
                        c2w,
                        {
                            "camera": cam_id,
                            "file_path": str(frame),
                            "file_name": name,
                            "transform_matrix": c2w,
                            "fl_x": K_scaled[0][0],
                            "fl_y": K_scaled[1][1],
                            "cx": K_scaled[0][2],
                            "cy": K_scaled[1][2],
                            "w": w,
                            "h": h,
                            "time_sec": t,
                            "segment_offset_sec": seg_off,
                            "clip_path": str(mp4),
                        },
                    )
                )
                global_fi += 1
        if progress:
            progress(f"Extracted {cam_id}", 0.1 + 0.1 * (ci + 1) / max(len(cam_ids), 1))

    # Interleave by timeline so MapAnything windows mix cameras at similar times
    rows.sort(key=lambda r: (r[0], r[4]["camera"]))
    image_paths: list[Path] = []
    Ks: list[list[list[float]]] = []
    c2ws: list[list[list[float]]] = []
    cameras: list[dict] = []
    for i, (t, frame, K_scaled, c2w, meta) in enumerate(rows):
        meta = {**meta, "id": i}
        image_paths.append(frame)
        Ks.append(K_scaled)
        c2ws.append(c2w)
        cameras.append(meta)

    report = {
        "cameras": decode_devices,
        "host_chip": host.chip,
        "placeholder_used": used_placeholder,
        "warning": (
            "Placeholder frames used (ffmpeg decode failed or unavailable). "
            "Geometry quality will be poor."
            if used_placeholder
            else None
        ),
    }
    return image_paths, Ks, c2ws, cameras, report
