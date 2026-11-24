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
    probe_frame_pts_sec,
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
) -> list[Path]:
    frames, _device = ffmpeg_extract_frames(
        mp4, out_dir, max_frames=max_frames, prefix=prefix, host=host
    )
    return frames


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


def _frame_times_sec(mp4: Path, n_frames: int, clip_duration: float | None) -> list[float]:
    """Best-effort timestamps for extracted frames within a clip."""
    pts = probe_frame_pts_sec(mp4)
    if pts and n_frames > 0:
        if len(pts) >= n_frames:
            # Evenly sample from available PTS
            idxs = [int(round(i * (len(pts) - 1) / max(n_frames - 1, 1))) for i in range(n_frames)]
            return [pts[j] for j in idxs]
        return pts[:n_frames] + [
            pts[-1] + (i + 1) * 0.1 for i in range(n_frames - len(pts))
        ]
    dur = clip_duration if clip_duration and clip_duration > 0 else float(max(n_frames - 1, 1))
    if n_frames <= 1:
        return [0.0] * n_frames
    return [dur * (i / (n_frames - 1)) for i in range(n_frames)]


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

    Iterates **all** event segments/clips per camera (not just the first 1-min clip).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    Ks: list[list[list[float]]] = []
    c2ws: list[list[list[float]]] = []
    cameras: list[dict] = []
    offsets = load_offsets(offsets_path or (work_dir / "timing_offsets.json"))
    host = host or detect_host()
    decode_devices: dict[str, str] = {}

    cam_ids: list[CameraId] = list(event.cameras) or ["front"]  # type: ignore[arg-type]
    use_ffmpeg = _have_ffmpeg()

    # Per camera: all segment MP4s in timeline order
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

    # Budget frames across all clips for each camera
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

    decoded: dict[str, tuple[list[Path], str]] = {}
    if jobs:
        if progress:
            progress("Decoding cameras/clips in parallel…", 0.1)
        decoded = decode_cameras_parallel(
            jobs,
            host=host,
            max_workers=max(1, min(parallel_cameras, len(jobs))),
            progress=progress,
        )

    idx = 0
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
            if prefix in decoded:
                frames, device = decoded[prefix]
                decode_devices[cam_id] = device
            elif use_ffmpeg:
                frames = extract_frames_ffmpeg(
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
                if decode_devices.get(cam_id) in ("videotoolbox", "ffmpeg"):
                    decode_devices[cam_id] = "software_fallback_or_placeholder"
                else:
                    decode_devices[cam_id] = decode_devices.get(cam_id, "placeholder")

            clip_dur = probe_duration_sec(mp4)
            frame_times = _frame_times_sec(mp4, len(frames), clip_dur)

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
                local_t = frame_times[fi] if fi < len(frame_times) else float(fi)
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
                image_paths.append(frame)
                Ks.append(K_scaled)
                c2ws.append(c2w)
                cameras.append(
                    {
                        "id": idx,
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
                    }
                )
                idx += 1
                global_fi += 1
        if progress:
            progress(f"Extracted {cam_id}", 0.1 + 0.1 * (ci + 1) / max(len(cam_ids), 1))
    return image_paths, Ks, c2ws, cameras, {"cameras": decode_devices, "host_chip": host.chip}
