"""Extract subsampled frames from TeslaCam MP4s for MapAnything / training."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from dashsplat.cameras import HW3_CAMERA_RIG, CameraId, CameraExtrinsics, intrinsics_matrix
from dashsplat.events import DashcamEvent
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
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%04d.jpg"
    # Aim for ~max_frames across the clip; fps filter is approximate
    vf = f"fps={max(1, max_frames)}/60"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4),
        "-vf",
        vf,
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=False)
    frames = sorted(out_dir.glob(f"{prefix}_*.jpg"))[:max_frames]
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
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic color from camera prefix hash
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
    progress: ProgressCb | None = None,
) -> tuple[list[Path], list[list[list[float]]], list[list[list[float]]], list[dict]]:
    """Return image paths, intrinsics, c2w poses, and camera dicts for training."""
    work_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    Ks: list[list[list[float]]] = []
    c2ws: list[list[list[float]]] = []
    cameras: list[dict] = []

    cam_ids: list[CameraId] = list(event.cameras) or ["front"]
    use_ffmpeg = _have_ffmpeg()
    idx = 0
    for ci, cam_id in enumerate(cam_ids):
        if progress:
            progress(f"Extracting frames ({cam_id})…", 0.1 + 0.1 * ci / max(len(cam_ids), 1))
        # Prefer first segment that has this camera
        mp4: Path | None = None
        for seg in event.segments:
            if cam_id in seg.cameras:
                mp4 = seg.cameras[cam_id]
                break
        if mp4 is None:
            continue
        cam: CameraExtrinsics = HW3_CAMERA_RIG[cam_id]
        out = work_dir / "frames" / cam_id
        if use_ffmpeg:
            frames = extract_frames_ffmpeg(
                mp4, out, max_frames=max_frames_per_camera, prefix=cam_id
            )
            if not frames:
                frames = extract_placeholder_frames(
                    mp4,
                    out,
                    max_frames=max_frames_per_camera,
                    prefix=cam_id,
                    width=cam.width // 4,
                    height=cam.height // 4,
                )
        else:
            frames = extract_placeholder_frames(
                mp4,
                out,
                max_frames=max_frames_per_camera,
                prefix=cam_id,
                width=cam.width // 4,
                height=cam.height // 4,
            )

        K = intrinsics_matrix(cam)
        # Scale intrinsics if placeholder resolution differs
        from PIL import Image

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
            # Sample pose along trajectory
            if poses:
                pidx = min(int(fi / max(len(frames) - 1, 1) * (len(poses) - 1)), len(poses) - 1)
                vehicle = poses[pidx]
            else:
                from dashsplat.trajectory import Pose as P

                vehicle = P(0, (0, 0, 0), 0, 0, 0)
            yaw_off = cam.rotation_deg[1]
            cam_pose = apply_vehicle_camera_offset(vehicle, cam.position, yaw_off)
            c2w = cam_pose.c2w_matrix(z=cam.position[2])
            name = f"{cam_id}_{fi:04d}.jpg"
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
                    "time_sec": cam_pose.time_sec,
                }
            )
            idx += 1
    return image_paths, Ks, c2ws, cameras
