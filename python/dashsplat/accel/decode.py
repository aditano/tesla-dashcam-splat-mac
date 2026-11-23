"""Hardware video decode helpers (VideoToolbox on Apple Silicon)."""

from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from dashsplat.accel.device import HostInfo, preferred_decode_backend

ProgressCb = Callable[[str, float], None]


def ffmpeg_extract_frames(
    mp4: Path,
    out_dir: Path,
    *,
    max_frames: int,
    prefix: str,
    host: HostInfo | None = None,
    select_timestamps_sec: list[float] | None = None,
) -> tuple[list[Path], str]:
    """Extract up to max_frames with VideoToolbox when available.

    Returns (paths, device_label). Decodes only a capped set of frames (keyframes /
    fps filter), not the full stream when possible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = preferred_decode_backend(host) if host else (
        "videotoolbox" if shutil.which("ffmpeg") else "software"
    )
    if not shutil.which("ffmpeg"):
        return [], "none"

    pattern = out_dir / f"{prefix}_%04d.jpg"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    device = "software"
    if backend == "videotoolbox":
        cmd += ["-hwaccel", "videotoolbox"]
        device = "videotoolbox"
    # Aim for ~max_frames across ~60s clip
    vf = f"fps={max(1, max_frames)}/60"
    if select_timestamps_sec:
        # Select nearest frames via fps still; timestamp select is best-effort
        vf = f"fps={max(1, min(max_frames, len(select_timestamps_sec)))}/60"
    cmd += ["-i", str(mp4), "-vf", vf, "-frames:v", str(max_frames), "-q:v", "3", str(pattern)]
    subprocess.run(cmd, capture_output=True, check=False)
    frames = sorted(out_dir.glob(f"{prefix}_*.jpg"))[:max_frames]
    if not frames and device == "videotoolbox":
        # Explicit retry without hwaccel — caller must log CPU fallback
        cmd2 = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            vf,
            "-frames:v",
            str(max_frames),
            "-q:v",
            "3",
            str(pattern),
        ]
        subprocess.run(cmd2, capture_output=True, check=False)
        frames = sorted(out_dir.glob(f"{prefix}_*.jpg"))[:max_frames]
        device = "software_fallback"
    return frames, device


def decode_cameras_parallel(
    jobs: list[tuple[Path, Path, str, int]],
    *,
    host: HostInfo | None = None,
    max_workers: int = 4,
    progress: ProgressCb | None = None,
) -> dict[str, tuple[list[Path], str]]:
    """Decode multiple cameras in parallel. jobs: (mp4, out_dir, prefix, max_frames)."""
    results: dict[str, tuple[list[Path], str]] = {}
    workers = max(1, min(max_workers, len(jobs) or 1))

    def _one(job: tuple[Path, Path, str, int]) -> tuple[str, list[Path], str]:
        mp4, out_dir, prefix, max_frames = job
        paths, device = ffmpeg_extract_frames(
            mp4, out_dir, max_frames=max_frames, prefix=prefix, host=host
        )
        return prefix, paths, device

    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, j): j[2] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            prefix, paths, device = fut.result()
            results[prefix] = (paths, device)
            done += 1
            if progress:
                progress(f"Decoded {prefix} ({device})", done / len(jobs))
    return results
