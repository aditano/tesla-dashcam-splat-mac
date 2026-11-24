"""Hardware video decode helpers (VideoToolbox on Apple Silicon)."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from dashsplat.accel.device import HostInfo, preferred_decode_backend

ProgressCb = Callable[[str, float], None]


def probe_duration_sec(mp4: Path) -> float | None:
    """Return container duration via ffprobe, or None if unavailable."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
        dur = float(data.get("format", {}).get("duration"))
        return dur if dur > 0 else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def probe_frame_pts_sec(mp4: Path, max_frames: int | None = None) -> list[float]:
    """Return presentation timestamps (seconds) for video packets via ffprobe."""
    if not shutil.which("ffprobe"):
        return []
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        str(mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    pts: list[float] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip().rstrip(",")
        if not line or line == "N/A":
            continue
        try:
            pts.append(float(line.split(",")[0]))
        except ValueError:
            continue
        if max_frames is not None and len(pts) >= max_frames:
            break
    if pts:
        t0 = pts[0]
        pts = [p - t0 for p in pts]
    return pts


def ffmpeg_extract_frames(
    mp4: Path,
    out_dir: Path,
    *,
    max_frames: int,
    prefix: str,
    host: HostInfo | None = None,
    select_timestamps_sec: list[float] | None = None,
    duration_sec: float | None = None,
) -> tuple[list[Path], str]:
    """Extract up to max_frames with VideoToolbox when available.

    Returns (paths, device_label). Frame spacing uses the real clip duration from
    ffprobe when available (not a hard-coded 60 s assumption).
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

    dur = duration_sec if duration_sec and duration_sec > 0 else probe_duration_sec(mp4)
    if dur is None or dur <= 0:
        dur = 60.0  # last-resort only when probe fails
    # Spread max_frames across the real clip duration
    vf = f"fps={max(1, max_frames)}/{max(dur, 0.1)}"
    if select_timestamps_sec:
        vf = f"fps={max(1, min(max_frames, len(select_timestamps_sec)))}/{max(dur, 0.1)}"
    cmd += ["-i", str(mp4), "-vf", vf, "-frames:v", str(max_frames), "-q:v", "3", str(pattern)]
    subprocess.run(cmd, capture_output=True, check=False)
    frames = sorted(out_dir.glob(f"{prefix}_*.jpg"))[:max_frames]
    if not frames and device == "videotoolbox":
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
