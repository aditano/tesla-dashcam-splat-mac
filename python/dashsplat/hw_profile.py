"""HW3 / HW4 TeslaCam hardware profiles.

Citations (do not invent numbers):
- Layout HW3 = 4 cams (front, back, left_repeater, right_repeater); HW4 adds
  left_pillar, right_pillar:
  https://www.clipsalvage.com/articles/how-tesla-stores-video.html
  https://github.com/geeksbaek/tesla-dashcam-viewer/
- Dashcam encode size/fps commonly reported as 1280×960 @ ~36 fps on HW3:
  https://github.com/conscientsystems/teslacam (README: "Clips are 1280x960 at 36 fps")
  https://github.com/kandotrun/teslacam-viewer (README: "1280 × 960 at 36 fps, like HW3 cars")
  https://www.nasrtech.dev/tools/video-repair/tesla-dashcam-video-corrupted/ (1280×960 H.264, about 36 fps)
- SEI on firmware 2025.44.25+ HW3+:
  https://github.com/teslamotors/dashcam
- Extrinsics/intrinsics in cameras.py are approximations (labeled there),
  informed by community rigs and R44VC0RP/dashsplat calibration approach
  (HW4 Model 3; HW3 four-cam untested upstream):
  https://github.com/R44VC0RP/dashsplat
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from dashsplat.cameras import ALL_CAMERA_IDS, CameraId

HW3_CAMERAS: tuple[CameraId, ...] = (
    "front",
    "back",
    "left_repeater",
    "right_repeater",
)
HW4_CAMERAS: tuple[CameraId, ...] = HW3_CAMERAS + ("left_pillar", "right_pillar")

# Cited nominal dashcam encode (see module docstring). Prefer probing real clips.
HW3_NOMINAL_WIDTH = 1280
HW3_NOMINAL_HEIGHT = 960
HW3_NOMINAL_FPS = 36.0


class HWGeneration(str, Enum):
    HW3 = "hw3"
    HW4 = "hw4"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HWProfile:
    generation: HWGeneration
    cameras: tuple[CameraId, ...]
    width: int
    height: int
    fps_nominal: float
    sources: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "generation": self.generation.value,
            "cameras": list(self.cameras),
            "width": self.width,
            "height": self.height,
            "fps_nominal": self.fps_nominal,
            "sources": list(self.sources),
        }


_SOURCES = (
    "https://github.com/conscientsystems/teslacam",
    "https://www.clipsalvage.com/articles/how-tesla-stores-video.html",
    "https://github.com/teslamotors/dashcam",
    "https://github.com/R44VC0RP/dashsplat",
)


def resolve_clip_resolution(
    width: int | None,
    height: int | None,
    fps: float | None,
    *,
    generation: HWGeneration = HWGeneration.HW3,
) -> tuple[int, int, float]:
    """Use probed metadata when present; else cited HW3 nominals."""
    w = int(width) if width and width > 0 else HW3_NOMINAL_WIDTH
    h = int(height) if height and height > 0 else HW3_NOMINAL_HEIGHT
    f = float(fps) if fps and fps > 0 else HW3_NOMINAL_FPS
    _ = generation  # HW4 dashcam encode size varies; prefer probe over inventing
    return w, h, f


def profile_for_cameras(
    cameras: Iterable[str],
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> HWProfile:
    cams = {c for c in cameras if c in ALL_CAMERA_IDS}
    has_pillar = "left_pillar" in cams or "right_pillar" in cams
    if has_pillar:
        gen = HWGeneration.HW4
        ordered = tuple(c for c in HW4_CAMERAS if c in cams) or HW4_CAMERAS
    elif cams & set(HW3_CAMERAS):
        gen = HWGeneration.HW3
        ordered = tuple(c for c in HW3_CAMERAS if c in cams) or HW3_CAMERAS
    else:
        gen = HWGeneration.UNKNOWN
        ordered = tuple(sorted(cams))  # type: ignore[arg-type]
    w, h, f = resolve_clip_resolution(width, height, fps, generation=gen)
    return HWProfile(
        generation=gen,
        cameras=ordered,  # type: ignore[arg-type]
        width=w,
        height=h,
        fps_nominal=f,
        sources=_SOURCES,
    )


def collect_camera_names_from_folder(root: Path) -> set[str]:
    """Scan TeslaCam tree for camera suffixes in filenames."""
    from dashsplat.events import parse_clip_filename

    found: set[str] = set()
    if not root.exists():
        return found
    for path in root.rglob("*.mp4"):
        parsed = parse_clip_filename(path.name)
        if parsed:
            found.add(parsed["camera"])
    return found


def detect_hw_profile(root: Path, *, width: int | None = None, height: int | None = None) -> HWProfile:
    cams = collect_camera_names_from_folder(root)
    # Prefer probing a front clip for resolution when available
    probed_w, probed_h, probed_fps = width, height, None
    if probed_w is None or probed_h is None:
        for path in root.rglob("*-front.mp4"):
            meta = probe_mp4_meta(path)
            if meta:
                probed_w = probed_w or meta.get("width")
                probed_h = probed_h or meta.get("height")
                probed_fps = meta.get("fps")
                break
    return profile_for_cameras(cams, width=probed_w, height=probed_h, fps=probed_fps)


def probe_mp4_meta(path: Path) -> dict | None:
    """Best-effort ffprobe; returns None when unavailable (CI stubs)."""
    import json
    import shutil
    import subprocess

    if not path.exists() or path.stat().st_size < 64:
        return None
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        s = streams[0]
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        fps = 0.0
        for key in ("avg_frame_rate", "r_frame_rate"):
            rate = s.get(key) or ""
            if isinstance(rate, str) and "/" in rate:
                a, b = rate.split("/", 1)
                if float(b) != 0:
                    fps = float(a) / float(b)
                    break
        if w <= 0 or h <= 0:
            return None
        return {"width": w, "height": h, "fps": fps or None}
    except Exception:
        return None
