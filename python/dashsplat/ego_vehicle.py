"""Ego vehicle 3D proxy for the splat viewer / exports.

The ego body is masked out of training; a low-poly glTF fills that gap and
follows the SEI-driven path (position + heading). Default mesh is a generic
MIT-licensed sedan/crossover placeholder. A later version can swap in a
Model 3/Y/S/X + paint mesh via ``ego_vehicle.manifest.json`` without changing
this placement API — no automatic paint/model identification in v1.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Bundled default relative to repo viewer/ and app Resources/
DEFAULT_MODEL_ID = "generic_sedan_crossover"
DEFAULT_GLB_NAME = "sedan_crossover.glb"
DEFAULT_LICENSE = "MIT"
DEFAULT_LICENSE_URL = "viewer/assets/ego/LICENSE.txt"
DEFAULT_WHEELBASE_M = 2.7
DEFAULT_LENGTH_M = 4.4
DEFAULT_WIDTH_M = 1.85
DEFAULT_HEIGHT_M = 1.45


def bundled_assets_dir() -> Path:
    """Repo path to bundled ego GLB (also copied into Apps/DashSplat/Resources)."""
    return Path(__file__).resolve().parents[2] / "viewer" / "assets" / "ego"


def default_manifest() -> dict[str, Any]:
    return {
        "model_id": DEFAULT_MODEL_ID,
        "display_name": "Generic sedan / crossover (placeholder)",
        "glb": DEFAULT_GLB_NAME,
        "license": DEFAULT_LICENSE,
        "license_file": "LICENSE.txt",
        "source": "Original DashSplat procedural low-poly mesh",
        "length_m": DEFAULT_LENGTH_M,
        "width_m": DEFAULT_WIDTH_M,
        "height_m": DEFAULT_HEIGHT_M,
        "wheelbase_m": DEFAULT_WHEELBASE_M,
        # Local mesh: +Z forward, +Y up, origin between wheels on ground plane
        "forward_axis": "+Z",
        "up_axis": "+Y",
        "ground_offset_m": 0.0,
        # Slot for a later matching Model 3/Y/S/X + paint mesh:
        "swap_hint": {
            "model_variant": None,  # e.g. "model_y_juniper"
            "paint_name": None,
            "paint_hex": None,
        },
    }


@dataclass(frozen=True)
class EgoPose:
    """Interpolated ego pose in ENU meters (east, north, up) + yaw."""

    time_sec: float
    east: float
    north: float
    up: float
    yaw_rad: float
    heading_deg: float

    def position(self) -> tuple[float, float, float]:
        return self.east, self.north, self.up

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_sec": self.time_sec,
            "position": [self.east, self.north, self.up],
            "yaw_rad": self.yaw_rad,
            "heading_deg": self.heading_deg,
            # Y-up OpenGL-ish (x=east, y=up, z=-north) for viewers that remap
            "position_y_up": [self.east, self.up, -self.north],
            "yaw_y_up_rad": self.yaw_rad,
        }


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Shortest-path lerp for radians."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t


def interpolate_ego_pose(
    poses: Sequence[dict[str, Any] | Any],
    time_sec: float,
) -> EgoPose | None:
    """Linear position + shortest-arc yaw interpolation along a SEI path.

    ``poses`` entries are either ``trajectory.Pose``-like objects or dicts with
    ``time_sec``, ``position`` [e,n,u], ``yaw_rad`` / ``heading_deg``.
    """
    if not poses:
        return None

    def _unpack(p: Any) -> tuple[float, float, float, float, float, float]:
        if isinstance(p, dict):
            t = float(p["time_sec"])
            pos = p["position"]
            e, n, u = float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0
            yaw = float(p.get("yaw_rad", math.radians(90.0 - float(p.get("heading_deg", 0.0)))))
            hdg = float(p.get("heading_deg", 90.0 - math.degrees(yaw)))
            return t, e, n, u, yaw, hdg
        t = float(p.time_sec)
        e, n, u = p.position
        return t, float(e), float(n), float(u), float(p.yaw_rad), float(p.heading_deg)

    samples = [_unpack(p) for p in poses]
    samples.sort(key=lambda s: s[0])
    t0 = samples[0][0]
    t1 = samples[-1][0]
    if time_sec <= t0:
        _, e, n, u, yaw, hdg = samples[0]
        return EgoPose(t0, e, n, u, yaw, hdg)
    if time_sec >= t1:
        _, e, n, u, yaw, hdg = samples[-1]
        return EgoPose(t1, e, n, u, yaw, hdg)
    for i in range(len(samples) - 1):
        ta, ea, na, ua, ya, _ = samples[i]
        tb, eb, nb, ub, yb, _ = samples[i + 1]
        if ta <= time_sec <= tb:
            u_t = 0.0 if tb <= ta else (time_sec - ta) / (tb - ta)
            yaw = _lerp_angle(ya, yb, u_t)
            return EgoPose(
                time_sec=time_sec,
                east=_lerp(ea, eb, u_t),
                north=_lerp(na, nb, u_t),
                up=_lerp(ua, ub, u_t),
                yaw_rad=yaw,
                heading_deg=90.0 - math.degrees(yaw),
            )
    _, e, n, u, yaw, hdg = samples[-1]
    return EgoPose(time_sec, e, n, u, yaw, hdg)


def sample_path(
    poses: Sequence[dict[str, Any] | Any],
    *,
    dt: float = 0.1,
) -> list[EgoPose]:
    """Dense samples for the viewer scrubber / animation."""
    if not poses:
        return []
    times = []
    for p in poses:
        if isinstance(p, dict):
            times.append(float(p["time_sec"]))
        else:
            times.append(float(p.time_sec))
    t0, t1 = min(times), max(times)
    if t1 <= t0:
        one = interpolate_ego_pose(poses, t0)
        return [one] if one else []
    out: list[EgoPose] = []
    t = t0
    while t <= t1 + 1e-9:
        pose = interpolate_ego_pose(poses, t)
        if pose:
            out.append(pose)
        t += dt
    if out and abs(out[-1].time_sec - t1) > 1e-6:
        last = interpolate_ego_pose(poses, t1)
        if last:
            out.append(last)
    return out


def write_ego_path_json(
    path: Path,
    poses: Sequence[dict[str, Any] | Any],
    *,
    dt: float = 0.1,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write viewer-consumable ego path + manifest reference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = sample_path(poses, dt=dt)
    payload = {
        "coordinate_frame": "enu",
        "axes": {"x": "east", "y": "north", "z": "up"},
        "manifest": manifest or default_manifest(),
        "t0": samples[0].time_sec if samples else 0.0,
        "t1": samples[-1].time_sec if samples else 0.0,
        "samples": [s.to_dict() for s in samples],
    }
    path.write_text(json.dumps(payload, indent=2))
    return payload


def install_ego_assets(out_dir: Path, *, model_id: str | None = None) -> dict[str, Any]:
    """Copy bundled GLB + license + manifest into a job output folder.

    ``model_id`` reserved for future Model 3/Y/S/X mesh packs; ignored for now
    except recorded in the manifest swap_hint slot.
    """
    src = bundled_assets_dir()
    dest = out_dir / "ego"
    dest.mkdir(parents=True, exist_ok=True)
    glb_src = src / DEFAULT_GLB_NAME
    lic_src = src / "LICENSE.txt"
    if glb_src.is_file():
        shutil.copy2(glb_src, dest / DEFAULT_GLB_NAME)
    if lic_src.is_file():
        shutil.copy2(lic_src, dest / "LICENSE.txt")
    manifest = default_manifest()
    if model_id:
        manifest["swap_hint"]["model_variant"] = model_id
    (dest / "ego_vehicle.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_ego_sidecar(
    out_dir: Path,
    poses: Sequence[dict[str, Any] | Any],
    *,
    dt: float = 0.1,
) -> dict[str, Any]:
    """Install assets + path JSON for the viewer and as a separate export object."""
    manifest = install_ego_assets(out_dir)
    path_payload = write_ego_path_json(
        out_dir / "ego" / "ego_path.json",
        poses,
        dt=dt,
        manifest=manifest,
    )
    # Also mirror at job root for easy discovery
    shutil.copy2(out_dir / "ego" / "ego_path.json", out_dir / "ego_path.json")
    return {
        "ego_dir": str(out_dir / "ego"),
        "glb": str(out_dir / "ego" / DEFAULT_GLB_NAME),
        "path": str(out_dir / "ego" / "ego_path.json"),
        "manifest": manifest,
        "n_samples": len(path_payload["samples"]),
        "t0": path_payload["t0"],
        "t1": path_payload["t1"],
    }
