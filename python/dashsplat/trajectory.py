"""ENU trajectory from SEI GPS — ported from src/trajectory/*."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dashsplat.sei import SeiSample

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


@dataclass
class EnuOrigin:
    lat_deg: float
    lon_deg: float
    meters_per_deg_lat: float
    meters_per_deg_lon: float


@dataclass
class Pose:
    time_sec: float
    position: tuple[float, float, float]  # east, north, up
    yaw_rad: float
    heading_deg: float
    speed_mps: float
    source: str = "sei"

    def c2w_matrix(self, z: float = 1.2) -> list[list[float]]:
        """OpenCV-ish camera-to-world (right, down, forward columns).

        Right-handed ENU: facing yaw, forward=[cos,sin,0], down=[0,0,-1],
        right = down × forward = [sin, -cos, 0]. (Previously [-sin, cos, 0]
        mirrored left/right.)
        """
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        right = [s, -c, 0.0]
        down = [0.0, 0.0, -1.0]
        forward = [c, s, 0.0]
        east, north, _ = self.position
        return [
            [right[0], down[0], forward[0], east],
            [right[1], down[1], forward[1], north],
            [right[2], down[2], forward[2], z],
            [0.0, 0.0, 0.0, 1.0],
        ]


def create_enu_origin(lat_deg: float, lon_deg: float) -> EnuOrigin:
    lat0 = math.radians(lat_deg)
    sin_lat = math.sin(lat0)
    m_lat = (math.pi / 180) * (WGS84_A * (1 - WGS84_E2)) / (1 - WGS84_E2 * sin_lat * sin_lat) ** 1.5
    m_lon = (math.pi / 180) * (WGS84_A * math.cos(lat0)) / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    return EnuOrigin(lat_deg, lon_deg, m_lat, m_lon)


def llh_to_enu(lat_deg: float, lon_deg: float, origin: EnuOrigin) -> tuple[float, float, float]:
    east = (lon_deg - origin.lon_deg) * origin.meters_per_deg_lon
    north = (lat_deg - origin.lat_deg) * origin.meters_per_deg_lat
    return east, north, 0.0


def heading_deg_to_enu_yaw(heading_deg: float) -> float:
    return math.radians(90.0 - heading_deg)


def yaw_to_forward(yaw_rad: float) -> tuple[float, float, float]:
    return math.cos(yaw_rad), math.sin(yaw_rad), 0.0


def smooth_series(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values) - 1, i + half)
        chunk = values[lo : hi + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def unwrap_heading_deg(headings: list[float]) -> list[float]:
    if not headings:
        return []
    out = [headings[0]]
    for h in headings[1:]:
        prev = out[-1]
        while h - prev > 180:
            h -= 360
        while h - prev < -180:
            h += 360
        out.append(h)
    return out


@dataclass
class TrajectoryResult:
    poses: list[Pose]
    source: str
    warning: str | None = None
    origin_lat_lon: tuple[float, float] | None = None


def build_trajectory_from_sei(
    samples: list[SeiSample],
    *,
    fallback_duration_sec: float = 4.0,
    fallback_speed_mps: float = 8.0,
    fallback_heading_deg: float = 0.0,
) -> TrajectoryResult:
    # fallback_* kept for API compat but MUST NOT invent a pose prior
    _ = fallback_duration_sec, fallback_speed_mps, fallback_heading_deg
    usable = [
        s
        for s in samples
        if math.isfinite(s.latitude_deg)
        and math.isfinite(s.longitude_deg)
        and abs(s.latitude_deg) > 0.01
        and abs(s.longitude_deg) > 0.01
    ]
    if not usable:
        return build_unknown_poses_fallback()

    origin = create_enu_origin(usable[0].latitude_deg, usable[0].longitude_deg)
    raw_headings = unwrap_heading_deg([s.heading_deg or 0.0 for s in usable])
    smooth_headings = smooth_series(raw_headings, 7)
    smooth_speeds = smooth_series(
        [s.vehicle_speed_mps if math.isfinite(s.vehicle_speed_mps) else 0.0 for s in usable],
        5,
    )
    gps_pos = [llh_to_enu(s.latitude_deg, s.longitude_deg, origin) for s in usable]
    # Keep event/clip time — do NOT re-zero at first GPS fix (frames use the same clock)
    times = [float(s.time_sec) if s.time_sec is not None else i / 36.0 for i, s in enumerate(usable)]

    integrated: list[tuple[float, float, float]] = []
    x, y = gps_pos[0][0], gps_pos[0][1]
    integrated.append((x, y, 0.0))
    for i in range(1, len(usable)):
        dt = max(1e-3, times[i] - times[i - 1])
        yaw = heading_deg_to_enu_yaw(smooth_headings[i - 1])
        fx, fy, _ = yaw_to_forward(yaw)
        x += fx * smooth_speeds[i - 1] * dt
        y += fy * smooth_speeds[i - 1] * dt
        alpha = 0.35
        x = (1 - alpha) * x + alpha * gps_pos[i][0]
        y = (1 - alpha) * y + alpha * gps_pos[i][1]
        integrated.append((x, y, 0.0))

    sx = smooth_series([p[0] for p in integrated], 5)
    sy = smooth_series([p[1] for p in integrated], 5)
    poses = [
        Pose(
            time_sec=times[i],
            position=(sx[i], sy[i], 0.0),
            yaw_rad=heading_deg_to_enu_yaw(smooth_headings[i]),
            heading_deg=smooth_headings[i],
            speed_mps=smooth_speeds[i],
            source="sei",
        )
        for i in range(len(usable))
    ]
    return TrajectoryResult(
        poses=poses,
        source="sei",
        origin_lat_lon=(origin.lat_deg, origin.lon_deg),
    )


def build_unknown_poses_fallback() -> TrajectoryResult:
    """No SEI GPS — do not invent a constant-velocity prior for geometry."""
    return TrajectoryResult(
        poses=[],
        source="unknown",
        warning=(
            "No SEI GPS telemetry found. Poses marked unknown (no invented prior). "
            "Requires firmware 2025.44.25+ with SEI for accurate poses."
        ),
    )


def build_constant_velocity_fallback(
    duration_sec: float,
    speed_mps: float,
    heading_deg: float,
    fps: float = 12.0,
) -> TrajectoryResult:
    """Deprecated: do not use as a pose prior. Delegates to unknown poses."""
    _ = duration_sec, speed_mps, heading_deg, fps
    return build_unknown_poses_fallback()


def apply_vehicle_camera_offset(
    pose: Pose,
    cam_position: tuple[float, float, float],
    cam_yaw_offset_deg: float,
) -> Pose:
    """Transform vehicle pose by body-frame camera offset."""
    cy, sy = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    # body +X forward, +Y left → ENU
    bx, by, bz = cam_position
    east = pose.position[0] + cy * bx - sy * by
    north = pose.position[1] + sy * bx + cy * by
    up = pose.position[2] + bz
    yaw = pose.yaw_rad + math.radians(cam_yaw_offset_deg)
    # Compass heading increases clockwise; ENU yaw increases CCW.
    # heading ≈ 90° − degrees(yaw), so Δheading = −Δyaw_offset.
    return Pose(
        time_sec=pose.time_sec,
        position=(east, north, up),
        yaw_rad=yaw,
        heading_deg=pose.heading_deg - cam_yaw_offset_deg,
        speed_mps=pose.speed_mps,
        source=pose.source,
    )
