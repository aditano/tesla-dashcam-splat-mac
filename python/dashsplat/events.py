"""TeslaCam event grouping — ported from src/clips/events.ts + groupClips.ts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashsplat.cameras import CameraId

IGNORE_NAMES = {
    "thumb.png",
    "thumb.jpg",
    "thumb.jpeg",
    ".ds_store",
    "desktop.ini",
}

CAMERA_ALIASES: dict[str, CameraId] = {
    "front": "front",
    "back": "back",
    "rear": "back",
    "left_repeater": "left_repeater",
    "right_repeater": "right_repeater",
    "left_pillar": "left_pillar",
    "right_pillar": "right_pillar",
    "left": "left_repeater",
    "right": "right_repeater",
}

NAME_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2})"
    r"[_-](?P<cam>front|back|rear|left_repeater|right_repeater|left_pillar|right_pillar|left|right)\.mp4$",
    re.IGNORECASE,
)
EVENT_FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")
TS_KEY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$")


def should_ignore_file(name: str) -> bool:
    base = Path(name).name.lower()
    if base in IGNORE_NAMES:
        return True
    if base.startswith("."):
        return True
    if base.endswith(".tmp"):
        return True
    return False


def parse_clip_filename(name: str) -> dict[str, Any] | None:
    base = Path(name).name
    m = NAME_RE.search(base)
    if not m:
        return None
    cam = CAMERA_ALIASES.get(m.group("cam").lower())
    if not cam:
        return None
    raw_ts = m.group("ts").replace("_", "-")
    parts = [int(p) for p in raw_ts.split("-")]
    if len(parts) < 6:
        return None
    wall = datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], tzinfo=timezone.utc)
    timestamp_key = (
        f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}_"
        f"{parts[3]:02d}-{parts[4]:02d}-{parts[5]:02d}"
    )
    return {"timestamp_key": timestamp_key, "camera": cam, "wall_clock": wall}


def parse_timestamp_key(key: str) -> datetime | None:
    m = TS_KEY_RE.match(key)
    if not m:
        return None
    return datetime(
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
        int(m.group(6)),
        tzinfo=timezone.utc,
    )


def format_event_label(d: datetime) -> str:
    return d.strftime("%Y-%m-%d %H:%M:%S")


def parse_event_json(text: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    lat = raw.get("est_lat") if isinstance(raw.get("est_lat"), (int, float)) else None
    if lat is None and isinstance(raw.get("latitude"), (int, float)):
        lat = raw["latitude"]
    lon = raw.get("est_lon") if isinstance(raw.get("est_lon"), (int, float)) else None
    if lon is None and isinstance(raw.get("longitude"), (int, float)):
        lon = raw["longitude"]
    return {
        "timestamp": raw.get("timestamp") if isinstance(raw.get("timestamp"), str) else None,
        "city": raw.get("city") if isinstance(raw.get("city"), str) else None,
        "reason": raw.get("reason") if isinstance(raw.get("reason"), str) else None,
        "est_lat": raw.get("est_lat") if isinstance(raw.get("est_lat"), (int, float)) else None,
        "est_lon": raw.get("est_lon") if isinstance(raw.get("est_lon"), (int, float)) else None,
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "raw": raw,
    }


def _event_folder_from_path(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    for i in range(len(parts) - 2, -1, -1):
        if EVENT_FOLDER_RE.match(parts[i]):
            return parts[i]
    return None


def _relative_root_hint(path: str) -> str | None:
    for p in path.replace("\\", "/").split("/"):
        if re.match(r"^(SavedClips|SentryClips|RecentClips|TeslaCam)$", p, re.I):
            return p
    return None


@dataclass
class EventSegment:
    timestamp_key: str
    wall_clock: datetime | None
    time_offset_sec: float
    cameras: dict[CameraId, Path] = field(default_factory=dict)


@dataclass
class DashcamEvent:
    id: str
    label: str
    start: datetime
    duration_sec: float
    cameras: list[CameraId]
    location_label: str | None
    latitude: float | None
    longitude: float | None
    source_folder: str | None
    event_json: dict[str, Any] | None
    segments: list[EventSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "start": self.start.isoformat(),
            "duration_sec": self.duration_sec,
            "cameras": list(self.cameras),
            "location_label": self.location_label,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source_folder": self.source_folder,
            "event_json": {k: v for k, v in (self.event_json or {}).items() if k != "raw"}
            if self.event_json
            else None,
            "segments": [
                {
                    "timestamp_key": s.timestamp_key,
                    "time_offset_sec": s.time_offset_sec,
                    "cameras": {k: str(v) for k, v in s.cameras.items()},
                }
                for s in self.segments
            ],
        }


def discover_files(root: Path) -> list[tuple[Path, str]]:
    """Return (absolute path, relative posix path) for ingestable files under root."""
    root = root.resolve()
    out: list[tuple[Path, str]] = []
    if root.is_file():
        if not should_ignore_file(root.name):
            out.append((root, root.name))
        return out
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_ignore_file(path.name):
            continue
        rel = path.relative_to(root).as_posix()
        out.append((path, rel))
    return out


def group_into_events(
    files: list[tuple[Path, str]],
    event_json_by_folder: dict[str, dict[str, Any]] | None = None,
) -> list[DashcamEvent]:
    """Group MP4 paths into Tesla dashcam events. Never merges distinct event folders."""
    event_json_by_folder = event_json_by_folder or {}
    clips: list[dict[str, Any]] = []
    for path, rel in files:
        if path.suffix.lower() != ".mp4":
            continue
        parsed = parse_clip_filename(path.name)
        if not parsed:
            continue
        clips.append(
            {
                "path": path,
                "relative_path": rel,
                "camera": parsed["camera"],
                "timestamp_key": parsed["timestamp_key"],
                "wall_clock": parsed["wall_clock"],
                "event_folder": _event_folder_from_path(rel),
            }
        )

    buckets: dict[str, dict[str, Any]] = {}
    for c in clips:
        source = _relative_root_hint(c["relative_path"])
        # Never merge same-named SavedClips and SentryClips folders.
        if c["event_folder"]:
            key = f"{source}/{c['event_folder']}" if source else c["event_folder"]
        else:
            key = c["timestamp_key"]
        b = buckets.get(key)
        if not b:
            b = {
                "key": key,
                "folder": c["event_folder"],
                "clips": [],
                "source_folder": source,
            }
            buckets[key] = b
        b["clips"].append(c)

    # RecentClips-style: merge adjacent folderless timestamps within 90s of the
    # *end* of the current run (not the first clip), so long drives stay one event.
    folderless = [b for b in buckets.values() if not b["folder"]]
    if len(folderless) > 1:
        folderless.sort(key=lambda b: b["key"])
        current = folderless[0]
        for next_b in folderless[1:]:
            cur_times = [
                parse_timestamp_key(c["timestamp_key"]) for c in current["clips"]
            ]
            next_times = [
                parse_timestamp_key(c["timestamp_key"]) for c in next_b["clips"]
            ]
            cur_times_f = [t for t in cur_times if t is not None]
            next_times_f = [t for t in next_times if t is not None]
            if not cur_times_f or not next_times_f:
                current = next_b
                continue
            t_end = max(cur_times_f)
            t_start = min(next_times_f)
            gap = (t_start - t_end).total_seconds()
            if 0 <= gap <= 90:
                current["clips"].extend(next_b["clips"])
                buckets.pop(next_b["key"], None)
            else:
                current = next_b

    events: list[DashcamEvent] = []
    for b in buckets.values():
        by_ts: dict[str, EventSegment] = {}
        for c in b["clips"]:
            ts = c["timestamp_key"]
            seg = by_ts.get(ts)
            if not seg:
                seg = EventSegment(
                    timestamp_key=ts,
                    wall_clock=c["wall_clock"],
                    time_offset_sec=0.0,
                    cameras={},
                )
                by_ts[ts] = seg
            seg.cameras[c["camera"]] = c["path"]
        segments = sorted(by_ts.values(), key=lambda s: s.timestamp_key)
        if not segments:
            continue
        start = parse_timestamp_key(segments[0].timestamp_key) or segments[0].wall_clock or datetime(
            1970, 1, 1, tzinfo=timezone.utc
        )
        for seg in segments:
            t = parse_timestamp_key(seg.timestamp_key)
            seg.time_offset_sec = (t - start).total_seconds() if t else 0.0
        cam_set: set[CameraId] = set()
        for seg in segments:
            cam_set.update(seg.cameras.keys())
        duration_sec = max(60.0, (segments[-1].time_offset_sec or 0.0) + 60.0)
        # event.json keyed by folder name; also try composite key
        ej = event_json_by_folder.get(b["folder"] or "") or event_json_by_folder.get(b["key"])
        lat = (ej or {}).get("latitude") or (ej or {}).get("est_lat")
        lon = (ej or {}).get("longitude") or (ej or {}).get("est_lon")
        location_label = (ej or {}).get("city")
        if not location_label and lat is not None and lon is not None:
            location_label = f"{float(lat):.4f}, {float(lon):.4f}"
        # Prefer stable folder id; include source when present to avoid collisions
        event_id = b["folder"] or b["key"]
        if b.get("source_folder") and b["folder"]:
            event_id = f"{b['source_folder']}/{b['folder']}"
        events.append(
            DashcamEvent(
                id=event_id,
                label=format_event_label(start),
                start=start,
                duration_sec=duration_sec,
                cameras=sorted(cam_set),
                location_label=location_label,
                latitude=float(lat) if lat is not None else None,
                longitude=float(lon) if lon is not None else None,
                source_folder=b.get("source_folder"),
                event_json=ej,
                segments=segments,
            )
        )

    events.sort(key=lambda e: e.start)
    return events


def scan_teslacam(root: Path) -> list[DashcamEvent]:
    """Scan a TeslaCam root / SavedClips / event folder / loose MP4s."""
    files = discover_files(root)
    json_map: dict[str, dict[str, Any]] = {}
    for path, rel in files:
        if path.name.lower() != "event.json":
            continue
        try:
            parsed = parse_event_json(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not parsed:
            continue
        parts = rel.replace("\\", "/").split("/")
        folder_name = parts[-2] if len(parts) >= 2 else parts[0]
        json_map[folder_name] = parsed
        ef = _event_folder_from_path(rel + "/x")
        if ef:
            json_map[ef] = parsed
    return group_into_events(files, json_map)
