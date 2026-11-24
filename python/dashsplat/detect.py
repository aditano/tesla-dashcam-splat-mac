"""Encrypted-clip and missing-SEI detection with user-facing messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dashsplat.events import DashcamEvent
from dashsplat.sei import extract_sei_from_mp4


@dataclass
class ClipHealth:
    ok: bool
    has_sei: bool
    encrypted_suspected: bool
    message: str | None
    sample_count: int = 0


def looks_encrypted_mp4(path: Path, *, header_bytes: int = 256 * 1024) -> bool:
    """Heuristic: Tesla encrypted clips often have 'encv' / 'senc' / 'enca' atoms.

    Reads only the first ``header_bytes`` (default 256 KiB) — never the whole file.
    """
    try:
        with path.open("rb") as f:
            data = f.read(header_bytes)
    except OSError:
        return True
    if len(data) < 64:
        return True
    if b"encv" in data or b"senc" in data or b"enca" in data:
        return True
    return False


def check_event_telemetry(event: DashcamEvent) -> ClipHealth:
    samples = []
    encrypted = False
    for seg in event.segments:
        for cam, path in seg.cameras.items():
            if looks_encrypted_mp4(path):
                encrypted = True
            if cam == "front" or not samples:
                try:
                    samples = extract_sei_from_mp4(path)
                except Exception:
                    samples = []
            if samples:
                break
        if samples:
            break

    if encrypted and not samples:
        return ClipHealth(
            ok=False,
            has_sei=False,
            encrypted_suspected=True,
            message=(
                "These clips look encrypted. Since firmware ~2026.20 Tesla encrypts "
                "dashcam by default in many markets. Decrypt in the car or at "
                "https://dashcam.tesla.com, then try again."
            ),
            sample_count=0,
        )
    if not samples:
        return ClipHealth(
            ok=True,
            has_sei=False,
            encrypted_suspected=False,
            message=(
                "No SEI telemetry found (needs firmware 2025.44.25+). "
                "Poses will fall back to MapAnything estimates — quality will be lower."
            ),
            sample_count=0,
        )
    return ClipHealth(
        ok=True,
        has_sei=True,
        encrypted_suspected=False,
        message=None,
        sample_count=len(samples),
    )
