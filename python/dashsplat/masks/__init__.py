"""Mask builders (ego / sky / movers). Import submodules directly to avoid cycles."""

from __future__ import annotations

__all__ = ["build_masks_for_frames"]


def __getattr__(name: str):
    if name == "build_masks_for_frames":
        from dashsplat.masks.pipeline import build_masks_for_frames

        return build_masks_for_frames
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
