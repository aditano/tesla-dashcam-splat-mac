"""Preview / Full quality budgets.

Timing numbers in docs are *estimates* from the 2026-11-23 research note;
this module only encodes budgets (spacing, caps, iterations), not wall times.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QualityTier(str, Enum):
    PREVIEW = "preview"
    FULL = "full"


@dataclass(frozen=True)
class TierBudget:
    name: QualityTier
    meters_per_keyframe: float
    max_frames_per_camera: int
    target_width: int
    mapanything_window: int
    mapanything_stride: int
    train_iterations: int
    max_splats: int
    tile_length_m: float
    tile_overlap_m: float
    include_mover_masks: bool
    # ETA priors are rough first-boot defaults (seconds on Max-class); overwritten by bench.
    eta_prior_sec: float


PREVIEW = TierBudget(
    name=QualityTier.PREVIEW,
    meters_per_keyframe=3.0,
    max_frames_per_camera=60,
    target_width=640,
    mapanything_window=16,
    mapanything_stride=8,
    train_iterations=2500,
    max_splats=300_000,
    tile_length_m=100.0,
    tile_overlap_m=20.0,
    include_mover_masks=False,
    eta_prior_sec=120.0,  # estimate ~2 min Preview on Max-class
)

FULL = TierBudget(
    name=QualityTier.FULL,
    meters_per_keyframe=1.0,
    max_frames_per_camera=300,
    target_width=1280,
    mapanything_window=16,
    mapanything_stride=8,
    train_iterations=7000,
    max_splats=1_000_000,
    tile_length_m=100.0,
    tile_overlap_m=20.0,
    include_mover_masks=True,
    eta_prior_sec=720.0,  # estimate ~10–15 min Full on Max-class
)


def budget_for(tier: QualityTier | str) -> TierBudget:
    t = QualityTier(tier) if not isinstance(tier, QualityTier) else tier
    if t is QualityTier.PREVIEW:
        return PREVIEW
    if t is QualityTier.FULL:
        return FULL
    raise ValueError(f"unknown tier: {tier}")
