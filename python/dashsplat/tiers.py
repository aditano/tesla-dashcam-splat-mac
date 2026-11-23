"""Preview / Full quality budgets.

Timing numbers in docs are *estimates* from the 2026-11-23 research note;
this module only encodes budgets (spacing, caps, iterations), not wall times.
Fine-detail (trees) knobs live in detail_quality.py and are Full-tier defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dashsplat.detail_quality import detail_for_tier


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


def _from_detail(tier: QualityTier, *, meters: float, max_frames: int, movers: bool, eta: float) -> TierBudget:
    d = detail_for_tier(tier.value)
    return TierBudget(
        name=tier,
        meters_per_keyframe=meters,
        max_frames_per_camera=max_frames,
        target_width=d.target_width,
        mapanything_window=16,
        mapanything_stride=8,
        train_iterations=d.train_iterations + d.high_freq_extra_iters,
        max_splats=int(d.max_splats_base * d.vegetation_splat_boost),
        tile_length_m=100.0,
        tile_overlap_m=20.0,
        include_mover_masks=movers,
        eta_prior_sec=eta,
    )


PREVIEW = _from_detail(
    QualityTier.PREVIEW,
    meters=3.0,
    max_frames=60,
    movers=False,
    eta=120.0,
)

FULL = _from_detail(
    QualityTier.FULL,
    meters=1.0,
    max_frames=300,
    movers=True,
    eta=720.0,
)


def budget_for(tier: QualityTier | str) -> TierBudget:
    t = QualityTier(tier) if not isinstance(tier, QualityTier) else tier
    if t is QualityTier.PREVIEW:
        return PREVIEW
    if t is QualityTier.FULL:
        return FULL
    raise ValueError(f"unknown tier: {tier}")
