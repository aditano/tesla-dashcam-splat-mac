"""Auto-tune splat / MapAnything budgets from detected RAM and chip class."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from dashsplat.accel.device import HostInfo
from dashsplat.detail_quality import detail_for_tier, effective_iterations, effective_max_splats
from dashsplat.tiers import QualityTier, budget_for


@dataclass(frozen=True)
class TunedBudget:
    tier: str
    max_splats: int
    train_iterations: int
    mapanything_window: int
    mapanything_stride: int
    decode_parallel_cameras: int
    mask_batch_size: int
    tile_length_m: float
    ram_gb: float
    chip: str
    detail_densify: str = "mcmc"
    max_scale_m: float = 0.35

    def to_dict(self) -> dict:
        return asdict(self)


def _chip_class(chip: str) -> str:
    c = chip.lower()
    if "ultra" in c:
        return "ultra"
    if "max" in c:
        return "max"
    if "pro" in c:
        return "pro"
    if "m1" in c or "m2" in c or "m3" in c or "m4" in c or "m5" in c:
        return "base"
    return "unknown"


def tune_for_host(host: HostInfo, tier: QualityTier | str = "preview") -> TunedBudget:
    base = budget_for(tier)
    ram_gb = max(host.ram_gb, 8.0)
    klass = _chip_class(host.chip)

    # Scale splat cap and window with unified memory (rough, measured later via bench).
    mem_scale = min(2.5, max(0.5, ram_gb / 36.0))
    if klass == "ultra":
        mem_scale = max(mem_scale, 1.8)
    elif klass == "max":
        mem_scale = max(mem_scale, 1.4)
    elif klass == "base":
        mem_scale = min(mem_scale, 1.0)

    detail = detail_for_tier(base.name.value)
    max_splats = effective_max_splats(detail, ram_gb)
    # Also honor legacy mem_scale floor from base
    max_splats = max(max_splats, int(base.max_splats * mem_scale * 0.5))
    window = int(base.mapanything_window)
    if ram_gb >= 64:
        window = min(32, window * 2)
    elif ram_gb < 24:
        window = max(8, window // 2)

    # Parallel camera decode: scale with performance cores and RAM.
    parallel = max(1, min(4, host.performance_cores // 2 or 1))
    if ram_gb >= 32:
        parallel = max(parallel, 4)
    if ram_gb >= 64:
        parallel = max(parallel, 6)
    parallel = min(parallel, 6)

    mask_batch = 8 if ram_gb >= 32 else 4
    if ram_gb >= 64:
        mask_batch = 16

    return TunedBudget(
        tier=base.name.value if hasattr(base.name, "value") else str(base.name),
        max_splats=max_splats,
        train_iterations=effective_iterations(detail),
        mapanything_window=window,
        mapanything_stride=base.mapanything_stride,
        decode_parallel_cameras=parallel,
        mask_batch_size=mask_batch,
        tile_length_m=base.tile_length_m,
        ram_gb=round(ram_gb, 2),
        chip=host.chip,
        detail_densify=detail.densify_mode,
        max_scale_m=detail.max_scale_m,
    )
