"""3DGS training entry — Brush (default) / msplat / seeded fallback.

OpenSplat (AGPL-3.0) was removed. See docs/ARCHITECTURE.md and the 2026-11-23 research note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from dashsplat.export import write_cameras_json
from dashsplat.trainers.base import TrainerConfig
from dashsplat.trainers.colmap_io import write_colmap_sparse
from dashsplat.trainers.registry import get_trainer, resolve_trainer_name

ProgressCb = Callable[[str, float], None]


def train_splat(
    means: np.ndarray,
    colors: np.ndarray,
    cameras: list[dict[str, Any]],
    out_dir: Path,
    *,
    iterations: int = 2500,
    max_splats: int = 300_000,
    masks_dir: Path | None = None,
    trainer_name: str | None = None,
    force_fallback: bool = False,
    detail: dict[str, Any] | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_json(out_dir / "cameras.json", cameras)

    name = resolve_trainer_name(trainer_name, force_fallback=force_fallback)
    config = TrainerConfig(
        iterations=iterations,
        max_splats=max_splats,
        masks_dir=masks_dir,
        force_fallback=force_fallback,
        detail=detail or {},
    )

    if name == "seeded":
        trainer = get_trainer(
            "seeded",
            means=means,
            colors=colors,
            cameras=cameras,
            force_fallback=True,
        )
        result = trainer.train(out_dir, out_dir, config=config, progress=progress)
        return {
            "trainer": result.trainer,
            "ply": result.ply,
            "spz": result.spz,
            "cameras": result.cameras,
            "exit_code": result.exit_code,
            "num_gaussians": result.num_gaussians,
            **result.meta,
        }

    project = write_colmap_sparse(
        out_dir / f"{name}_project",
        means,
        colors,
        cameras,
        masks_dir=masks_dir,
    )
    trainer = get_trainer(name, means=means, colors=colors, cameras=cameras)
    try:
        result = trainer.train(project, out_dir, config=config, progress=progress)
        return {
            "trainer": result.trainer,
            "ply": result.ply,
            "spz": result.spz,
            "cameras": result.cameras or str(out_dir / "cameras.json"),
            "exit_code": result.exit_code,
            "num_gaussians": result.num_gaussians,
            "log": result.log,
            **result.meta,
        }
    except Exception as e:
        if progress:
            progress(f"{name} failed ({e}); using seeded fallback", 0.45)
        seeded = get_trainer(
            "seeded",
            means=means,
            colors=colors,
            cameras=cameras,
            force_fallback=True,
        )
        result = seeded.train(out_dir, out_dir, config=config, progress=progress)
        return {
            "trainer": result.trainer,
            "ply": result.ply,
            "spz": result.spz,
            "cameras": result.cameras,
            "exit_code": result.exit_code,
            "num_gaussians": result.num_gaussians,
            "fallback_from": name,
            "fallback_error": str(e),
            **result.meta,
        }
