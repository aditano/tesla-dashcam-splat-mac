"""3DGS training entry — Brush (default) / msplat / seeded fallback.

OpenSplat (AGPL-3.0) was removed. See docs/ARCHITECTURE.md and the 2026-11-23 research note.

Fallback chain is always Brush → msplat → seeded CPU. The returned `trainer`
field names the trainer that actually produced the PLY (never silently report
seeded CPU as a Full Metal success).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from dashsplat.export import write_cameras_json
from dashsplat.trainers.base import TrainerConfig
from dashsplat.trainers.colmap_io import write_colmap_sparse
from dashsplat.trainers.registry import available_trainers, get_trainer, resolve_trainer_name

ProgressCb = Callable[[str, float], None]


def _fallback_chain(preferred: str) -> list[str]:
    """Ordered trainers to try after `preferred` fails (excluding preferred)."""
    chain = ["brush", "msplat", "seeded"]
    out: list[str] = []
    seen = {preferred}
    for name in chain:
        if name in seen:
            continue
        if name == "seeded" or name in available_trainers():
            out.append(name)
            seen.add(name)
    return out


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

    attempts: list[str] = [name] + ([] if force_fallback or name == "seeded" else _fallback_chain(name))
    errors: list[dict[str, str]] = []
    last_error: Exception | None = None

    for attempt in attempts:
        try:
            if attempt == "seeded":
                trainer = get_trainer(
                    "seeded",
                    means=means,
                    colors=colors,
                    cameras=cameras,
                    force_fallback=True,
                )
                result = trainer.train(out_dir, out_dir, config=config, progress=progress)
            else:
                project = write_colmap_sparse(
                    out_dir / f"{attempt}_project",
                    means,
                    colors,
                    cameras,
                    masks_dir=masks_dir,
                )
                trainer = get_trainer(attempt, means=means, colors=colors, cameras=cameras)
                if not trainer.is_available():
                    raise RuntimeError(f"{attempt} not available")
                result = trainer.train(project, out_dir, config=config, progress=progress)

            payload: dict[str, Any] = {
                "trainer": result.trainer,
                "ply": result.ply,
                "spz": result.spz,
                "cameras": result.cameras or str(out_dir / "cameras.json"),
                "exit_code": result.exit_code,
                "num_gaussians": result.num_gaussians,
                "log": result.log,
                **result.meta,
            }
            if attempt != name:
                payload["fallback_from"] = name
                payload["fallback_chain"] = attempts
                payload["fallback_errors"] = errors
            if result.trainer == "seeded" and name != "seeded":
                payload["cpu_fallback"] = True
                if progress:
                    progress(
                        f"CPU seeded fallback used (requested {name}; tried {attempts})",
                        0.95,
                    )
            return payload
        except Exception as e:
            last_error = e
            errors.append({"trainer": attempt, "error": str(e)})
            if progress:
                nxt = attempts[attempts.index(attempt) + 1] if attempts.index(attempt) + 1 < len(attempts) else None
                if nxt:
                    progress(f"{attempt} failed ({e}); trying {nxt}…", 0.45)
            continue

    raise RuntimeError(f"All trainers failed: {errors}") from last_error
