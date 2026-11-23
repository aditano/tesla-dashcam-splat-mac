from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dashsplat.export import write_cameras_json, write_gaussian_ply, write_spz_stub
from dashsplat.trainers.base import ProgressCb, TrainerConfig, TrainResult


class SeededTrainer:
    """MapAnything-point seeded Gaussians — always available, used in CI/smoke."""

    name = "seeded"

    def __init__(self, means: np.ndarray | None = None, colors: np.ndarray | None = None, cameras: list[dict[str, Any]] | None = None):
        self._means = means
        self._colors = colors
        self._cameras = cameras or []

    def is_available(self) -> bool:
        return True

    def bind(self, means: np.ndarray, colors: np.ndarray, cameras: list[dict[str, Any]]) -> SeededTrainer:
        self._means = means
        self._colors = colors
        self._cameras = cameras
        return self

    def train(
        self,
        project_dir: Path,
        out_dir: Path,
        *,
        config: TrainerConfig,
        progress: ProgressCb | None = None,
    ) -> TrainResult:
        if self._means is None or self._colors is None:
            raise RuntimeError("SeededTrainer requires means/colors via bind()")
        if progress:
            progress("Seeding Gaussians from MapAnything points…", 0.5)
        out_dir.mkdir(parents=True, exist_ok=True)
        ply = out_dir / "splat.ply"
        spz = out_dir / "splat.spz"
        means = self._means
        colors = self._colors
        if means.shape[0] > 1:
            sample = means[: min(2000, means.shape[0])]
            d = np.linalg.norm(sample[None, :, :] - sample[:, None, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            nn = np.min(d, axis=1)
            med = float(np.median(nn[np.isfinite(nn)])) if np.any(np.isfinite(nn)) else 0.05
            scale = max(0.01, min(0.25, med * 1.5))
        else:
            scale = 0.05
        scales = np.full((means.shape[0], 3), scale, dtype=np.float32)
        # Honor max_splats cap
        if means.shape[0] > config.max_splats:
            rng = np.random.default_rng(0)
            idx = rng.choice(means.shape[0], config.max_splats, replace=False)
            means = means[idx]
            colors = colors[idx]
            scales = scales[idx]
        write_gaussian_ply(ply, means, colors, scales=scales)
        write_spz_stub(spz, means, colors)
        write_cameras_json(out_dir / "cameras.json", self._cameras)
        if progress:
            progress("Seeded splat export done", 0.95)
        return TrainResult(
            trainer=self.name,
            ply=str(ply),
            spz=str(spz),
            cameras=str(out_dir / "cameras.json"),
            exit_code=0,
            num_gaussians=int(means.shape[0]),
            meta={"reason": "seeded-fallback", "scale": scale},
        )
