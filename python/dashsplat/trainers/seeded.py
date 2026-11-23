from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dashsplat.detail_quality import FULL_DETAIL, PREVIEW_DETAIL, refine_thin_structure_scales
from dashsplat.export import write_cameras_json, write_gaussian_ply, write_spz_stub
from dashsplat.trainers.base import ProgressCb, TrainerConfig, TrainResult


def _estimate_tangents(means: np.ndarray) -> np.ndarray | None:
    """Cheap local PCA tangent without sklearn."""
    n = means.shape[0]
    if n < 3:
        return None
    # For each point, use a random subsample of neighbours by spatial hash bins
    tangents = np.zeros_like(means)
    # Sort by x then take sliding window
    order = np.argsort(means[:, 0])
    sorted_m = means[order]
    window = min(12, n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, lo + window)
        lo = max(0, hi - window)
        pts = sorted_m[lo:hi]
        pts = pts - pts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(pts, full_matrices=False)
            tangents[order[i]] = vh[0]
        except np.linalg.LinAlgError:
            tangents[order[i]] = np.array([1.0, 0.0, 0.0])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    return tangents / norms


class SeededTrainer:
    """MapAnything-point seeded Gaussians — always available, used in CI/smoke."""

    name = "seeded"

    def __init__(
        self,
        means: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        cameras: list[dict[str, Any]] | None = None,
    ):
        self._means = means
        self._colors = colors
        self._cameras = cameras or []

    def is_available(self) -> bool:
        return True

    def bind(
        self, means: np.ndarray, colors: np.ndarray, cameras: list[dict[str, Any]]
    ) -> SeededTrainer:
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
        detail = config.detail or {}
        # Prefer Full thin-structure settings when absgrad / tight max_scale requested
        if detail.get("absgrad") or float(detail.get("max_scale", 1.0)) <= 0.15:
            dq = FULL_DETAIL
        else:
            dq = PREVIEW_DETAIL
        if detail.get("max_scale") is not None:
            # Override cap from trainer config
            from dataclasses import replace

            dq = replace(dq, max_scale_m=float(detail["max_scale"]))
        tangents = _estimate_tangents(means)
        scales = refine_thin_structure_scales(means, tangents, dq)
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
            meta={
                "reason": "seeded-fallback",
                "max_scale": float(np.max(scales)),
                "detail": detail,
            },
        )
