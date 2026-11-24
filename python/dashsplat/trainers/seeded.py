from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dashsplat.detail_quality import FULL_DETAIL, PREVIEW_DETAIL, refine_thin_structure_scales
from dashsplat.export import write_cameras_json, write_gaussian_ply, write_spz_stub
from dashsplat.trainers.base import ProgressCb, TrainerConfig, TrainResult


def _estimate_tangents(means: np.ndarray) -> np.ndarray | None:
    """Cheap local PCA tangent without sklearn — vectorized sliding windows."""
    n = means.shape[0]
    if n < 3:
        return None
    order = np.argsort(means[:, 0])
    sorted_m = means[order]
    window = min(12, n)
    # Build a (n, window, 3) neighbourhood via clip+broadcast indices
    centers = np.arange(n)
    offsets = np.arange(window) - window // 2
    idxs = np.clip(centers[:, None] + offsets[None, :], 0, n - 1)
    pts = sorted_m[idxs]  # (n, window, 3)
    pts = pts - pts.mean(axis=1, keepdims=True)
    # Batched SVD on neighbourhoods
    try:
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        tangents_sorted = vh[:, 0, :]  # (n, 3)
    except np.linalg.LinAlgError:
        tangents_sorted = np.zeros((n, 3), dtype=means.dtype)
        tangents_sorted[:, 0] = 1.0
    tangents = np.zeros_like(means)
    tangents[order] = tangents_sorted
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
