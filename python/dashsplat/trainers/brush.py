"""Brush trainer (Apache-2.0) — default Mac Metal trainer.

Brush CLI: https://github.com/ArthurBrussee/brush
Supports COLMAP / Nerfstudio datasets, a `masks/` folder (black=ignore),
and MCMC-like training with a max splat cap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dashsplat.bootstrap import support_layout
from dashsplat.trainers.base import ProgressCb, TrainerConfig, TrainResult


def find_brush() -> Path | None:
    env = os.environ.get("DASHSPLAT_BRUSH")
    if env and Path(env).exists():
        return Path(env)
    which = shutil.which("brush")
    if which:
        return Path(which)
    # Application Support install location
    cand = support_layout()["root"] / "bin" / "brush"
    if cand.exists():
        return cand
    return None


class BrushTrainer:
    name = "brush"

    def is_available(self) -> bool:
        return find_brush() is not None

    def train(
        self,
        project_dir: Path,
        out_dir: Path,
        *,
        config: TrainerConfig,
        progress: ProgressCb | None = None,
    ) -> TrainResult:
        brush = find_brush()
        if not brush:
            raise RuntimeError("brush binary not found")
        if progress:
            progress("Training with Brush (Metal)…", 0.4)
        out_dir.mkdir(parents=True, exist_ok=True)
        ply = out_dir / "splat.ply"
        # Brush CLI evolves; try common flags. Dataset = COLMAP project root.
        cmd = [
            str(brush),
            str(project_dir),
            "--export-path",
            str(ply),
            "--total-steps",
            str(config.iterations),
            "--max-splats",
            str(config.max_splats),
        ]
        if config.masks_dir and config.masks_dir.exists():
            # Brush looks for a sibling `masks/` folder; ensure it's under project
            dest = project_dir / "masks"
            if not dest.exists():
                shutil.copytree(config.masks_dir, dest, dirs_exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Retry with a shorter flag set for older builds
            cmd2 = [str(brush), str(project_dir), "--export-path", str(ply)]
            proc = subprocess.run(cmd2, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Brush failed ({proc.returncode}): {proc.stderr[-2000:]}")
        if progress:
            progress("Brush finished", 0.95)
        spz = ply.with_suffix(".spz")
        return TrainResult(
            trainer=self.name,
            ply=str(ply),
            spz=str(spz) if spz.exists() else None,
            cameras=str(out_dir / "cameras.json") if (out_dir / "cameras.json").exists() else None,
            exit_code=proc.returncode,
            log=proc.stdout[-2000:] if proc.stdout else None,
        )
