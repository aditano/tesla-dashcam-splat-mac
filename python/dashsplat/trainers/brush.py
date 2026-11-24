"""Brush trainer (Apache-2.0) — default Mac Metal trainer.

Brush CLI: https://github.com/ArthurBrussee/brush
Flags verified against main-branch sources:
- crates/brush-train/src/config.rs — `--total-train-iters`, `--max-splats`,
  `--min-scale-factor`, `--opac-decay`, `--growth-grad-threshold`, …
- crates/brush-process/src/config.rs — `--export-path` (directory),
  `--export-name` (filename, supports `{iter}`)

Unsupported / removed CLI flags (not in Brush main): `--total-steps`,
`--mcmc`, `--max-scale`, `--antialias`, `--opacity-reset-every`, `--absgrad`.
MCMC-like densification is built into Brush ≥0.3; detail prefs map onto real
equivalents where they exist.
"""

from __future__ import annotations

import json
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
    cand = support_layout()["root"] / "bin" / "brush"
    if cand.exists():
        return cand
    return None


def detail_flags_for_brush(detail: dict | None) -> list[str]:
    """Map DetailQuality prefs onto Brush CLI flags that actually exist.

    Sources: ArthurBrussee/brush crates/brush-train/src/config.rs
    - antialias / mip-splatting → `--min-scale-factor` (default 0.1; 0 disables)
    - absgrad densify aggressiveness → lower `--growth-grad-threshold`
    - opacity reset → continuous `--opac-decay` (no periodic reset flag)
    - max_scale → `--split-at-screen-size` (screen-space split threshold)
    - mcmc → no flag (Brush 0.3+ always uses MCMC-like growth)
    """
    d = detail or {}
    flags: list[str] = []
    if d.get("antialias"):
        # Mip-Splatting-style 3D filter strength (paper `s`); keep Brush default-ish
        flags += ["--min-scale-factor", "0.1"]
    if d.get("absgrad") or d.get("densify") == "absgrad":
        # AbsGS-like: densify more readily (Brush default is 0.0025)
        thresh = d.get("grow_grad2d")
        if thresh is not None:
            flags += ["--growth-grad-threshold", str(float(thresh))]
        else:
            flags += ["--growth-grad-threshold", "0.0008"]
    if d.get("opacity_reset_every"):
        # No periodic reset; continuous opacity decay is the supported control
        flags += ["--opac-decay", "0.004"]
    if d.get("max_scale") is not None:
        # Lower screen-size split threshold encourages smaller splats
        max_s = float(d["max_scale"])
        split = 0.25 if max_s <= 0.15 else 0.5
        flags += ["--split-at-screen-size", str(split)]
    return flags


def build_brush_argv(
    brush: Path | str,
    project_dir: Path | str,
    *,
    export_dir: Path | str,
    export_name: str = "splat.ply",
    iterations: int = 2500,
    max_splats: int = 300_000,
    detail: dict | None = None,
    include_detail_flags: bool = True,
) -> list[str]:
    """Build Brush CLI argv.

    `--export-path` is a directory; the PLY filename is `--export-name`.
    Iteration count is `--total-train-iters` (renamed from `--total-steps`).
    """
    cmd = [
        str(brush),
        str(project_dir),
        "--export-path",
        str(export_dir),
        "--export-name",
        export_name,
        "--total-train-iters",
        str(int(iterations)),
        "--max-splats",
        str(int(max_splats)),
    ]
    if include_detail_flags:
        cmd += detail_flags_for_brush(detail)
    return cmd


def _locate_exported_ply(export_dir: Path, export_name: str) -> Path | None:
    direct = export_dir / export_name
    if direct.exists():
        return direct
    # Brush may substitute {iter} even when we omit it; pick newest .ply
    ply_files = sorted(export_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ply_files[0] if ply_files else None


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
        export_dir = out_dir / "brush_export"
        # Clear prior exports so a failed/partial run cannot leave a stale PLY
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_name = "splat.ply"
        ply = out_dir / "splat.ply"
        detail_path = out_dir / "trainer_detail.json"
        detail_path.write_text(json.dumps(config.detail or {}, indent=2))

        # Always recopy masks every run (stale masks from a prior refine are wrong)
        if config.masks_dir and config.masks_dir.exists():
            dest = project_dir / "masks"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(config.masks_dir, dest, dirs_exist_ok=True)

        cmd = build_brush_argv(
            brush,
            project_dir,
            export_dir=export_dir.resolve(),
            export_name=export_name,
            iterations=config.iterations,
            max_splats=config.max_splats,
            detail=config.detail,
            include_detail_flags=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Retry without detail-mapped flags (older / stricter Brush builds)
            cmd = build_brush_argv(
                brush,
                project_dir,
                export_dir=export_dir.resolve(),
                export_name=export_name,
                iterations=config.iterations,
                max_splats=config.max_splats,
                detail=None,
                include_detail_flags=False,
            )
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Brush failed ({proc.returncode}): {proc.stderr[-2000:]}")

        exported = _locate_exported_ply(export_dir, export_name)
        if exported is None:
            raise RuntimeError(f"Brush succeeded but no PLY in {export_dir}")
        if exported.resolve() != ply.resolve():
            shutil.copy2(exported, ply)

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
            meta={"detail": config.detail, "brush_argv": cmd},
        )
