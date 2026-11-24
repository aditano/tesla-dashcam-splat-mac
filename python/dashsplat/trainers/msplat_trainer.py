"""msplat trainer (Apache-2.0) — speed option on Apple Silicon Metal.

https://github.com/rayanht/msplat — `pip install msplat`
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

from dashsplat.trainers.base import ProgressCb, TrainerConfig, TrainResult


def msplat_importable() -> bool:
    return importlib.util.find_spec("msplat") is not None


def find_msplat_cli() -> Path | None:
    which = shutil.which("msplat")
    return Path(which) if which else None


def msplat_trainer_works() -> bool:
    """True only if a working trainer entrypoint responds (not merely importable)."""
    cli = find_msplat_cli()
    if cli is not None:
        try:
            proc = subprocess.run(
                [str(cli), "--help"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0 or "usage" in (proc.stdout + proc.stderr).lower():
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not msplat_importable():
        return False
    try:
        import msplat  # type: ignore

        # Prefer an explicit train/run hook if present
        for attr in ("train", "run", "main", "Trainer"):
            if hasattr(msplat, attr):
                return True
        # Module import alone is not enough
        return False
    except Exception:
        return False


class MSplatTrainer:
    name = "msplat"

    def is_available(self) -> bool:
        return msplat_trainer_works()

    def train(
        self,
        project_dir: Path,
        out_dir: Path,
        *,
        config: TrainerConfig,
        progress: ProgressCb | None = None,
    ) -> TrainResult:
        if progress:
            progress("Training with msplat (Metal)…", 0.4)
        out_dir.mkdir(parents=True, exist_ok=True)
        ply = out_dir / "splat.ply"

        detail = config.detail or {}
        cli = find_msplat_cli()
        if cli:
            cmd = [
                str(cli),
                str(project_dir),
                "--iterations",
                str(config.iterations),
                "--output",
                str(ply),
            ]
            # gsplat-compatible AbsGS / scale prune when CLI supports them
            if detail.get("absgrad"):
                cmd += ["--absgrad"]
            if detail.get("max_scale") is not None:
                cmd += ["--max-scale", str(detail["max_scale"])]
            if detail.get("antialias"):
                cmd += ["--antialias"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # Retry without experimental detail flags
                cmd = [
                    str(cli),
                    str(project_dir),
                    "--iterations",
                    str(config.iterations),
                    "--output",
                    str(ply),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"msplat CLI failed ({proc.returncode}): {proc.stderr[-2000:]}")
            return TrainResult(
                trainer=self.name,
                ply=str(ply),
                spz=None,
                cameras=str(out_dir / "cameras.json") if (out_dir / "cameras.json").exists() else None,
                exit_code=proc.returncode,
                log=proc.stdout[-2000:] if proc.stdout else None,
                meta={"detail": detail},
            )

        # Python API path
        try:
            import msplat  # type: ignore
        except ImportError as e:
            raise RuntimeError("msplat not installed") from e

        # Best-effort: API surface varies by version; prefer a train() helper if present.
        kwargs = {
            "output": str(ply),
            "iterations": config.iterations,
        }
        if detail.get("absgrad"):
            kwargs["absgrad"] = True
        if detail.get("max_scale") is not None:
            kwargs["max_scale"] = detail["max_scale"]
        if hasattr(msplat, "train"):
            try:
                msplat.train(str(project_dir), **kwargs)  # type: ignore[attr-defined]
            except TypeError:
                msplat.train(  # type: ignore[attr-defined]
                    str(project_dir),
                    output=str(ply),
                    iterations=config.iterations,
                )
        else:
            raise RuntimeError(
                "msplat is installed but exposes no known train() API; "
                "install the CLI extra or set DASHSPLAT_TRAINER=brush"
            )
        if progress:
            progress("msplat finished", 0.95)
        return TrainResult(
            trainer=self.name,
            ply=str(ply),
            spz=None,
            cameras=str(out_dir / "cameras.json") if (out_dir / "cameras.json").exists() else None,
            exit_code=0,
            meta={"detail": detail},
        )
