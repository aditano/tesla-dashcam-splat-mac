"""First-run dependency + model + trainer install into Application Support."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

ProgressCb = Callable[[str, float], None]

APP_SUPPORT_NAME = "DashSplat"
MAPANYTHING_REPO = "mlx-community/map-anything-apache-fp16"


def default_support_dir() -> Path:
    override = os.environ.get("DASHSPLAT_SUPPORT_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_SUPPORT_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_SUPPORT_NAME
    return Path.home() / ".local" / "share" / APP_SUPPORT_NAME


def support_layout(root: Path | None = None) -> dict[str, Path]:
    root = root or default_support_dir()
    return {
        "root": root,
        "venv": root / "venv",
        "models": root / "models",
        "cache": root / "cache",
        "jobs": root / "jobs",
        "bin": root / "bin",
        "marker": root / ".bootstrap_ok",
    }


def _run(cmd: list[str], env: dict | None = None) -> int:
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def ensure_uv() -> str:
    import shutil

    uv = shutil.which("uv")
    if uv:
        return uv
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import urllib.request; urllib.request.urlretrieve('https://astral.sh/uv/install.sh', '/tmp/uv-install.sh')",
        ],
        check=False,
    )
    if Path("/tmp/uv-install.sh").exists():
        subprocess.run(["sh", "/tmp/uv-install.sh"], check=False)
    return shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


def _install_trainers(py: Path, bin_dir: Path, report: Callable[[str, float], None]) -> dict:
    """Install msplat via pip; attempt Brush binary into Application Support/bin."""
    info: dict = {"msplat": False, "brush": False}
    bin_dir.mkdir(parents=True, exist_ok=True)
    report("Installing msplat (Apache-2.0 Metal trainer)…", 0.78)
    code = _run([str(py), "-m", "pip", "install", "msplat"])
    info["msplat"] = code == 0
    # Brush: look for existing, else note download URL for Apple Silicon release
    import shutil

    brush = shutil.which("brush")
    if brush:
        dest = bin_dir / "brush"
        try:
            if not dest.exists():
                shutil.copy2(brush, dest)
                dest.chmod(0o755)
            info["brush"] = True
        except OSError:
            info["brush"] = True  # still on PATH
            info["brush_path"] = brush
    else:
        info["brush"] = False
        info["brush_hint"] = (
            "Download the Apple Silicon Brush release from "
            "https://github.com/ArthurBrussee/brush/releases and place it at "
            f"{bin_dir / 'brush'} (or set DASHSPLAT_BRUSH)."
        )
    return info


def bootstrap(
    *,
    support_dir: Path | None = None,
    with_mlx: bool = True,
    download_model: bool = True,
    with_trainers: bool = True,
    progress: ProgressCb | None = None,
) -> dict:
    """Install Python deps, trainers, and optionally MapAnything weights. Idempotent."""
    paths = support_layout(support_dir)
    for key in ("root", "models", "cache", "jobs", "bin"):
        paths[key].mkdir(parents=True, exist_ok=True)

    def report(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)
        print(json.dumps({"type": "progress", "message": msg, "fraction": frac}), flush=True)

    report("Locating uv…", 0.05)
    uv = ensure_uv()
    report("Creating virtualenv…", 0.15)
    venv = paths["venv"]
    code = _run([uv, "venv", str(venv)])
    if code != 0:
        return {"ok": False, "exit_code": code, "step": "venv"}

    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    report("Installing dashsplat package…", 0.35)
    pkg_root = Path(__file__).resolve().parents[1]
    extras = "dev"
    if with_mlx and sys.platform == "darwin":
        extras = "dev,mlx"
    code = _run([uv, "pip", "install", "--python", str(py), "-e", f"{pkg_root}[{extras}]"])
    if code != 0:
        code = _run([uv, "pip", "install", "--python", str(py), "-e", f"{pkg_root}[dev]"])
        if code != 0:
            return {"ok": False, "exit_code": code, "step": "pip"}

    model_path = paths["models"] / "map-anything-apache-fp16"
    if download_model and with_mlx and sys.platform == "darwin":
        report(f"Downloading {MAPANYTHING_REPO}…", 0.55)
        try:
            code = _run(
                [
                    str(py),
                    "-c",
                    (
                        "from huggingface_hub import snapshot_download; "
                        f"snapshot_download('{MAPANYTHING_REPO}', local_dir=r'{model_path}')"
                    ),
                ]
            )
        except Exception:
            code = 1
        if code != 0:
            report("Model download skipped/failed (will retry on first build)", 0.7)
    else:
        report("Skipping MapAnything download (non-Mac or --no-model)", 0.7)

    trainer_info: dict = {}
    if with_trainers:
        trainer_info = _install_trainers(py, paths["bin"], report)
    else:
        report("Skipping trainer install", 0.85)

    marker = {
        "version": "1.0.0",
        "venv": str(venv),
        "python": str(py),
        "model_dir": str(model_path),
        "mapanything_repo": MAPANYTHING_REPO,
        "trainers": trainer_info,
        "default_trainer": "brush",
    }
    paths["marker"].write_text(json.dumps(marker, indent=2))
    report("Bootstrap complete", 1.0)
    return {"ok": True, "exit_code": 0, **marker, "paths": {k: str(v) for k, v in paths.items()}}
