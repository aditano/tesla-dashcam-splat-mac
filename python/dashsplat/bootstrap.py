"""First-run dependency + model install into Application Support."""

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
    # Install uv via official installer script when missing
    subprocess.run(
        [sys.executable, "-c", "import urllib.request; urllib.request.urlretrieve('https://astral.sh/uv/install.sh', '/tmp/uv-install.sh')"],
        check=False,
    )
    if Path("/tmp/uv-install.sh").exists():
        subprocess.run(["sh", "/tmp/uv-install.sh"], check=False)
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    return uv


def bootstrap(
    *,
    support_dir: Path | None = None,
    with_mlx: bool = True,
    download_model: bool = True,
    progress: ProgressCb | None = None,
) -> dict:
    """Install Python deps + optionally download MapAnything weights. Idempotent."""
    paths = support_layout(support_dir)
    for key in ("root", "models", "cache", "jobs"):
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
    # Install this package editable from repo when available
    pkg_root = Path(__file__).resolve().parents[1]
    extras = "dev"
    if with_mlx and sys.platform == "darwin":
        extras = "dev,mlx"
    code = _run([uv, "pip", "install", "--python", str(py), "-e", f"{pkg_root}[{extras}]"])
    if code != 0:
        # Retry without mlx extras
        code = _run([uv, "pip", "install", "--python", str(py), "-e", f"{pkg_root}[dev]"])
        if code != 0:
            return {"ok": False, "exit_code": code, "step": "pip"}

    model_path = paths["models"] / "map-anything-apache-fp16"
    if download_model and with_mlx and sys.platform == "darwin":
        report(f"Downloading {MAPANYTHING_REPO}…", 0.6)
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
            report("Model download skipped/failed (will retry on first build)", 0.75)
    else:
        report("Skipping MapAnything download (non-Mac or --no-model)", 0.7)

    marker = {
        "version": "0.1.0",
        "venv": str(venv),
        "python": str(py),
        "model_dir": str(model_path),
        "mapanything_repo": MAPANYTHING_REPO,
    }
    paths["marker"].write_text(json.dumps(marker, indent=2))
    report("Bootstrap complete", 1.0)
    return {"ok": True, "exit_code": 0, **marker, "paths": {k: str(v) for k, v in paths.items()}}
