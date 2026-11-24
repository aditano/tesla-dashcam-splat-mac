"""First-run dependency + model + trainer install into Application Support."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from dashsplat.mapanything import MAPANYTHING_HF_APACHE, MAPANYTHING_GIT_COMMIT

ProgressCb = Callable[[str, float], None]

APP_SUPPORT_NAME = "DashSplat"
# Official HF weights (Apache). Legacy mlx-community id kept only as alias in mapanything.py.
MAPANYTHING_REPO = MAPANYTHING_HF_APACHE
# Keep in sync with Apps/DashSplat/Services/PythonBridge.swift bootstrapMarkerVersion
BOOTSTRAP_MARKER_VERSION = "1.2.0"  # bump: PyTorch MapAnything + uv pin


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
    # Pin installer script version; write to a unique mktemp path (not fixed /tmp).
    import tempfile

    uv_installer_version = "0.6.14"
    install_url = f"https://astral.sh/uv/{uv_installer_version}/install.sh"
    with tempfile.NamedTemporaryFile(prefix="uv-install-", suffix=".sh", delete=False) as tf:
        script_path = Path(tf.name)
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import urllib.request; "
                    f"urllib.request.urlretrieve({install_url!r}, {str(script_path)!r})"
                ),
            ],
            check=False,
        )
        if script_path.exists() and script_path.stat().st_size > 0:
            subprocess.run(["sh", str(script_path)], check=False)
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass
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


def bootstrap_is_complete(support_dir: Path | None = None) -> bool:
    """True when marker version matches and venv python exists (outside the app bundle)."""
    paths = support_layout(support_dir)
    marker = paths["marker"]
    py = paths["venv"] / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not marker.exists() or not py.exists():
        return False
    try:
        data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("version") == BOOTSTRAP_MARKER_VERSION


def bootstrap(
    *,
    support_dir: Path | None = None,
    with_mlx: bool = True,
    download_model: bool = True,
    with_trainers: bool = True,
    progress: ProgressCb | None = None,
    force: bool = False,
) -> dict:
    """Install Python deps, trainers, and optionally MapAnything weights.

    Always writes into Application Support (never the app bundle). Idempotent:
    skips work when the marker version already matches unless ``force``.
    """
    paths = support_layout(support_dir)
    for key in ("root", "models", "cache", "jobs", "bin"):
        paths[key].mkdir(parents=True, exist_ok=True)

    def report(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)
        print(json.dumps({"type": "progress", "message": msg, "fraction": frac}), flush=True)

    if not force and bootstrap_is_complete(support_dir):
        report("Bootstrap already complete — skipping", 1.0)
        existing = json.loads(paths["marker"].read_text())
        return {
            "ok": True,
            "exit_code": 0,
            "skipped": True,
            **existing,
            "paths": {k: str(v) for k, v in paths.items()},
        }

    report("Locating uv…", 0.05)
    uv = ensure_uv()
    report("Creating virtualenv (with pip via --seed)…", 0.15)
    venv = paths["venv"]
    # --seed ensures pip exists so msplat and other pip-only packages install cleanly
    code = _run([uv, "venv", "--seed", str(venv)])
    if code != 0:
        return {"ok": False, "exit_code": code, "step": "venv"}

    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    report("Installing dashsplat package…", 0.35)
    # Install from the repo/bundle *source* into Application Support venv — never mutate the bundle.
    pkg_root = Path(__file__).resolve().parents[1]
    extras = "dev"
    if with_mlx and sys.platform == "darwin":
        # Official PyTorch MapAnything (facebookresearch), not mlx-vlm
        extras = "dev,mapanything"
    # Non-editable install into Application Support venv — never `-e` into the
    # app bundle / source tree (Finder-launched apps must not mutate the bundle).
    code = _run([uv, "pip", "install", "--python", str(py), f"{pkg_root}[{extras}]"])
    if code != 0:
        code = _run([uv, "pip", "install", "--python", str(py), f"{pkg_root}[dev]"])
        if code != 0:
            return {"ok": False, "exit_code": code, "step": "pip"}

    model_path = paths["models"] / "facebook-map-anything-apache"
    if download_model and with_mlx and sys.platform == "darwin":
        report(f"Downloading {MAPANYTHING_HF_APACHE} (pkg @{MAPANYTHING_GIT_COMMIT[:7]})…", 0.55)
        try:
            code = _run(
                [
                    str(py),
                    "-c",
                    (
                        "from huggingface_hub import snapshot_download; "
                        f"snapshot_download('{MAPANYTHING_HF_APACHE}', local_dir=r'{model_path}')"
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
        "version": BOOTSTRAP_MARKER_VERSION,
        "venv": str(venv),
        "python": str(py),
        "model_dir": str(model_path),
        "mapanything_repo": MAPANYTHING_REPO,
        "trainers": trainer_info,
        "default_trainer": "brush",
        "support_root": str(paths["root"]),
    }
    paths["marker"].write_text(json.dumps(marker, indent=2))
    report("Bootstrap complete", 1.0)
    return {"ok": True, "exit_code": 0, **marker, "paths": {k: str(v) for k, v in paths.items()}}
