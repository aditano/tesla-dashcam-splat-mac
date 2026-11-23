"""dashsplat doctor — report chip, RAM, MLX, Core ML, Metal, VideoToolbox."""

from __future__ import annotations

import json
import shutil
from typing import Any

from dashsplat.accel.device import (
    detect_host,
    preferred_coreml_units,
    preferred_decode_backend,
    preferred_mlx_device,
    preferred_train_device,
)
from dashsplat.accel.tune import tune_for_host


def run_doctor(*, as_json: bool = True) -> dict[str, Any]:
    host = detect_host()
    tuned = tune_for_host(host, tier="preview")
    report: dict[str, Any] = {
        "ok": True,
        "host": host.to_dict(),
        "preferences": {
            "mlx": preferred_mlx_device(host).value,
            "train": preferred_train_device(host).value,
            "coreml_compute_units": preferred_coreml_units(host),
            "decode": preferred_decode_backend(host),
        },
        "tuned_preview": tuned.to_dict(),
        "binaries": {
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "ffprobe": bool(shutil.which("ffprobe")),
            "brush": bool(shutil.which("brush")),
        },
        "cpu_fallbacks": list(host.warnings),
    }
    if host.warnings:
        report["ok"] = False if any("Not Apple Silicon" in w for w in host.warnings) else True
        # Soft: warnings don't fail doctor exit unless nothing accelerated
    if as_json:
        print(json.dumps({"type": "doctor", **report}, indent=2))
    else:
        print(f"Chip: {host.chip}")
        print(f"RAM: {host.ram_gb:.1f} GB")
        print(f"Apple Silicon: {host.is_apple_silicon}")
        print(f"MLX device: {host.mlx_device or preferred_mlx_device(host).value}")
        print(f"Core ML units: {preferred_coreml_units(host)}")
        print(f"Metal: {host.metal_available}")
        print(f"VideoToolbox: {host.videotoolbox_available}")
        print(f"Decode backend: {preferred_decode_backend(host)}")
        print(f"Train device: {preferred_train_device(host).value}")
        for w in host.warnings:
            print(f"WARN: {w}")
    return report
