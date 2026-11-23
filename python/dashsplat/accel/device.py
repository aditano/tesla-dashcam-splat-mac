"""Host / accelerator detection for Apple Silicon pipelines."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum


class ComputeDevice(str, Enum):
    MLX_GPU = "mlx_gpu"
    GPU = "gpu"
    METAL = "metal"
    ANE = "ane"
    COREML_ALL = "coreml_all"
    VIDEOTOOLBOX = "videotoolbox"
    CPU = "cpu"


@dataclass
class HostInfo:
    chip: str
    ram_bytes: int
    cpu_brand: str
    performance_cores: int = 4
    gpu_cores: int = 0
    is_apple_silicon: bool = False
    videotoolbox_available: bool = False
    metal_available: bool = False
    mlx_device: str | None = None
    coreml_compute_units: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / float(1 << 30)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ram_gb"] = round(self.ram_gb, 2)
        return d


def _sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return (out.stdout or "").strip() or None
    except Exception:
        return None
    return None


def _ram_bytes() -> int:
    # Darwin: hw.memsize; Linux CI/dev: /proc/meminfo or os.sysconf
    raw = _sysctl("hw.memsize")
    if raw and raw.isdigit():
        return int(raw)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page > 0:
            return int(pages * page)
    except (ValueError, OSError, AttributeError):
        pass
    return 8 * (1 << 30)


def _ffmpeg_has_videotoolbox() -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    try:
        out = subprocess.run(
            [ff, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        text = (out.stdout or "") + (out.stderr or "")
        return "videotoolbox" in text.lower()
    except Exception:
        return False


def _probe_mlx_device() -> str | None:
    try:
        import mlx.core as mx  # type: ignore

        # Prefer GPU on Apple Silicon
        try:
            # mlx exposes default_device()
            dev = str(mx.default_device())
            return dev
        except Exception:
            return "mlx"
    except Exception:
        return None


def detect_host() -> HostInfo:
    brand = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    machine = platform.machine().lower()
    is_as = platform.system() == "Darwin" and machine in ("arm64", "aarch64")
    chip = brand if brand else ("Apple Silicon" if is_as else platform.platform())
    perf = 4
    # hw.perflevel0.logicalcpu is performance cores on Apple Silicon when present
    pl0 = _sysctl("hw.perflevel0.logicalcpu")
    if pl0 and pl0.isdigit():
        perf = max(1, int(pl0))
    else:
        ncpu = _sysctl("hw.ncpu") or _sysctl("hw.logicalcpu")
        if ncpu and ncpu.isdigit():
            perf = max(1, int(ncpu) // 2 if is_as else int(ncpu))

    gpu_cores = 0
    # Not always exposed via sysctl; leave 0 when unknown
    gpu_raw = _sysctl("hw.gpu.count")  # may not exist
    if gpu_raw and gpu_raw.isdigit():
        gpu_cores = int(gpu_raw)

    vt = is_as and _ffmpeg_has_videotoolbox()
    metal = is_as  # Metal present on Apple Silicon macOS; trainers may still be missing
    mlx = _probe_mlx_device() if is_as else None
    warnings: list[str] = []
    if is_as and mlx and "cpu" in mlx.lower():
        warnings.append("MLX default device is CPU — expected GPU on Apple Silicon")
    if is_as and not vt:
        warnings.append("VideoToolbox hwaccel not reported by ffmpeg")
    if not is_as:
        warnings.append("Not Apple Silicon — decode/train/masks fall back to CPU paths")

    return HostInfo(
        chip=chip,
        ram_bytes=_ram_bytes(),
        cpu_brand=brand or chip,
        performance_cores=perf,
        gpu_cores=gpu_cores,
        is_apple_silicon=is_as,
        videotoolbox_available=vt,
        metal_available=metal,
        mlx_device=mlx,
        coreml_compute_units="all" if is_as else "cpuOnly",
        warnings=warnings,
    )


def preferred_mlx_device(host: HostInfo) -> ComputeDevice:
    if host.is_apple_silicon:
        return ComputeDevice.MLX_GPU
    return ComputeDevice.CPU


def preferred_train_device(host: HostInfo) -> ComputeDevice:
    if host.is_apple_silicon and host.metal_available:
        return ComputeDevice.METAL
    return ComputeDevice.CPU


def preferred_coreml_units(host: HostInfo) -> str:
    """Core ML computeUnits string: 'all' = ANE+GPU when available."""
    if host.is_apple_silicon:
        return "all"
    return "cpuOnly"


def preferred_decode_backend(host: HostInfo) -> str:
    if host.is_apple_silicon and host.videotoolbox_available:
        return "videotoolbox"
    if host.is_apple_silicon:
        return "avfoundation"
    return "software"
