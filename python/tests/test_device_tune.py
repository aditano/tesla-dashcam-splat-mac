"""Failing-first tests for Apple Silicon device selection and RAM-based tuning."""

from __future__ import annotations

from dashsplat.accel.device import (
    ComputeDevice,
    HostInfo,
    detect_host,
    preferred_coreml_units,
    preferred_decode_backend,
    preferred_mlx_device,
    preferred_train_device,
)
from dashsplat.accel.tune import TunedBudget, tune_for_host


def test_tune_scales_with_ram():
    low = HostInfo(
        chip="Apple M2",
        ram_bytes=16 * (1 << 30),
        cpu_brand="Apple M2",
        performance_cores=4,
        gpu_cores=10,
        is_apple_silicon=True,
    )
    high = HostInfo(
        chip="Apple M4 Max",
        ram_bytes=128 * (1 << 30),
        cpu_brand="Apple M4 Max",
        performance_cores=12,
        gpu_cores=40,
        is_apple_silicon=True,
    )
    a = tune_for_host(low, tier="preview")
    b = tune_for_host(high, tier="preview")
    assert isinstance(a, TunedBudget)
    assert b.max_splats >= a.max_splats
    assert b.mapanything_window >= a.mapanything_window
    assert b.decode_parallel_cameras >= 1


def test_preferred_devices_prefer_gpu_on_apple_silicon():
    host = HostInfo(
        chip="Apple M3 Max",
        ram_bytes=64 * (1 << 30),
        cpu_brand="Apple M3 Max",
        performance_cores=12,
        gpu_cores=40,
        is_apple_silicon=True,
        videotoolbox_available=True,
        metal_available=True,
    )
    assert preferred_mlx_device(host) in (ComputeDevice.MLX_GPU, ComputeDevice.GPU)
    assert preferred_train_device(host) is ComputeDevice.METAL
    assert preferred_coreml_units(host) == "all"
    assert preferred_decode_backend(host) in ("videotoolbox", "avfoundation")


def test_cpu_fallback_flagged_when_not_apple_silicon():
    host = HostInfo(
        chip="Unknown",
        ram_bytes=8 * (1 << 30),
        cpu_brand="Intel",
        performance_cores=4,
        gpu_cores=0,
        is_apple_silicon=False,
        videotoolbox_available=False,
        metal_available=False,
    )
    assert preferred_mlx_device(host) is ComputeDevice.CPU
    assert preferred_train_device(host) is ComputeDevice.CPU
    assert preferred_decode_backend(host) == "software"


def test_detect_host_returns_required_fields():
    info = detect_host()
    assert info.ram_bytes > 0
    assert info.cpu_brand
    assert isinstance(info.is_apple_silicon, bool)
    assert info.to_dict()["ram_gb"] > 0
