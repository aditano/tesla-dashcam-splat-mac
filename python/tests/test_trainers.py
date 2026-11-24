from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dashsplat.trainers.brush import build_brush_argv, detail_flags_for_brush
from dashsplat.trainers.registry import available_trainers, get_trainer, resolve_trainer_name
from dashsplat.train_splat import _fallback_chain, train_splat


def test_opensplat_rejected():
    with pytest.raises(ValueError, match="AGPL"):
        resolve_trainer_name("opensplat")


def test_seeded_always_available():
    assert "seeded" in available_trainers()
    assert resolve_trainer_name(force_fallback=True) == "seeded"


def test_brush_argv_uses_total_train_iters_and_export_dir():
    """Brush main: --total-train-iters; --export-path is a directory + --export-name."""
    argv = build_brush_argv(
        "/opt/brush",
        "/data/project",
        export_dir="/data/out/brush_export",
        export_name="splat.ply",
        iterations=2500,
        max_splats=100_000,
        include_detail_flags=False,
    )
    assert argv[0] == "/opt/brush"
    assert argv[1] == "/data/project"
    assert "--export-path" in argv
    assert argv[argv.index("--export-path") + 1] == "/data/out/brush_export"
    assert "--export-name" in argv
    assert argv[argv.index("--export-name") + 1] == "splat.ply"
    assert "--total-train-iters" in argv
    assert argv[argv.index("--total-train-iters") + 1] == "2500"
    assert "--total-steps" not in argv
    assert "--max-splats" in argv
    # export-path must not be a .ply file path
    export_path = Path(argv[argv.index("--export-path") + 1])
    assert export_path.suffix != ".ply"


def test_brush_argv_omits_unsupported_detail_flags():
    detail = {
        "mcmc": True,
        "max_scale": 0.12,
        "antialias": True,
        "opacity_reset_every": 3000,
        "absgrad": True,
        "grow_grad2d": 0.0008,
    }
    flags = detail_flags_for_brush(detail)
    unsupported = {
        "--mcmc",
        "--max-scale",
        "--antialias",
        "--opacity-reset-every",
        "--absgrad",
        "--total-steps",
    }
    assert unsupported.isdisjoint(flags)
    # Mapped equivalents that exist on Brush main
    assert "--min-scale-factor" in flags
    assert "--growth-grad-threshold" in flags
    assert "--opac-decay" in flags
    assert "--split-at-screen-size" in flags

    argv = build_brush_argv(
        "brush",
        "proj",
        export_dir="out",
        detail=detail,
        include_detail_flags=True,
    )
    assert unsupported.isdisjoint(argv)


def test_fallback_chain_brush_then_msplat_then_seeded():
    chain = _fallback_chain("brush")
    assert chain[0] == "msplat" or "msplat" not in available_trainers()
    assert "seeded" in chain
    assert chain[-1] == "seeded"


def test_train_splat_seeded(tmp_path):
    means = np.random.default_rng(0).normal(size=(200, 3)).astype(np.float32)
    colors = np.clip(np.random.default_rng(1).random((200, 3)), 0, 1).astype(np.float32)
    cameras = [
        {
            "file_name": "front_0000.jpg",
            "transform_matrix": np.eye(4).tolist(),
            "fl_x": 500,
            "fl_y": 500,
            "cx": 160,
            "cy": 120,
            "w": 320,
            "h": 240,
        }
    ]
    out = tmp_path / "out"
    result = train_splat(means, colors, cameras, out, force_fallback=True, iterations=10, max_splats=100)
    assert result["exit_code"] == 0
    assert result["trainer"] == "seeded"
    assert (out / "splat.ply").exists()
    assert (out / "splat.spz").exists()
    assert result["num_gaussians"] <= 100


def test_train_splat_tries_msplat_before_seeded(tmp_path):
    """When Brush fails, msplat must be attempted before seeded CPU."""
    means = np.zeros((8, 3), np.float32)
    colors = np.ones((8, 3), np.float32)
    cameras = [
        {
            "file_name": "front_0000.jpg",
            "transform_matrix": np.eye(4).tolist(),
            "fl_x": 100,
            "fl_y": 100,
            "cx": 50,
            "cy": 40,
            "w": 100,
            "h": 80,
        }
    ]
    tried: list[str] = []

    class FailBrush:
        name = "brush"

        def is_available(self):
            return True

        def train(self, *a, **k):
            tried.append("brush")
            raise RuntimeError("brush boom")

    class FailMSplat:
        name = "msplat"

        def is_available(self):
            return True

        def train(self, *a, **k):
            tried.append("msplat")
            raise RuntimeError("msplat boom")

    def fake_get(name, **kwargs):
        if name == "brush":
            return FailBrush()
        if name == "msplat":
            return FailMSplat()
        return get_trainer("seeded", means=means, colors=colors, cameras=cameras, force_fallback=True)

    with (
        patch("dashsplat.train_splat.resolve_trainer_name", return_value="brush"),
        patch("dashsplat.train_splat.available_trainers", return_value=["brush", "msplat", "seeded"]),
        patch("dashsplat.train_splat.get_trainer", side_effect=fake_get),
        patch("dashsplat.train_splat.write_colmap_sparse", return_value=tmp_path / "proj"),
    ):
        result = train_splat(means, colors, cameras, tmp_path / "out", trainer_name="brush")

    assert tried == ["brush", "msplat"]
    assert result["trainer"] == "seeded"
    assert result.get("cpu_fallback") is True
    assert result.get("fallback_from") == "brush"


def test_get_trainer_seeded_bind():
    t = get_trainer("seeded", means=np.zeros((2, 3), np.float32), colors=np.ones((2, 3), np.float32), cameras=[], force_fallback=True)
    assert t.name == "seeded"
    assert t.is_available()
