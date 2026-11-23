from __future__ import annotations

import numpy as np
import pytest

from dashsplat.trainers.registry import available_trainers, get_trainer, resolve_trainer_name
from dashsplat.train_splat import train_splat


def test_opensplat_rejected():
    with pytest.raises(ValueError, match="AGPL"):
        resolve_trainer_name("opensplat")


def test_seeded_always_available():
    assert "seeded" in available_trainers()
    assert resolve_trainer_name(force_fallback=True) == "seeded"


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


def test_get_trainer_seeded_bind():
    t = get_trainer("seeded", means=np.zeros((2, 3), np.float32), colors=np.ones((2, 3), np.float32), cameras=[], force_fallback=True)
    assert t.name == "seeded"
    assert t.is_available()
