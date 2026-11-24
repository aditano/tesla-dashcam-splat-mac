"""mlx-vlm 0.8.1: local cache must be the first positional arg, not path=."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from dashsplat import mapanything as ma


def _install_fake_mlx(fake_load):
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    vlm = types.ModuleType("mlx_vlm")
    vlm.load = fake_load
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    sys.modules["mlx_vlm"] = vlm


def test_load_uses_cache_dir_as_positional_path(tmp_path: Path):
    cache = tmp_path / "map-anything-apache-fp16"
    cache.mkdir()
    (cache / "config.json").write_text("{}")
    captured: dict = {}

    def fake_load(path_or_hf_repo, **kwargs):
        captured["arg"] = path_or_hf_repo
        captured["kwargs"] = kwargs
        return ("model", "processor")

    _install_fake_mlx(fake_load)
    result = ma._load_mapanything("mlx-community/map-anything-apache-fp16", cache_dir=cache)
    assert result == ("model", "processor")
    assert captured["arg"] == str(cache)
    assert "path" not in captured["kwargs"]


def test_load_falls_back_to_repo_id_when_cache_missing(tmp_path: Path):
    captured: dict = {}

    def fake_load(path_or_hf_repo, **kwargs):
        captured["arg"] = path_or_hf_repo
        captured["kwargs"] = kwargs
        return ("m", "p")

    _install_fake_mlx(fake_load)
    missing = tmp_path / "nope"
    ma._load_mapanything("mlx-community/map-anything-apache-fp16", cache_dir=missing)
    assert captured["arg"] == "mlx-community/map-anything-apache-fp16"
