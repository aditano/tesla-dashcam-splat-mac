"""Unit tests for MapAnything output adapter (mocked predictions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dashsplat.mapanything import (
    MAPANYTHING_GIT_COMMIT,
    MAPANYTHING_HF_APACHE,
    MAPANYTHING_PKG_VERSION,
    _resolve_pretrained_id,
    predictions_to_pointmap,
)


def test_pinned_version_constants():
    assert MAPANYTHING_PKG_VERSION == "1.1.4"
    assert MAPANYTHING_GIT_COMMIT.startswith("3d10cf7")
    assert MAPANYTHING_HF_APACHE == "facebook/map-anything-apache"


def test_resolve_pretrained_uses_local_cache(tmp_path: Path):
    cache = tmp_path / "facebook-map-anything-apache"
    cache.mkdir()
    (cache / "config.json").write_text("{}")
    assert _resolve_pretrained_id(MAPANYTHING_HF_APACHE, cache) == str(cache)
    assert _resolve_pretrained_id(MAPANYTHING_HF_APACHE, tmp_path / "missing") == MAPANYTHING_HF_APACHE


def test_predictions_to_pointmap_adapter(tmp_path: Path):
    img = tmp_path / "a.jpg"
    Image.fromarray(np.full((32, 48, 3), 120, dtype=np.uint8)).save(img)
    pred = {
        "pts3d": np.zeros((1, 32, 48, 3), dtype=np.float32) + 1.0,
        "conf": np.ones((1, 32, 48), dtype=np.float32),
        "camera_poses": np.eye(4, dtype=np.float32)[None, ...],
        "intrinsics": np.eye(3, dtype=np.float32)[None, ...],
        "img_no_norm": np.full((1, 32, 48, 3), 0.4, dtype=np.float32),
    }
    out = predictions_to_pointmap([pred], [img])
    assert out["pts3d"].shape[0] == 1
    assert out["colors"].shape[-1] == 3
    assert out["conf"].shape[0] == 1
    assert len(out["camera_poses"]) == 1
