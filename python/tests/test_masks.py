from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dashsplat.masks.ego import ego_mask_array
from dashsplat.masks.pipeline import build_masks_for_frames, combine_masks
from dashsplat.masks.sky import sky_mask_heuristic


def test_ego_mask_front_blacks_bottom():
    m = ego_mask_array("front", 100, 100)
    assert m.shape == (100, 100)
    assert m[90, 50] == 0  # hood
    assert m[10, 50] == 255  # sky area kept by ego mask


def test_ego_masks_differ_per_camera():
    f = ego_mask_array("front", 64, 64)
    l = ego_mask_array("left_repeater", 64, 64)
    assert not np.array_equal(f, l)


def test_combine_masks_intersection():
    a = np.full((4, 4), 255, np.uint8)
    b = np.full((4, 4), 255, np.uint8)
    a[0, 0] = 0
    b[0, 1] = 0
    c = combine_masks(a, b)
    assert c[0, 0] == 0 and c[0, 1] == 0 and c[1, 1] == 255


def test_build_masks_writes_pngs(tmp_path: Path):
    img = tmp_path / "front_0000.jpg"
    Image.new("RGB", (64, 48), (100, 140, 200)).save(img)
    out = build_masks_for_frames(
        [{"path": img, "camera": "front", "file_name": "front_0000.jpg", "speed_mps": 5.0}],
        tmp_path / "masks",
        tier="preview",
    )
    assert (out / "front_0000.png").exists()


def test_sky_heuristic_masks_blue_top(tmp_path: Path):
    arr = np.zeros((60, 40, 3), dtype=np.uint8)
    arr[:20] = (120, 150, 220)  # blue sky
    arr[20:] = (40, 40, 40)
    path = tmp_path / "x.jpg"
    Image.fromarray(arr).save(path)
    m = sky_mask_heuristic(path)
    assert m[:20].mean() < 200  # mostly masked
    assert m[40:].mean() > 200  # kept
