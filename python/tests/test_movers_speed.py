from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dashsplat.masks.movers import mover_mask_ego_motion


def test_driving_speed_does_not_blank_static_scene(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    # Large frame difference (as when the car moves)
    Image.fromarray(np.full((64, 64), 40, dtype=np.uint8)).save(a)
    Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(b)
    keep = mover_mask_ego_motion(b, a, speed_mps=12.0)
    assert keep.mean() > 250, "raw differencing while driving must not blank the scene"
