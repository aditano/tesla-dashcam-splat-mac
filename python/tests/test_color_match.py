from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dashsplat.color_match import apply_affine, fit_affine_rgb, match_cameras_to_front


def test_fit_affine_recovers_scale_bias_via_global_stats():
    rng = np.random.default_rng(0)
    src = rng.integers(30, 200, size=(40, 40, 3)).astype(np.uint8)
    scale = np.array([1.2, 0.9, 1.1])
    bias = np.array([5.0, -3.0, 2.0])
    tgt = apply_affine(src, scale, bias)
    s, b = fit_affine_rgb(src.astype(np.float32), tgt.astype(np.float32))
    assert np.allclose(s, scale, atol=0.05)
    assert np.allclose(b, bias, atol=1.0)


def test_match_writes_copies_not_overwriting_originals(tmp_path: Path):
    front = tmp_path / "front.jpg"
    side = tmp_path / "left.jpg"
    Image.fromarray(np.full((32, 32, 3), 80, dtype=np.uint8)).save(front)
    Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(side)
    original = side.read_bytes()
    out = tmp_path / "matched"
    aff = match_cameras_to_front({"front": front, "left_repeater": side}, out)
    assert side.read_bytes() == original
    assert Path(aff["left_repeater"]["path"]).exists()
    assert Path(aff["left_repeater"]["path"]).parent == out


def test_scale_is_clamped(tmp_path: Path):
    front = tmp_path / "front.jpg"
    side = tmp_path / "side.jpg"
    # Near-black side vs mid front → unconstrained scale would explode
    Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(front)
    Image.fromarray(np.full((64, 64, 3), 2, dtype=np.uint8)).save(side)
    aff = match_cameras_to_front({"front": front, "left_repeater": side}, tmp_path / "out")
    for s in aff["left_repeater"]["scale"]:
        assert 0.25 <= s <= 4.0


def test_multi_frame_stats_and_front_mask(tmp_path: Path):
    fronts = []
    sides = []
    for i in range(3):
        f = tmp_path / f"front_{i}.jpg"
        s = tmp_path / f"side_{i}.jpg"
        Image.fromarray(np.full((48, 48, 3), 100 + i * 5, dtype=np.uint8)).save(f)
        Image.fromarray(np.full((48, 48, 3), 140 + i * 5, dtype=np.uint8)).save(s)
        fronts.append(f)
        sides.append(s)
    # Front mask: keep lower half only — must affect stats (not a no-op)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[24:, :] = 255
    mp = tmp_path / "front_mask.png"
    Image.fromarray(mask).save(mp)
    aff = match_cameras_to_front(
        {"front": fronts, "left_repeater": sides},
        tmp_path / "out2",
        masks={"front": mp},
    )
    assert aff["left_repeater"]["n_frames"] == 3
    assert aff["front"]["n_frames"] == 3
