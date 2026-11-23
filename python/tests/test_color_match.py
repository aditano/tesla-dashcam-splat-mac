from __future__ import annotations

import numpy as np

from dashsplat.color_match import apply_affine, fit_affine_rgb


def test_fit_affine_recovers_scale_bias():
    rng = np.random.default_rng(0)
    src = rng.integers(30, 200, size=(40, 40, 3)).astype(np.uint8)
    scale = np.array([1.2, 0.9, 1.1])
    bias = np.array([5.0, -3.0, 2.0])
    tgt = apply_affine(src, scale, bias)
    s, b = fit_affine_rgb(src.astype(np.float32), tgt.astype(np.float32))
    assert np.allclose(s, scale, atol=0.05)
    assert np.allclose(b, bias, atol=1.0)
