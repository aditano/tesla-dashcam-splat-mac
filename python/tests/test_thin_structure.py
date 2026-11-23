"""Failing-first: thin structures (trees/branches) must not get clumpy scales."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dashsplat.detail_quality import (
    DetailQuality,
    detail_for_tier,
    legacy_clumpy_scales,
    refine_thin_structure_scales,
    sharpness_proxy_2d,
    synthesize_branch_cloud,
)


def test_full_tier_enables_absgs_mcmc_scale_cap_aa():
    d = detail_for_tier("full")
    assert d.densify_mode in ("mcmc", "absgrad")
    assert d.use_absgrad is True
    assert d.max_scale_m < 0.5
    assert d.anisotropy_reg > 0
    assert d.opacity_reset_every > 0
    assert d.antialiasing is True
    assert d.target_width >= 1280
    assert d.high_freq_extra_iters > 0
    assert d.vegetation_splat_boost >= 1.2


def test_preview_is_lighter_than_full():
    p = detail_for_tier("preview")
    f = detail_for_tier("full")
    assert f.max_splats_base >= p.max_splats_base
    assert f.train_iterations >= p.train_iterations
    assert f.target_width >= p.target_width


def test_thin_branch_max_scale_and_sharpness_improve(tmp_path: Path):
    """Synthetic branches: refined scales beat legacy clumpy isotropic blobs."""
    means, colors, tangents = synthesize_branch_cloud(n_branches=5, pts_per=80, seed=0)
    assert means.shape[0] > 100

    old_scales = legacy_clumpy_scales(means)
    new_scales = refine_thin_structure_scales(
        means, tangents, detail_for_tier("full")
    )

    old_max = float(np.max(old_scales))
    new_max = float(np.max(new_scales))
    assert new_max < old_max
    assert new_max <= detail_for_tier("full").max_scale_m + 1e-6

    # Anisotropy: longest axis along tangent, min axis much smaller (thin)
    new_min_axis = float(np.median(np.min(new_scales, axis=1)))
    old_min_axis = float(np.median(np.min(old_scales, axis=1)))
    assert new_min_axis < old_min_axis

    old_sharp = sharpness_proxy_2d(means, old_scales, colors)
    new_sharp = sharpness_proxy_2d(means, new_scales, colors)
    assert new_sharp["skeleton_precision"] > old_sharp["skeleton_precision"]
    assert new_sharp["laplacian_var"] > old_sharp["laplacian_var"]
    assert new_sharp["psnr_proxy_db"] > old_sharp["psnr_proxy_db"]

    # Persist before/after crops for the PR report
    from dashsplat.detail_quality import render_scale_crop

    before = render_scale_crop(means, old_scales, colors, size=256)
    after = render_scale_crop(means, new_scales, colors, size=256)
    # Side-by-side — always write under pytest tmp; optionally mirror to agent artifacts
    from PIL import Image

    combo = Image.new("RGB", (512, 256))
    combo.paste(before, (0, 0))
    combo.paste(after, (256, 0))
    before.save(tmp_path / "trees-before-clumpy.png")
    after.save(tmp_path / "trees-after-thin.png")
    combo.save(tmp_path / "trees-before-after.png")
    docs = Path(__file__).resolve().parents[2] / "docs" / "screenshots"
    if docs.is_dir():
        combo.save(docs / "trees-before-after.png")

    agent_out = Path("/opt/cursor/artifacts")
    try:
        agent_out.mkdir(parents=True, exist_ok=True)
        before.save(agent_out / "trees-before-clumpy.png")
        after.save(agent_out / "trees-after-thin.png")
        combo.save(agent_out / "trees-before-after.png")
    except OSError:
        pass  # GitHub Actions runners cannot write /opt/cursor

    # Write metrics JSON for the report
    import json

    metrics = {
        "old_max_scale_m": old_max,
        "new_max_scale_m": new_max,
        "old_median_min_axis_m": old_min_axis,
        "new_median_min_axis_m": new_min_axis,
        "old_laplacian_var": old_sharp["laplacian_var"],
        "new_laplacian_var": new_sharp["laplacian_var"],
        "old_skeleton_precision": old_sharp["skeleton_precision"],
        "new_skeleton_precision": new_sharp["skeleton_precision"],
        "old_psnr_proxy_db": old_sharp["psnr_proxy_db"],
        "new_psnr_proxy_db": new_sharp["psnr_proxy_db"],
        "sources": list(DetailQuality.SOURCES),
    }
    (tmp_path / "trees-metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        (agent_out / "trees-metrics.json").write_text(json.dumps(metrics, indent=2))
    except OSError:
        pass
    assert metrics["new_skeleton_precision"] > metrics["old_skeleton_precision"]
    assert metrics["new_psnr_proxy_db"] > metrics["old_psnr_proxy_db"]
