"""Full-tier fine-detail defaults for thin structures (trees / branches).

Problem: large isotropic Gaussians over-reconstruct high-frequency vegetation
("clumpy" trees) — also noted in R44VC0RP/dashsplat posted results.

Cited fixes wired as Full-tier trainer defaults (not user modes):
- AbsGS absolute / homodirectional densification gradients
  https://arxiv.org/abs/2404.10484
- MCMC densification with splat cap (3DGS-MCMC; Brush "MCMC like")
  https://arxiv.org/abs/2404.09591
- Scale prune / grow thresholds (gsplat DefaultStrategy prune_scale3d)
  https://github.com/nerfstudio-project/gsplat
- Opacity reset (classic 3DGS / gsplat reset_every)
- Mip-Splatting-style antialiasing / 3D frequency constraint
  https://arxiv.org/abs/2311.16493
- Size / anisotropy regularization against huge blobs (PrismGS-style size reg)
  https://arxiv.org/abs/2510.07830

Seeded fallback applies scale-cap + branch-aligned anisotropy so CI can
regression-test thin-structure metrics without Metal trainers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class DetailQuality:
    densify_mode: str  # "mcmc" | "absgrad" | "legacy"
    use_absgrad: bool
    max_scale_m: float
    prune_scale3d: float
    grow_grad2d: float
    anisotropy_reg: float
    opacity_reset_every: int
    antialiasing: bool
    target_width: int
    train_iterations: int
    high_freq_extra_iters: int
    max_splats_base: int
    vegetation_splat_boost: float
    sky_mask_erode_px: int
    mover_mask_erode_px: int
    protect_vegetation_edges: bool

    SOURCES: tuple[str, ...] = (
        "https://arxiv.org/abs/2404.10484",  # AbsGS
        "https://arxiv.org/abs/2404.09591",  # 3DGS-MCMC
        "https://arxiv.org/abs/2311.16493",  # Mip-Splatting
        "https://arxiv.org/abs/2510.07830",  # PrismGS size reg
        "https://github.com/nerfstudio-project/gsplat",
        "https://github.com/ArthurBrussee/brush",
        "https://github.com/R44VC0RP/dashsplat",
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = list(self.SOURCES)
        return d

    def trainer_extra(self) -> dict[str, Any]:
        """Flags passed to Brush/msplat CLI / config JSON."""
        return {
            "densify": self.densify_mode,
            "absgrad": self.use_absgrad,
            "max_scale": self.max_scale_m,
            "prune_scale3d": self.prune_scale3d,
            "grow_grad2d": self.grow_grad2d,
            "anisotropy_reg": self.anisotropy_reg,
            "opacity_reset_every": self.opacity_reset_every,
            "antialias": self.antialiasing,
            "mcmc": self.densify_mode == "mcmc",
        }


# Preview: fast, coarser — still mild scale cap so we don't regress wildly.
PREVIEW_DETAIL = DetailQuality(
    densify_mode="mcmc",
    use_absgrad=False,
    max_scale_m=0.35,
    prune_scale3d=0.12,
    grow_grad2d=0.0002,
    anisotropy_reg=0.01,
    opacity_reset_every=3000,
    antialiasing=True,
    target_width=640,
    train_iterations=2500,
    high_freq_extra_iters=0,
    max_splats_base=300_000,
    vegetation_splat_boost=1.0,
    sky_mask_erode_px=2,
    mover_mask_erode_px=1,
    protect_vegetation_edges=True,
)

# Full: thin-structure defaults (trees). Higher res, AbsGS+MCMC, tight scale cap,
# vegetation splat boost, extra iters for high-frequency regions.
FULL_DETAIL = DetailQuality(
    densify_mode="mcmc",
    use_absgrad=True,  # AbsGS absgrad=True; gsplat recommends grow_grad2d≈0.0008
    max_scale_m=0.12,
    prune_scale3d=0.08,
    grow_grad2d=0.0008,
    anisotropy_reg=0.05,
    opacity_reset_every=3000,
    antialiasing=True,
    target_width=1280,  # Full-res HW3 encode (cited 1280×960)
    train_iterations=7000,
    high_freq_extra_iters=2000,  # extra steps after main densify window
    max_splats_base=1_200_000,  # vegetation needs headroom vs 1M
    vegetation_splat_boost=1.5,
    sky_mask_erode_px=4,
    mover_mask_erode_px=3,
    protect_vegetation_edges=True,
)

# Legacy "clumpy" baseline for A/B tests (plain large isotropic scales).
LEGACY_CLUMPY = DetailQuality(
    densify_mode="legacy",
    use_absgrad=False,
    max_scale_m=0.8,
    prune_scale3d=0.25,
    grow_grad2d=0.0002,
    anisotropy_reg=0.0,
    opacity_reset_every=0,
    antialiasing=False,
    target_width=640,
    train_iterations=3000,
    high_freq_extra_iters=0,
    max_splats_base=400_000,
    vegetation_splat_boost=1.0,
    sky_mask_erode_px=0,
    mover_mask_erode_px=0,
    protect_vegetation_edges=False,
)


def detail_for_tier(tier: str) -> DetailQuality:
    t = str(tier).lower()
    if t == "full":
        return FULL_DETAIL
    if t == "legacy" or t == "clumpy":
        return LEGACY_CLUMPY
    return PREVIEW_DETAIL


def effective_iterations(detail: DetailQuality) -> int:
    return int(detail.train_iterations + detail.high_freq_extra_iters)


def effective_max_splats(detail: DetailQuality, ram_gb: float) -> int:
    """Per-tile / scene splat budget: vegetation boost × RAM scale."""
    mem_scale = min(2.5, max(0.6, ram_gb / 36.0))
    return int(detail.max_splats_base * detail.vegetation_splat_boost * mem_scale)


def synthesize_branch_cloud(
    n_branches: int = 5,
    pts_per: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Procedural thin branches: means, colors, unit tangents."""
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    for _ in range(n_branches):
        origin = rng.uniform(-2, 2, size=3)
        origin[2] = rng.uniform(0.5, 3.0)  # up
        direction = rng.normal(size=3)
        direction[2] = abs(direction[2]) + 0.3
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        # Slight curve
        for i in range(pts_per):
            t = i / max(pts_per - 1, 1)
            bend = 0.15 * np.sin(t * np.pi) * np.array([-direction[1], direction[0], 0.0])
            p = origin + direction * (t * 2.5) + bend
            means.append(p)
            tangents.append(direction + bend * 0.2)
            # Bark / leaf colors
            if t > 0.55:
                colors.append(np.array([0.25, 0.55, 0.2]))
            else:
                colors.append(np.array([0.35, 0.22, 0.12]))
    m = np.asarray(means, dtype=np.float32)
    c = np.asarray(colors, dtype=np.float32)
    tan = np.asarray(tangents, dtype=np.float32)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-8
    return m, c, tan


def legacy_clumpy_scales(means: np.ndarray) -> np.ndarray:
    """Old-style large isotropic scales that blob thin structures."""
    n = means.shape[0]
    # ~0.45 m blobs — eats branch thickness
    return np.full((n, 3), 0.45, dtype=np.float32)


def refine_thin_structure_scales(
    means: np.ndarray,
    tangents: np.ndarray | None,
    detail: DetailQuality,
) -> np.ndarray:
    """Anisotropic scales: long along tangent, capped cross-section (AbsGS/Mip spirit)."""
    n = means.shape[0]
    # Neighbour spacing → base cross-section
    if n > 1:
        sample = means[: min(1500, n)]
        d = np.linalg.norm(sample[None, :, :] - sample[:, None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = np.min(d, axis=1)
        med = float(np.median(nn[np.isfinite(nn)])) if np.any(np.isfinite(nn)) else 0.03
    else:
        med = 0.03
    cross = float(np.clip(med * 0.85, 0.008, detail.max_scale_m * 0.35))
    along = float(np.clip(med * 2.2, cross * 1.5, detail.max_scale_m))
    scales = np.zeros((n, 3), dtype=np.float32)
    if tangents is None:
        scales[:, :] = cross
        scales[:, 0] = along
    else:
        # Assign largest axis as along-tangent magnitude; other two = cross
        scales[:, 0] = along
        scales[:, 1] = cross
        scales[:, 2] = cross * (1.0 - 0.3 * detail.anisotropy_reg)
    # Hard cap (prune_scale3d / max_scale)
    scales = np.clip(scales, 1e-4, detail.max_scale_m)
    return scales


def _skeleton_mask(means: np.ndarray, size: int = 128) -> np.ndarray:
    """1px orthographic branch skeleton (XY)."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    xy = means[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    margin = 0.08
    pts = []
    for i in range(means.shape[0]):
        u = (xy[i, 0] - lo[0]) / span[0]
        v = (xy[i, 1] - lo[1]) / span[1]
        px = int((margin + u * (1 - 2 * margin)) * (size - 1))
        py = int((margin + (1 - v) * (1 - 2 * margin)) * (size - 1))
        pts.append((px, py))
        draw.point((px, py), fill=255)
    # Connect sequential points within a branch-ish distance
    for a, b in zip(pts, pts[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < size * 0.08:
            draw.line([a, b], fill=255, width=1)
    return np.asarray(img, dtype=np.float32)


def sharpness_proxy_2d(
    means: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    *,
    size: int = 128,
) -> dict[str, float]:
    """Thin-structure fidelity: skeleton precision + edge energy + soft PSNR proxy.

    Clumpy large scales bleed off the branch skeleton (low precision). Refined
    scales hug the skeleton and raise held-out sharpness vs a blurred reference.
    """
    img = render_scale_crop(means, scales, colors, size=size)
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    skel = _skeleton_mask(means, size=size)
    mass = arr > 20
    skel_bin = skel > 0
    # Precision of splat mass vs thin skeleton (higher = less clump bleed)
    precision = float((mass & skel_bin).sum() + 1e-3) / float(mass.sum() + 1e-3)
    # Laplacian edge energy *on skeleton neighbourhood*
    padded = np.pad(arr, 1, mode="edge")
    lap = (
        -4 * arr
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
    )
    # Weight laplacian by proximity to skeleton (dilate skel)
    from PIL import ImageFilter as PILFilter

    skel_d = np.asarray(
        Image.fromarray(skel.astype(np.uint8), mode="L").filter(PILFilter.MaxFilter(5)),
        dtype=np.float32,
    )
    lap_var = float(np.var(lap[skel_d > 0])) if np.any(skel_d > 0) else float(np.var(lap))
    # Composite sharpness: precision-weighted edge energy
    sharpness = lap_var * (0.25 + precision)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=2)).convert("L"), dtype=np.float32)
    # Held-out: compare to a thin skeleton target image
    target = skel
    # Normalize render to 0-255 mass
    render_n = arr
    mse = float(np.mean((render_n / 255.0 - target / 255.0) ** 2)) + 1e-6
    psnr = float(10.0 * np.log10(1.0 / mse))
    return {
        "laplacian_var": sharpness,
        "raw_laplacian_var": lap_var,
        "skeleton_precision": precision,
        "psnr_proxy_db": psnr,
        "mse_vs_skeleton": mse,
    }


def render_scale_crop(
    means: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    *,
    size: int = 256,
) -> Image.Image:
    """Simple orthographic Gaussian blob render (XY) for visual A/B."""
    img = Image.new("RGB", (size, size), (18, 22, 16))
    draw = ImageDraw.Draw(img)
    xy = means[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    # Map to pixels with margin
    margin = 0.08
    for i in range(means.shape[0]):
        u = (xy[i, 0] - lo[0]) / span[0]
        v = (xy[i, 1] - lo[1]) / span[1]
        px = int((margin + u * (1 - 2 * margin)) * (size - 1))
        py = int((margin + (1 - v) * (1 - 2 * margin)) * (size - 1))
        # Radius from max scale (world → approx pixels)
        rad = float(np.max(scales[i]) / float(np.max(span)) * size * 0.5)
        rad = max(0.6, min(rad, size * 0.2))
        r = int(np.clip(colors[i, 0] * 255, 0, 255))
        g = int(np.clip(colors[i, 1] * 255, 0, 255))
        b = int(np.clip(colors[i, 2] * 255, 0, 255))
        draw.ellipse(
            [px - rad, py - rad, px + rad, py + rad],
            fill=(r, g, b),
        )
    return img


def protect_vegetation_in_mask(
    rgb_path,
    mask: np.ndarray,
    *,
    erode_px: int = 3,
) -> np.ndarray:
    """Keep green vegetation pixels that sky/mover masks would ignore.

    Erodes the ignore region (0) so tree edges against sky stay in the keep set.
    """
    from PIL import Image as PILImage
    from PIL import ImageFilter as PILFilter

    out = np.array(mask, dtype=np.uint8, copy=True)
    if erode_px > 0:
        # Dilate keep (255) = erode ignore (0)
        keep = PILImage.fromarray(out, mode="L")
        for _ in range(erode_px):
            keep = keep.filter(PILFilter.MaxFilter(3))
        out = np.array(keep, dtype=np.uint8, copy=True)
    # Force-keep green-ish pixels (foliage)
    arr = np.asarray(PILImage.open(rgb_path).convert("RGB"), dtype=np.float32)
    if arr.shape[:2] != out.shape:
        arr = np.asarray(
            PILImage.open(rgb_path).convert("RGB").resize((out.shape[1], out.shape[0]))
        ).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    foliage = (g > r * 1.05) & (g > b * 1.05) & (g > 40) & (g < 220)
    out[foliage] = 255
    return out
