# Thin structures (trees / branches)

R44VC0RP/dashsplat’s posted results often show **clumpy trees**: branches exist but as large blobs. DashSplat Full tier defaults target that failure mode.

## Fixes (cited)

| Fix | Role | Source |
| --- | --- | --- |
| AbsGS absolute densification (`absgrad`) | Split over-reconstructed large Gaussians that plain avg-grad misses | https://arxiv.org/abs/2404.10484 |
| MCMC densification + splat cap | Controlled population; Brush “MCMC like” | https://arxiv.org/abs/2404.09591 ; Brush |
| Scale cap / prune_scale3d | Prevent huge primitives | gsplat DefaultStrategy |
| Anisotropy regularizer | Prefer thin axes on vegetation | PrismGS size-reg idea https://arxiv.org/abs/2510.07830 |
| Opacity reset | Classic 3DGS / gsplat `reset_every` | 3DGS |
| Antialiasing | Mip-Splatting-style AA when trainer supports it | https://arxiv.org/abs/2311.16493 |
| Higher-res Full frames | `target_width=1280` (HW3 encode) | docs/HW3.md |
| Vegetation splat boost | RAM-tuned per-tile budget | `detail_quality.effective_max_splats` |
| Sky / mover mask edge protect | Don’t eat tree edges against sky or wind parallax | `protect_vegetation_in_mask` |

These are **Full-tier defaults** in `python/dashsplat/detail_quality.py` — not user-facing modes. Brush/msplat receive them via `trainer_detail.json` / CLI flags (best-effort).

## Synthetic A/B (CI)

`tests/test_thin_structure.py` builds procedural branches and compares legacy isotropic 0.45 m scales vs Full refined scales. See `docs/screenshots/trees-before-after.png` and `trees-metrics.json` from the latest run.
