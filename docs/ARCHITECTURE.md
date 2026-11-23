# Architecture

## Overview

```
TeslaCam folder
    │
    ▼
SwiftUI app (DashSplat)  ──subprocess──►  dashsplat CLI (uv Python env)
    │                                         │
    │                                         ├─ events.py         group clips / event.json
    │                                         ├─ sei.py / detect.py  Tesla SEI + encrypted-clip check
    │                                         ├─ trajectory.py     GPS→ENU poses
    │                                         ├─ keyframes.py      distance keyframes, blur/turn skip
    │                                         ├─ tiles.py          ~100 m route tiles / 20 m overlap
    │                                         ├─ frames.py         ffmpeg + timing_offsets
    │                                         ├─ color_match.py    per-camera affine to front
    │                                         ├─ masks/            ego + skyseg + RF-DETR movers
    │                                         ├─ mapanything.py    MLX MapAnything (Apache; NC opt-in)
    │                                         ├─ trainers/         Brush (default) | msplat | seeded
    │                                         └─ bench.py          held-out proxy PSNR + stage timings
    ▼
WKWebView orbit viewer  ◄── splat.ply / splat.spz + cameras.json
Results library         ◄── jobs/<folder>/library/<event>/<preview|full>/
```

## Trainer choice (no OpenSplat)

OpenSplat was removed (AGPL-3.0). Trainers are pluggable via `dashsplat.trainers`:

| Trainer | License | Role |
| --- | --- | --- |
| **Brush** (default) | Apache-2.0 | Native Metal, image masks, splat cap |
| **msplat** | Apache-2.0 | Speed option (Metal) |
| **seeded** | in-repo | MapAnything-point fallback for CI / no-Metal |

**License clarification:** [zacsmms/mpsplat](https://github.com/zacsmms/mpsplat) is **Apache-2.0**. [iffyloop/gsplat-mps](https://github.com/iffyloop/gsplat-mps) is AGPL and is **not** used.

Bootstrap installs trainers into `~/Library/Application Support/DashSplat/{venv,bin}` on first run (same pattern as models).

## Quality tiers (Preview → Refine)

No modes picker beyond this flow:

| Tier | Keyframe spacing | Cap / cam | Train iters | Splat cap | ETA prior (Max-class, estimate) |
| --- | --- | --- | --- | --- | --- |
| Preview | ~1 / 3 m | 60 | 2500 | 300k | ~2 min / 1-min event |
| Full | ~1 / 1 m | 300 | 7000 | 1M | ~10–15 min / 1-min event |

ETA priors are labeled estimates; base chips are slower. The UI shows stage progress + remaining-time estimate. Route tiles are ~100 m with 20 m overlap.

## MapAnything

Default: [`mlx-community/map-anything-apache-fp16`](https://huggingface.co/mlx-community/map-anything-apache-fp16) (Apache-2.0).

Always passes HW3 intrinsics + SEI poses when available, 16-view windows mixing all 4 cameras, confidence filtering, per-tile init cloud. CC BY-NC weights are an **off-by-default** personal-use toggle (`--nc-mapanything`).

## Masks

1. Fixed per-camera HW3 ego-car masks  
2. Sky via `skyseg-coreml` (MIT) when available; geometric fallback otherwise  
3. Moving objects via RF-DETR (Apache-2.0); parked cars kept static  

No ultralytics / AGPL YOLO.

## First-run bootstrap

`dashsplat bootstrap` creates:

`~/Library/Application Support/DashSplat/{venv,models,cache,jobs,bin}`

## Expected runtime (Apple Silicon) — estimates only

Targets for a ~1-minute four-camera event on an **M-series Max** (base chips slower):

| Stage | Preview (est.) | Full (est.) |
| --- | --- | --- |
| Ingest + keyframes | seconds | seconds |
| Masks | seconds | tens of seconds |
| MapAnything | tens of seconds | tens of seconds |
| Train (Brush/msplat) | ~1–2 min total pipeline | ~10–15 min total pipeline |
| Seeded fallback only | &lt; 10 s | &lt; 10 s |

Numbers above are **estimates** from the 2026-11-23 research note, not measured CI timings. Use `dashsplat bench <folder> --event ID` for measured stage timings + held-out proxy PSNR.
