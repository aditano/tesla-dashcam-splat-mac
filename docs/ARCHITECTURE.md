# Architecture

## Overview

```
TeslaCam folder
    │
    ▼
SwiftUI app (DashSplat)  ──subprocess──►  dashsplat CLI (uv Python env)
    │                                         │
    │                                         ├─ events.py      group clips / event.json
    │                                         ├─ sei.py         Tesla protobuf SEI
    │                                         ├─ trajectory.py  GPS→ENU poses
    │                                         ├─ frames.py      ffmpeg / placeholders
    │                                         ├─ mapanything.py MLX MapAnything (Apache)
    │                                         ├─ yolo_mask.py   motion / YOLO masks
    │                                         └─ train_splat.py OpenSplat or seeded PLY
    ▼
WKWebView orbit viewer  ◄── splat.ply / splat.spz + cameras.json
```

## Trainer choice: OpenSplat

**Chosen primary trainer:** [WebODM/OpenSplat](https://github.com/WebODM/OpenSplat) (Metal/MPS).

| Option | Pros | Cons |
| --- | --- | --- |
| **OpenSplat** | Native Metal, `.ply` + `.spz`, COLMAP ingest, battle-tested on M1–M4 | Needs Xcode Metal toolchain; binary install |
| gsplat-mps / mpsplat | Python-native | AGPL via OpenSplat Metal port; less maintained |
| MetalSplat | MIT, pure Python Metal shaders | Newer; no `.spz` exporter yet |

**Fallback:** when `opensplat` is not on `PATH`, the pipeline seeds anisotropic Gaussians from MapAnything points and writes `.ply` + a stub `.spz`. CI and first-run smoke tests always exercise this path.

## MapAnything

Model: [`mlx-community/map-anything-apache-fp16`](https://huggingface.co/mlx-community/map-anything-apache-fp16) (Apache-2.0) via `mlx-vlm`.
Telemetry poses + HW3 intrinsics are passed when SEI is present; otherwise MapAnything estimates poses.

## First-run bootstrap

`dashsplat bootstrap` (also triggered by the app) creates:

`~/Library/Application Support/DashSplat/{venv,models,cache,jobs}`

## Expected runtime (Apple Silicon)

Rough guide for a ~60–120 s four-camera event on M2/M3 Pro:

| Stage | Time |
| --- | --- |
| SEI + frame extract | 5–20 s |
| MapAnything (4–24 views) | 2–15 s |
| YOLO / motion mask | 5–30 s |
| OpenSplat 1.5–3k iters | 2–8 min |
| Seeded fallback only | &lt; 10 s |

Longer SavedClips (multi-minute stitches) scale roughly linearly with selected frames.
