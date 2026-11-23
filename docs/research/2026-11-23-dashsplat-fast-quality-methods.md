# DashSplat (Mac): fast, high-quality methods for TeslaCam to 3D Gaussian splats

Date: 2026-11-23 (ET). Author: Grok Bot (Bill, engineering) for Anthony.
Repo reviewed: github.com/aditano/tesla-dashcam-splat-mac @ 69086ac (event grouping, SEI to ENU poses, MapAnything MLX, OpenSplat Metal, seeded fallback, motion-blob mask fallback, dynamics stubbed).
Extends: /workspace/splat-ref/2026-11-22-dashcam-splat-ai-methods.md (Tesla Reality View vs generative splatting, depth model survey, feed-forward driving models such as DGGT). That material is not repeated here.

Rules used: every claim has a live URL checked on 2026-11-23. Licenses were read from the GitHub API or the model card on 2026-11-23. Timing numbers are quoted from their source with the hardware named. Anything I computed myself is labelled "estimate". Nothing here was run on Anthony's Mac.

---

## 0. Bottom line

1. Someone already built almost exactly this, two days ago: Ryan Vogel's **dashsplat** (R44VC0RP/dashsplat, Apache-2.0 code, created 2026-11-22). It runs on a Mac Studio (M5 Ultra), uses all six HW4 cameras plus SEI telemetry, and documents every stage with timings. It is slow (hours per minute of driving for the full tier) because it chases centimetre road accuracy and a diffusion chase view. It is the best "what works and what does not" reference for us. Note the name clash: our app is also called DashSplat.
2. The fastest documented Mac trainers are now native Metal and permissive: **msplat** (Apache-2.0, about 59 to 80 s for 7k iterations on full-res Mip-NeRF 360 scenes on an M4 Max), **Brush** (Apache-2.0, native Metal training merged 2026-11-21, supports image masks), **mpsplat** (Apache-2.0, the gsplat API on MPS). OpenSplat, our current trainer, is **AGPL-3.0** and its published Mac speed is about 3 minutes for 2,000 steps on an M2 laptop. It should be replaced.
3. Pose-conditioned geometry is the key to quality with only 4 low-overlap cameras. Ryan measured that every monocular "metric" depth model was off by 15 to 70 % on Tesla footage, while MapAnything given poses was within 1 to 8 % on every camera. We already pass telemetry poses to MapAnything; keep that and make it the backbone.
4. Recommended v1 target per 1-minute event: Preview in about 2 minutes, Full in about 10 to 15 minutes, on an M-series Max-class Mac. Both are estimates built from cited per-stage numbers and must be measured on Anthony's Mac (model not yet known to me).

---

## 1. What other people are doing (new since the 11-22 doc)

### 1A. Tesla-specific projects

| Who / what | Method | Hardware and time (as published) | License | Source |
|---|---|---|---|---|
| **Ryan Vogel, dashsplat** (Nov 21 to 22, 2026) | Ingest 6 cams + SEI, dead-reckon track; self-calibrate lenses, mounts, per-camera clock offset and rolling-shutter readout with Ceres BA; poses; road atlas; depth from pose-conditioned DA3-LARGE-1.1 (front/rear) and MapAnything (pillars/repeaters); Mask2Former sky/road/vehicle masks; splat fit with MetalSplat on Metal; vehicle tracks; optional Difix chase view | Preview tier 13.6 min and full geometry 51.9 min per minute of moving footage on an M5 Ultra; L40S port 32.9 and 82.9 min. Dead-reckoned track ends 7.5 m from GPS after 3.3 km | Code Apache-2.0. Several default models are non-commercial (Mask2Former Cityscapes, MapAnything CC BY-NC, Difix) | https://github.com/R44VC0RP/dashsplat ; https://github.com/R44VC0RP/dashsplat/blob/main/THIRD_PARTY.md ; https://github.com/R44VC0RP/dashsplat/blob/main/pipeline/run/README.md |
| Ryan's depth bake-off | Posed DA3-LARGE-1.1 front: scale 1.04 to 1.05, 0.041 s/image in a 6-image call. Posed MapAnything on repeaters: shape error 5.5 % vs 14.8 % for DA3-LARGE, 0.115 s/image. Mask2Former Swin-L 0.10 s/image | Mac Studio M5 Ultra | as above | https://github.com/R44VC0RP/dashsplat/blob/main/pipeline/depth/REPORT.md |
| Ryan's calibration finding | Per-camera clock offsets and rolling-shutter readout are real and measurable from footage: "For row v, use t + offset + (v/height - 0.5) x readout" | M5 Ultra, CPU Ceres | Apache-2.0 | https://github.com/R44VC0RP/dashsplat/blob/main/pipeline/calib/REPORT.md |
| Ryan's warnings | It "knows one car" (HW4 Model 3); HW3's four cameras are untested. Needs SEI (firmware 2025.44.25+). Since 2026.20 Tesla encrypts dashcam clips by default in most markets; decrypt in the car or at dashcam.tesla.com first | n/a | n/a | https://github.com/R44VC0RP/dashsplat (README, "What it does not do") |
| Reddit r/GaussianSplatting, "I did reconstruct a street in 3D using only Tesla's cameras" | Model 3, 4 angles; poses and splatting in Spirula Studio. "Biggest challenge: getting the camera positions to align... little overlap" | not stated | n/a | https://www.reddit.com/r/GaussianSplatting/comments/1x11vmc/i_did_reconstruct_a_street_in_3d_using_only/ |
| Barazzetti et al., ISPRS 2026, "3D Modeling and Rendering with a Tesla Model 3 Highland" | Calibrate the cameras as a rig with bundle adjustment, then mesh and splat. Artefacts from "limited acquisition geometry, temporal desynchronization, compression, and dynamic scene elements" | not stated in abstract | Open access paper | https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/ |
| DC-Gaussian / RIDE-Splat | Model windshield reflections (front camera behind glass) as a separate layer and drop it at render time | CUDA | DC-Gaussian code on GitHub; RIDE-Splat code + dataset | https://arxiv.org/abs/2405.17705 ; https://github.com/ridesplat/ride-splat |

Take-aways for us: (a) poses from telemetry plus pose-conditioned geometry beat monocular depth; (b) per-camera time offset and rolling shutter matter; (c) masks for sky, ego car and movers are standard; (d) the expensive parts of Ryan's pipeline (cm road atlas, 1.4M-splat tiles, diffusion view) are what make it take an hour per minute. We can skip them for v1.

### 1B. Mac trainers (Apple Silicon, no CUDA)

| Trainer | Published speed / quality | Features relevant to driving | License | Apple Silicon | Source |
|---|---|---|---|---|---|
| **msplat** (rayanht) v1.1.4 | M4 Max, 7k iters, full res, no downscale: bicycle 59 s, counter 80 s, garden 77 s, room 74 s. Default 2 downscales: garden 36 s (about 0.8 dB lower). Garden 30k: 700 s, 27.14 dB, 3.51M gaussians | Fused Metal shaders for the whole loop; Python, Swift package and C++ CLI; COLMAP / Nerfstudio / Polycam input. Loss masks, depth loss, exposure: not documented | Apache-2.0 | Yes, Metal only | https://github.com/rayanht/msplat ; https://pypi.org/project/msplat ; https://news.ycombinator.com/item?id=47262641 |
| **Brush** (ArthurBrussee) | v0.3.0 (Sep 2026) "MCMC like" training, max-splat cap, LPIPS loss; "Use native CubeCL Metal for macOS training" merged 2026-11-21. Repo states training generally faster than gsplat (kernel benchmarks, not Mac driving) | Image masks folder ("Black pixels... are ignored"), max splat count, datasets bigger than RAM, CLI, Apple Silicon download | Apache-2.0 | Yes | https://github.com/ArthurBrussee/brush ; https://github.com/ArthurBrussee/brush/releases/ ; https://github.com/ArthurBrussee/brush/commits/main |
| **mpsplat** (zacsmms) | Author: real scenes 5 to 15x faster than pure PyTorch; "roughly 2-4x" slower than upstream gsplat on NVIDIA | Same `import gsplat` API on `device="mps"`: 3DGS and 2DGS rasterizers, 3DGUT pinhole/OpenCV-distortion/fisheye projection in Metal. LiDAR and windshield distortion removed. Very new (5 stars) | Apache-2.0 | Yes (M1 to M4 stated) | https://github.com/zacsmms/mpsplat |
| **metal-gauss** (nandometzger) | Best 8-scene mean PSNR within a budget on NeRF-synthetic: 1 min 24.7 dB, 3 min 29.8, 6 min 31.3 (msplat 20.8 / 22.1 / 22.3, Brush 14.4 / 18.4 / 23.4). Object scenes, not unbounded street scenes | Credits 3DGS-MCMC, Taming 3DGS, Speedy-Splat; COLMAP input; viewer | MIT | Yes (M5 benchmarks) | https://github.com/nandometzger/metal-gauss |
| MetalSplat (tchauffi) | Used by Ryan's dashsplat for its splat fit on the Mac | Differentiable fit on Apple GPUs | MIT | Yes | https://github.com/tchauffi/metalsplat ; https://github.com/R44VC0RP/dashsplat/blob/main/THIRD_PARTY.md |
| OpenSplat (current) | M3 Pro 100 iters 5.9 s (GPU) vs 5 min 26 s (CPU); about 3 min for 2,000 steps on an M2 laptop; about 2 GB GPU memory per million gaussians | COLMAP/nerfstudio input, .ply/.splat/.spz output | **AGPL-3.0** | Yes (MPS) | https://github.com/WebODM/OpenSplat/ ; https://github.com/WebODM/OpenSplat/pull/76 ; https://radiancefields.com/opensplat-adds-mac-gpu-acceleration |
| gsplat-mps (iffyloop) | old gsplat 0.1.3 port | | AGPL-3.0 (derived from OpenSplat Metal) | Yes | https://github.com/iffyloop/gsplat-mps |
| 3D Splat App (Laan Labs, commercial app) | Fast / Medium / HD presets "roughly 2, 5 and 10 to 20 minutes" per scene; C++ and Metal, built-in COLMAP on Metal | Masks, SPZ/SOG export | Commercial product | Yes | https://labs.laan.com/blogs/gaussian-splat-training-on-mac-with-metal |
| Scantic experiment (radiancefields.com) | MacBook: Brush 30k baseline 12,381 s; msplat-based "Quality" 63.3 s at 25.80 dB vs 26.87 dB baseline. msplat "around five times faster per training step than the Brush setup" (pre native-Metal Brush) | | article | Yes | https://radiancefields.com/i-tried-to-make-gaussian-splatting-1-000%C3%97-faster |

Correction to our repo's docs/ARCHITECTURE.md: it says mpsplat is "AGPL via OpenSplat Metal port". The mpsplat repo is Apache-2.0 (GitHub API, 2026-11-23); it is gsplat-mps that is AGPL. OpenSplat itself is AGPL.

### 1C. Quality techniques that matter for street scenes (with Mac availability)

| Technique | Evidence | On Mac today? | Source |
|---|---|---|---|
| MCMC densification with a splat cap | gsplat AV trainer on NCore: +1.5 dB from MCMC, +1.4 dB from SH3 (A40 GPU, 10k steps, 6.1 to 26.8 min) | Brush has "MCMC like" + cap; mpsplat inherits gsplat MCMCStrategy (not verified on MPS) | https://github.com/nerfstudio-project/gsplat/blob/main/examples/AV_TRAINER.md ; https://github.com/ArthurBrussee/brush/releases/ |
| Per-image / per-camera appearance (exposure, ISP) | gsplat `post_processing="bilateral_grid"` (one grid per frame) and `"ppisp"` (per camera and per frame, NVIDIA PPISP, Apache-2.0) | Only via mpsplat path (bilateral grid is PyTorch; PPISP on MPS not verified). Otherwise do a per-camera color pre-normalization | https://raw.githubusercontent.com/nerfstudio-project/gsplat/refs/heads/main/examples/simple_trainer.py ; https://github.com/nv-tlabs/ppisp |
| Rolling shutter + per-camera time offset | gsplat 3DGUT takes `rolling_shutter`; SplatAD fork has per-camera readout (Apache-2.0); Ryan measured offsets per lens | No Metal trainer with rolling shutter found by Ryan's survey. On Mac: fix timing in data (frame selection / pose interpolation) | https://github.com/R44VC0RP/dashsplat/blob/main/research/03-offline-high-accuracy.md ; https://github.com/carlinds/splatad |
| Sky masking | Mask2Former used by Ryan (non-commercial weights). Small Core ML sky models exist | Yes: `melmass/skyseg-coreml` (MIT), `twistlabs/fotometis-sky` (MIT, 2.5 MB) | https://huggingface.co/melmass/skyseg-coreml ; https://huggingface.co/twistlabs/fotometis-sky |
| Mover masks / tracks | Ryan: Mask2Former + IoU tracking, moving vs parked from depth + ego motion. Apache detector alternative: RF-DETR (incl. segmentation). Promptable refine: SAM 2 (Core ML from Apple) | Yes (RF-DETR on MPS not timed; SAM 2 Core ML official) | https://github.com/roboflow/rf-detr ; https://huggingface.co/apple/coreml-sam2-tiny ; https://github.com/facebookresearch/sam2 |
| Dynamic objects as separate rigid splat sets | OmniRe / DriveStudio (MIT), Street Gaussians; open-revisualizer (MIT) shows "background Gaussians plus 45 moving vehicles" on a 7-camera + LiDAR rig | CUDA codebases; concept portable | https://github.com/ziyc/drivestudio ; https://github.com/waleedsworld/open-revisualizer |
| Fast multi-camera rig recon | "On-the-fly Large-scale 3D Reconstruction from Multi-Camera Rigs": "hundreds of meters of 3D scenes within just 2 minutes" from raw multi-camera video (abstract; no official code found) | Concept only | https://arxiv.org/abs/2512.08498 |
| Rig-aware feed-forward from calibrated multi-camera video | RigRecon (ZJU + Xiaomi EV, 2026): >1000 images per pass, optional Gaussian heads via gsplat, weights on HF | Code Apache-2.0; CUDA; Mac port not verified | https://github.com/zju3dv/RigRecon |
| Windshield reflection layer (front cam) | DC-Gaussian, RIDE-Splat | CUDA research code | https://github.com/ridesplat/ride-splat |

### 1D. Geometry on Mac (numbers)

- MapAnything Apache, MLX fp16 (2.65 GB), full `infer` on an Apple M5 Max at 518x294: 4 views 0.23 s, 8 views 0.48 s, 16 views 1.10 s, 32 views 2.89 s (PyTorch MPS 5.27 s for 16). Accepts calibration, depth and pose per view. Apache-2.0. The CC BY-NC variant is "recommended by the authors for the best accuracy" (non-commercial). https://huggingface.co/mlx-community/map-anything-apache-fp16
- Ryan used the CC BY-NC `facebook/map-anything` weights. Our Apache MLX weights are a different data mix, so his accuracy numbers are an upper reference, not a promise. https://github.com/R44VC0RP/dashsplat/blob/main/THIRD_PARTY.md
- DA3 licenses: DA3-SMALL/BASE Apache-2.0; DA3-LARGE-1.1 has conflicting labels (card Apache, official README CC BY-NC); Ryan treats it as non-commercial. Same source.

---

## 2. Recommended DashSplat v1 pipeline

Two tiers, same pipeline, different budgets. Everything Apache/MIT by default. Non-commercial models only behind an "Extra quality (personal use)" toggle.

```
TeslaCam folder
 -> 1 Ingest: events (done), SEI per frame (done), encrypted-clip + no-SEI detection (new)
 -> 2 Trajectory: SEI -> ENU (done) + smoothing; per-camera time offset table (new)
 -> 3 Keyframes: distance-based per camera, skip high yaw-rate and blurred frames (new)
 -> 4 Masks: static ego-car mask per HW3 camera + sky (Core ML) + movers (RF-DETR seg + ego-motion test) (new)
 -> 5 Geometry: MapAnything-apache MLX, 16-view windows mixing all 4 cameras, K + telemetry poses + metric flag (exists, upgrade)
 -> 6 Fuse: confidence filter, voxel downsample, drop masked pixels -> init.ply per tile (new)
 -> 7 Tiles: split route into ~100 m tiles with 20 m overlap (new)
 -> 8 Train: Brush (default) or msplat, masks folder, MCMC cap; per-camera color pre-normalization (swap)
 -> 9 Merge tiles, export .ply/.spz, cameras.json, viewer (exists)
 -> 10 v1.1: movers as rigid per-track splat sets on top (later)
```

### Budget targets per 1-minute, 4-camera event (Max-class M-series)

| Stage | Preview target | Full target | Basis |
|---|---|---|---|
| Ingest + decode keyframes | 10 to 20 s | 20 to 40 s | repo ARCHITECTURE.md guide "5-20 s"; estimate |
| Keyframes | 1 frame per 3 m per camera, cap 60/camera (240 views), 640 px | 1 per 1 m, cap 300/camera (1200 views), full 1280x960 | estimate |
| Masks | sky + ego only (~seconds) | + movers. Mask2Former was 0.10 s/image on M5 Ultra; RF-DETR expected same order, not timed: about 2 min for 1200 views | Ryan depth/masks report; estimate |
| MapAnything | 240 views in 16-view windows, stride 8: ~30 calls x 1.10 s = about 35 s on M5 Max | 1200 views: ~150 calls x 1.10 s = about 3 min on M5 Max | HF card timings; call counts are my estimate |
| Train | 1 tile pass, 2k to 3k iters at 1/4 to 1/2 res, cap 300k splats: under 60 s | ~100 m tiles, 7k iters each, cap ~1M splats per tile. msplat 7k full-res Mip-NeRF scenes took 59 to 80 s on M4 Max, so ~1 to 1.5 min per tile; an 800 m minute of driving = ~8 tiles = about 8 to 12 min | msplat README; tile count is an estimate (depends on speed) |
| **Total** | **about 2 min** | **about 10 to 15 min** | estimate, must be measured |

Scale notes: a stationary or slow event has far fewer keyframes (distance-based), so it gets faster, not slower. Base (non-Pro/Max) M chips have fewer GPU cores, so expect a multiple of these times; the app should show an ETA from a quick benchmark on first run.

Quality gates (reuse from 11-22 doc, adapted): hold out every 8th keyframe; Full must beat Preview by at least 2 dB PSNR; no sky floaters; lane width 3.0 to 3.9 m; no ghost cars in an orbit 3 to 5 m off the path.

---

## 3. Prioritized changes to aditano/tesla-dashcam-splat-mac

| # | Change | Why | Integrate | License |
|---|---|---|---|---|
| **1** | **Replace OpenSplat with a pluggable trainer; default Brush CLI, second backend msplat.** `train_splat.py` gets a `Trainer` protocol; COLMAP writer stays. Pass `masks/` folder, `--max-splats`, iteration count per tier. Bundle the Brush release binary for Apple Silicon. Add msplat via `pip install msplat` (or its Swift package for in-app preview) and keep whichever wins the bake-off (#5). Keep seeded fallback. | AGPL removal; Brush has masks + MCMC + cap today; msplat has the best published Mac wall times | github.com/ArthurBrussee/brush (releases), github.com/rayanht/msplat | Both Apache-2.0 |
| **2** | **Real masks stage.** (a) Static ego-car masks per HW3 camera (repeaters see the body, back sees bumper), auto-built from per-pixel temporal variance and shipped as defaults. (b) Sky via `melmass/skyseg-coreml`. (c) Movers via RF-DETR segmentation + "moving" test from ego motion and MapAnything depth; parked cars stay static. Replace the motion-blob fallback. Do not add ultralytics YOLO (AGPL-3.0). | Sky floaters, ego-car smear and ghost cars are the top visible artifacts; Ryan masks all three | huggingface.co/melmass/skyseg-coreml, github.com/roboflow/rf-detr, optional huggingface.co/apple/coreml-sam2-tiny | MIT, Apache-2.0, Apache-2.0 |
| **3** | **Distance-based keyframes, route tiling, and Preview/Full tiers with an ETA.** New `keyframes.py` and `tiles.py`; Preview and Full in the app as one "Build" with a "Refine to full quality" follow-up, not modes. | This is what keeps a 1-minute event at minutes instead of an hour; tiling bounds memory (OpenSplat cites ~2 GB per 1M gaussians) | in-repo | n/a |
| **4** | **MapAnything as the backbone, done right.** Always feed HW3 K + telemetry poses + metric flag; 16-view windows that mix cameras and times; use its confidence to filter; voxel-downsample to init.ply per tile; optionally emit per-view depth for a depth prior. Optional personal-use toggle for the CC BY-NC MapAnything weights (better accuracy per authors). | Posed MapAnything was within 1 to 8 % scale on every Tesla camera in Ryan's tests; monocular models were off 15 to 70 % | huggingface.co/mlx-community/map-anything-apache-fp16 (+ mlx-community/map-anything-fp16 for the toggle) | Apache-2.0 (default), CC BY-NC (toggle) |
| **5** | **Photometric + timing fixes and a benchmark harness.** (a) Per-camera color normalization (fit an affine RGB per camera and per segment against the front camera on overlapping road pixels) before training. (b) A per-camera time-offset table applied when pairing frames to SEI poses; skip frames with high yaw rate (rolling shutter). (c) `dashsplat bench <event>`: held-out PSNR/SSIM, stage timings, splat count, written to JSON, run in CI on the fixture and manually on one real event to pick Brush vs msplat vs mpsplat. | MCMC and appearance terms are the biggest measured dB gains; rolling shutter / offsets are real on Tesla rigs; we have no measured numbers yet | Ideas from gsplat bilateral grid/PPISP and Ryan's calib report; mpsplat as the gsplat-API option if exposure/pose optimization is needed | Apache-2.0 sources |

Next after v1 (v1.1+):
- Movers as rigid per-track splat sets (OmniRe concept) using the #2 tracks; render them along their tracks in the viewer.
- Try mpsplat for gsplat features (bilateral grid, pose optimization, 3DGUT fisheye/OpenCV distortion) once its MPS speed is measured on our data.
- Front-camera windshield reflection layer (RIDE-Splat idea).
- Read Ryan's dashsplat stage reports before building each stage; its code is Apache-2.0, so pieces can be reused with attribution (NOTICE). Its model choices include non-commercial weights; swap to the permissive ones listed in his THIRD_PARTY.md.
- Doc fix: ARCHITECTURE.md mpsplat license line (Apache-2.0, not AGPL).
- UX: detect encrypted clips or missing SEI and tell the user to decrypt at dashcam.tesla.com or in the car (per Ryan's README).

---

## 4. License summary (checked 2026-11-23)

Permissive (OK to bundle): Brush Apache-2.0; msplat Apache-2.0; mpsplat Apache-2.0; metal-gauss MIT; MetalSplat MIT; gsplat Apache-2.0; PPISP Apache-2.0; MapAnything code Apache-2.0 and `map-anything-apache` weights Apache-2.0; DA3 code Apache-2.0 (Small/Base weights Apache-2.0); SAM 2 Apache-2.0; RF-DETR Apache-2.0; skyseg-coreml MIT; fotometis-sky MIT; MLX MIT; DriveStudio MIT; SplatAD Apache-2.0; RigRecon Apache-2.0; Ryan's dashsplat code Apache-2.0.
Copyleft: OpenSplat AGPL-3.0; gsplat-mps AGPL-3.0; ultralytics AGPL-3.0.
Non-commercial: MapAnything default weights CC BY-NC 4.0; DA3-LARGE-1.1 (conflicting, treat as NC); Mask2Former Cityscapes weights (Cityscapes terms); Difix (NVIDIA non-commercial); VGGT and DGGT checkpoints (see 11-22 doc).

## 5. Sources

- https://github.com/R44VC0RP/dashsplat ; /blob/main/THIRD_PARTY.md ; /blob/main/pipeline/run/README.md ; /blob/main/pipeline/depth/REPORT.md ; /blob/main/pipeline/calib/REPORT.md ; /blob/main/research/03-offline-high-accuracy.md ; /blob/main/research/02-feedforward-and-online.md
- https://www.reddit.com/r/GaussianSplatting/comments/1x11vmc/i_did_reconstruct_a_street_in_3d_using_only/
- https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/25/2026/
- https://arxiv.org/abs/2405.17705 ; https://github.com/ridesplat/ride-splat
- https://github.com/rayanht/msplat ; https://pypi.org/project/msplat ; https://news.ycombinator.com/item?id=47262641
- https://github.com/ArthurBrussee/brush ; https://github.com/ArthurBrussee/brush/releases/ ; https://github.com/ArthurBrussee/brush/commits/main ; https://huggingface.co/Merlef/brush-cli-builds (unofficial build, CLI example)
- https://github.com/zacsmms/mpsplat
- https://github.com/nandometzger/metal-gauss
- https://github.com/tchauffi/metalsplat
- https://github.com/WebODM/OpenSplat/ ; https://github.com/WebODM/OpenSplat/pull/76 ; https://radiancefields.com/opensplat-adds-mac-gpu-acceleration ; https://github.com/iffyloop/gsplat-mps
- https://labs.laan.com/blogs/gaussian-splat-training-on-mac-with-metal
- https://radiancefields.com/i-tried-to-make-gaussian-splatting-1-000%C3%97-faster
- https://github.com/nerfstudio-project/gsplat/blob/main/examples/AV_TRAINER.md ; https://raw.githubusercontent.com/nerfstudio-project/gsplat/refs/heads/main/examples/simple_trainer.py ; https://github.com/nv-tlabs/ppisp ; https://github.com/carlinds/splatad
- https://huggingface.co/mlx-community/map-anything-apache-fp16 ; https://github.com/facebookresearch/map-anything
- https://huggingface.co/melmass/skyseg-coreml ; https://huggingface.co/twistlabs/fotometis-sky ; https://huggingface.co/apple/coreml-sam2-tiny ; https://github.com/facebookresearch/sam2 ; https://github.com/roboflow/rf-detr ; https://github.com/ultralytics/ultralytics
- https://github.com/ziyc/drivestudio ; https://github.com/waleedsworld/open-revisualizer ; https://arxiv.org/abs/2512.08498 ; https://github.com/zju3dv/RigRecon

Caveats: no stage was timed on Anthony's Mac; the Mac model is not known to me. Brush's native-Metal speed (merged 2026-11-21) has no published Mac street-scene number yet. msplat mask support is not documented. skyseg-coreml and RF-DETR were not run. The On-the-fly rig paper's hardware for its "2 minutes" claim was not verified.
